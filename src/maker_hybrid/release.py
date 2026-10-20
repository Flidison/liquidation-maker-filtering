"""Export frozen inference configurations, parameters, and reviewable artifacts."""
from .core import *
from .features import feature_columns
import joblib


def run():
    p=json.loads((OUT/'budget_policies.json').read_text())
    b=json.loads((OUT/'baseline60_policies.json').read_text())
    def select(policies,model,h):
        return next(x for x in policies if x['model']==model and x['tau']==h and x['scope']=='combined' and x['calibration_period']=='validation' and x['budget_per_day']==250000000)
    cfg={'version':2,'selection_reason':'Observable E7 research hybrid; no confirmed incremental advantage over E3 or flow-ranked60s baseline',
        'fallback':'fixed keep-all for missing model/config/BBO or unsupported ticker',
        'threshold_calibration':'February scores and clipped weights only; no markout values',
        'policy_key':'injective symbol/time/sign/alignment encoding, uint64 bijective mixing',
        'features':{k:feature_columns(k) for k in ['LR','E2','E3','E4','E5','E6','E7']},
        'policies':{str(h):{'combined':select(p,'E7',h)} for h in HORIZONS}}
    save_json(OUT/'hybrid_config.json',cfg)
    base={'version':2,'selection_reason':'Retain the simpler60s flow-ranked rule: hybrid incremental benefit is unconfirmed and its matched-budget point Scores are lower. This is a retrospective selection, not an independently confirmed winner across all candidates.',
          'fallback':'fixed keep-all for missing configuration or unsupported ticker; this rule requires no fitted model',
          'policies':{str(h):{'combined':select(b,'E1_60_strength',h)} for h in HORIZONS}}
    save_json(OUT/'submission_config.json',base)
    save_json(OUT/'keep_with60_ranked_config.json',base)
    save_json(OUT/'keep_with60_config.json',{'policies':{str(h):{'combined':{'model':'E1_60','native':True,'window':60}} for h in HORIZONS}})
    rows=[];parameters={}
    for side in ['buy','sell']:
        m=joblib.load(OUT/'models'/f'lr_{side}_final.joblib');sc=m['standardscaler'];lr=m['logisticregression']
        for c,beta,scale,mean in zip(feature_columns('LR'),lr.coef_[0],sc.scale_,sc.mean_):
            rows.append({'side':side,'feature':c,'coefficient_per_scaled_feature':float(beta),'coefficient_raw_feature':float(beta/scale),'scaler_mean':float(mean),'scaler_scale':float(scale)})
        parameters[side]={'parameters':lr.get_params(),'intercept_scaled':float(lr.intercept_[0]),'intercept_raw':float(lr.intercept_[0]-np.sum(lr.coef_[0]*sc.mean_/sc.scale_)),'converged_iterations':lr.n_iter_.tolist()}
    pd.DataFrame(rows).to_csv(OUT/'lr_coefficients.csv',index=False);save_json(OUT/'lr_parameters.json',parameters)
    params={}
    for k in ['E3','E4','E5','E6','E7']:
        for h in HORIZONS:params[f'{k}_{h}']=joblib.load(OUT/'models'/f'{k}_{h}_final.joblib').get_params()
    save_json(OUT/'regressor_parameters.json',params)
    records=[]
    for k,status in [('E0','full exact keep-all'),('E1_60','documented60s rule reconstructed; raw-price cleaning discrepancy disclosed'),
                     ('E1_60_strength','source-informed fixed flow ranking; additional baseline'),('E2','new Ridge; historical fitted artifact unavailable'),
                     ('E3','new LightGBM; historical V4 fitted artifact unavailable'),('E4','same LightGBM without liquidation features'),
                     ('E5','E3 plus chronological OOF LR probabilities'),('E6','E3 plus observable AS-inspired risk features'),('E7','E3 plus both feature sets'),
                     ('E8','not identifiable: no own inventory/queue/fills/k'),('Huber/Ranker/ensemble','V4 slides found; exact trained artifacts/code unavailable')]:records.append({'model':k,'status':status})
    pd.DataFrame(records).to_csv(OUT/'experiment_matrix.csv',index=False)
    save_json(OUT/'preprocessing_state.json',{'timestamp_unit':'int64 UTC microseconds','bybit_delay_us':200000,
        'features_clock':'strictly before integer-second floor of current trade timestamp',
        'alignment':'strictly preceding events in [t-window,t), both exchanges, signed price*amount',
        'hybrid_alignment_window_s':5,'baseline_alignment_window_s':60,'AS_risk_horizon_s':30,
        'feature_missing_values':'zero plus stale/age indicators, no fitted imputation',
        'inference_history':'1205s liquidation baseline and61s trade/BBO context for hybrid',
        'sorting':'stable chronological; outputs restore original row positions',
        'boundary_key':'uint64 injective symbol/second/side/alignment namespace'})


if __name__=='__main__':run()
