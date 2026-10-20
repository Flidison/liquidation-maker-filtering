"""Fixed-architecture maker regressors with controlled LR/AS feature ablations."""
import joblib
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .core import *
from .features import feature_columns, trade_features

KINDS = ['E2','E3','E4','E5','E6','E7']
WARMUP = int(pd.Timestamp('2025-12-22',tz='UTC').timestamp())
POLICY_START = int(pd.Timestamp('2026-01-22',tz='UTC').timestamp())


def sample_training():
    frames=[]
    for symbol in SYMBOLS:
        for path in sorted((OUT/'groups'/symbol).glob('*.parquet')):
            day=int(pd.Timestamp(path.stem,tz='UTC').timestamp())
            if day<WARMUP or day>=VALIDATION:continue
            g=pd.read_parquet(path)
            choose=uniform_hash(g.second,symbol,g.taker_sign,g.alignment)<1/16
            g=g.loc[choose & (g.second+301<VALIDATION)&(g.w_300>0)].copy()
            state=pd.read_parquet(OUT/'states'/symbol/path.name)
            prob=pd.read_parquet(OUT/'probabilities'/symbol/path.name)
            x=trade_features(g,state,prob)
            assert not x[feature_columns('E7')].isna().any().any()
            for h in HORIZONS:
                x[f'target_{h}']=g[f'wy_{h}'].to_numpy()/g[f'w_{h}'].to_numpy()
                x[f'weight_{h}']=g[f'w_{h}'].to_numpy()*16
            x['symbol']=symbol
            frames.append(x)
    data=pd.concat(frames,ignore_index=True).sort_values('second',kind='stable').reset_index(drop=True)
    data.to_parquet(OUT/'regression_training_sample.parquet',index=False)
    return data


def train():
    marker=OUT/'regressors_complete.json'
    if marker.exists():return
    if not (OUT/'baseline_summary.csv').exists():
        raise RuntimeError('Reproduce keep-all/keep-with before training the new maker regressors')
    if not (OUT/'lr_complete.json').exists():raise RuntimeError('Need chronological OOF probabilities first')
    data=sample_training()
    spec=json.loads((OUT/'experiment_specification.json').read_text())
    save_json(OUT/'regime_cutoffs.json',{s:np.quantile(data.loc[data.symbol.eq(s),'rv_20s_bps'],[.5,.9]).tolist() for s in SYMBOLS})
    configs=[];importance=[]
    for tag,boundary in [('holdout',POLICY_START),('final',VALIDATION)]:
        rows=data.second+301<boundary
        for h in HORIZONS:
            y=data.loc[rows,f'target_{h}']
            w=data.loc[rows,f'weight_{h}'].to_numpy()
            w=w/w.mean()
            for kind in KINDS:
                cols=feature_columns(kind)
                x=data.loc[rows,cols].astype('float32')
                if kind=='E2':
                    model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
                    with threadpool_limits(limits=4):model.fit(x,y,ridge__sample_weight=w)
                else:
                    model=LGBMRegressor(**spec['LGBMRegressor'],random_state=SEED,
                        deterministic=True,force_col_wise=True,verbosity=-1)
                    model.fit(x,y,sample_weight=w)
                filename=f'{kind}_{h}_{tag}.joblib'
                joblib.dump(model,OUT/'models'/filename)
                configs.append({'kind':kind,'tau':h,'tag':tag,'training_rows':int(rows.sum()),
                    'last_training_second':int(data.loc[rows,'second'].max()),
                    'last_label_available_before':int(data.loc[rows,'second'].max()+301),
                    'boundary':boundary,'features':cols,'model_file':filename,
                    'sha256':digest(OUT/'models'/filename)})
                if kind!='E2' and tag=='final':
                    importance.extend({'kind':kind,'tau':h,'feature':c,'gain':float(g)}
                        for c,g in zip(cols,model.booster_.feature_importance('gain')))
                print('Regressor',tag,kind,h,'rows',rows.sum(),'features',len(cols),flush=True)
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False)
    save_json(marker,{'models':configs,'specification_sha256':digest(OUT/'experiment_specification.json'),
        'training_sample_sha256':digest(OUT/'regression_training_sample.parquet'),
        'weighted_group_means':'Exactly equivalent squared-loss target for identical features, before uniform group sampling',
        'inverse_sampling_weight':16,'weight_normalization':'mean 1 within training fit',
        'validation_labels_used_for_model_fit':False})


def predict():
    manifest=json.loads((OUT/'regressors_complete.json').read_text())
    for tag,start,end in [('holdout',POLICY_START,VALIDATION),('final',VALIDATION,END)]:
        models={(k,h):joblib.load(OUT/'models'/f'{k}_{h}_{tag}.joblib') for k in KINDS for h in HORIZONS}
        for symbol in SYMBOLS:
            folder=OUT/'predictions'/tag/symbol;folder.mkdir(parents=True,exist_ok=True)
            for day in range(start,end,86400):
                name=pd.Timestamp(day,unit='s',tz='UTC').strftime('%Y-%m-%d')+'.parquet'
                path=folder/name
                if path.exists():continue
                groups=pd.read_parquet(OUT/'groups'/symbol/name,columns=['second','taker_sign','alignment'])
                states=pd.read_parquet(OUT/'states'/symbol/name)
                probabilities=pd.read_parquet(OUT/'probabilities'/symbol/name)
                x=trade_features(groups,states,probabilities)
                p=groups.copy()
                p['tie_key']=policy_hash(p.second,symbol,p.taker_sign,p.alignment)
                p['feature_quote_age_upper_s']=(np.expm1(x.log_quote_age)+1).astype('float32')
                cuts=json.loads((OUT/'regime_cutoffs.json').read_text())[symbol]
                p['volatility_regime']=np.searchsorted(cuts,x.rv_20s_bps,side='right').astype('int8')
                for kind in KINDS:
                    xx=x[feature_columns(kind)].astype('float32')
                    for h in HORIZONS:
                        with threadpool_limits(limits=4):
                            p[f'{kind}_{h}']=models[kind,h].predict(xx).astype('float32')
                p.to_parquet(path,index=False)
                print('Predictions',tag,symbol,name,len(p),flush=True)


if __name__=='__main__':train();predict()
