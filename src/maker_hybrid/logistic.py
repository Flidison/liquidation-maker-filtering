"""Chronological out-of-fold side-specific burst probabilities for stacking."""
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, average_precision_score
from threadpoolctl import threadpool_limits
from .core import *
from .features import feature_columns

FOLDS = ['2025-12-22','2025-12-29','2026-01-05','2026-01-12','2026-01-19','2026-01-26','2026-02-01']


def burst_onsets(symbol, side):
    frames = [pd.read_parquet(ROOT/'artifacts/results/bursts'/f'features_{symbol}_{ex}_{side}.parquet',
                             columns=['timestamp','burst']) for ex in ['binance','bybit']]
    np.testing.assert_array_equal(frames[0].timestamp, frames[1].timestamp)
    active = frames[0].burst.to_numpy() | frames[1].burst.to_numpy()
    first = active & ~np.r_[False,active[:-1]]
    return frames[0].timestamp.to_numpy()[first]//US - 5


def future_burst(seconds, onsets):
    t=np.asarray(seconds)
    after=np.searchsorted(onsets,t,side='right')
    through=np.searchsorted(onsets,t+5,side='right')
    return (through>after).astype(np.int8)


def load_train():
    frames=[]
    for symbol in SYMBOLS:
        events={side:burst_onsets(symbol,side) for side in ['buy','sell']}
        for p in sorted((OUT/'states'/symbol).glob('*.parquet')):
            day=int(pd.Timestamp(p.stem,tz='UTC').timestamp())
            if day<READY or day>=VALIDATION:continue
            f=pd.read_parquet(p)
            take=(f.second.to_numpy()%5==0) & (uniform_hash(f.second,symbol)<.2)
            f=f.loc[take].copy()
            for side in events:f['y_'+side]=future_burst(f.second,events[side])
            frames.append(f)
    return pd.concat(frames,ignore_index=True).sort_values('second',kind='stable').reset_index(drop=True)


def fit_models(frame, boundary, tag):
    eligible=frame.second+10<boundary
    f=frame.loc[eligible]
    cols=feature_columns('LR')
    saved=OUT/'models'/f'lr_{tag}.json'
    if saved.exists() and all((OUT/'models'/f'lr_{s}_{tag}.joblib').exists() for s in ['buy','sell']):
        meta=json.loads(saved.read_text())
        assert meta['boundary']==boundary and meta['rows']==len(f)
        assert meta['last_training_second']==int(f.second.max()) and meta['features']==cols
        print('Reusing fixed LR models',tag,flush=True)
        return {s:joblib.load(OUT/'models'/f'lr_{s}_{tag}.joblib') for s in ['buy','sell']}
    models={}
    for side in ['buy','sell']:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver='lbfgs',
                             max_iter=2000,tol=1e-6,random_state=SEED))
        with threadpool_limits(limits=4):model.fit(f[cols],f['y_'+side])
        lr=model[-1]
        if lr.n_iter_[0]>=2000:raise RuntimeError('LR did not converge')
        joblib.dump(model,OUT/'models'/f'lr_{side}_{tag}.joblib')
        models[side]=model
        print('LR',tag,side,'rows',len(f),'positives',int(f['y_'+side].sum()),'iterations',lr.n_iter_[0],flush=True)
    save_json(OUT/'models'/f'lr_{tag}.json',{'boundary':boundary,'last_training_second':int(f.second.max()),
        'last_label_available_before':int(f.second.max()+10),'rows':len(f),'features':cols,
        'train_prior':{s:float(f['y_'+s].mean()) for s in ['buy','sell']},
        'training_period':'December/January only; expanding temporal folds, no future labels'})
    return models


def run():
    if (OUT/'lr_complete.json').exists():return
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    frame=load_train()
    frame.to_parquet(OUT/'lr_training_sample.parquet',index=False)
    cols=feature_columns('LR')
    folds=[int(pd.Timestamp(d,tz='UTC').timestamp()) for d in FOLDS]
    for i,boundary in enumerate(folds):
        tag='final' if boundary==VALIDATION else FOLDS[i]
        models=fit_models(frame,boundary,tag)
        end=folds[i+1] if i+1<len(folds) else END
        for symbol in SYMBOLS:
            folder=OUT/'probabilities'/symbol;folder.mkdir(parents=True,exist_ok=True)
            for day in range(boundary,end,86400):
                name=pd.Timestamp(day,unit='s',tz='UTC').strftime('%Y-%m-%d')+'.parquet'
                receipt=(folder/name).with_suffix('.json')
                if receipt.exists():
                    previous=json.loads(receipt.read_text())
                    assert previous['model_tag']==tag and previous['probabilities_sha256']==digest(folder/name)
                    continue
                states=pd.read_parquet(OUT/'states'/symbol/name)
                p=pd.DataFrame({'second':states.second})
                for side,model in models.items():
                    with threadpool_limits(limits=4):p[f'p_{side}_burst_5s']=model.predict_proba(states[cols])[:,1]
                p.to_parquet(folder/name,index=False)
                save_json((folder/name).with_suffix('.json'),{'model_tag':tag,'training_boundary':boundary,
                    'latest_label_maturity_before':boundary,'prediction_start':day,'prediction_end':day+86400,
                    'probabilities_sha256':digest(folder/name),
                    'models':{side:digest(OUT/'models'/f'lr_{side}_{tag}.joblib') for side in models}})
        print('OOF prediction block complete',tag,flush=True)
    diagnostics()
    save_json(OUT/'lr_complete.json',{'folds':FOLDS,'warmup_before':'2025-12-22',
              'side_targets':['buy','sell'],'n_training_sample':len(frame),'new_calibrator':False})


def diagnostics():
    rows=[];bins=[];outputs=[]
    prior=json.loads((OUT/'models/lr_final.json').read_text())['train_prior']
    for symbol in SYMBOLS:
        events={s:burst_onsets(symbol,s) for s in ['buy','sell']}
        for path in sorted((OUT/'probabilities'/symbol).glob('2026-02-*.parquet')):
            p=pd.read_parquet(path)
            p=p[(p.second%5==0)&(p.second+10<END)].copy()
            p['symbol']=symbol
            for side in events:p['y_'+side]=future_burst(p.second,events[side])
            outputs.append(p)
    allp=pd.concat(outputs,ignore_index=True)
    allp.to_parquet(OUT/'lr_validation_probabilities.parquet',index=False)
    for symbol in [*SYMBOLS,'combined']:
        f=allp if symbol=='combined' else allp[allp.symbol.eq(symbol)]
        for side in ['buy','sell']:
            y=f['y_'+side].to_numpy();p=f[f'p_{side}_burst_5s'].to_numpy()
            cuts=np.unique(np.quantile(p,np.arange(1,10)/10))
            g=pd.DataFrame({'bin':np.searchsorted(cuts,p,side='right'),'y':y,'p':p}).groupby('bin').agg(
                n=('y','size'),predicted=('p','mean'),observed=('y','mean')).reset_index()
            g['symbol']=symbol;g['side']=side;bins.append(g)
            rows.append({'symbol':symbol,'side':side,'n':len(f),'positives':int(y.sum()),
                'prevalence':y.mean(),'brier':np.mean((p-y)**2),'constant_brier':np.mean((prior[side]-y)**2),
                'log_loss':log_loss(y,p,labels=[0,1]),'pr_auc_average_precision':average_precision_score(y,p),
                'ece':np.sum(g.n*np.abs(g.predicted-g.observed))/len(f)})
    pd.DataFrame(rows).to_csv(OUT/'lr_validation_metrics.csv',index=False)
    pd.concat(bins).to_csv(OUT/'lr_calibration_bins.csv',index=False)


if __name__=='__main__':run()
