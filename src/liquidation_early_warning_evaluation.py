"""Frozen-model evaluation, paired day bootstrap, calibration, alert episodes."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import liquidation_early_warning as ew


def ap_cache(y,p,day):
    order=np.argsort(-p,kind='stable')
    p=np.asarray(p)[order]
    starts=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1]
    return np.asarray(y)[order],np.asarray(day)[order],starts


def weighted_ap(cache,weights):
    y,day,starts=cache
    row_weight=weights[day]
    positives=np.add.reduceat(row_weight*y,starts)
    counts=np.add.reduceat(row_weight,starts)
    total=positives.sum()
    if total<=0:return np.nan
    precision=np.cumsum(positives)/np.maximum(np.cumsum(counts),1e-15)
    return float(np.sum(positives*precision)/total)


def analyze(bootstrap_repetitions=400):
    p=pd.read_parquet(ew.OUT/'predictions.parquet')
    ci=[];calibration=[];alerts=[]
    for phase in ('validation','test'):
        for h in ew.HORIZONS:
            d=p[p.split.eq(phase)&p.horizon_s.eq(h)]
            full=d[d.model.eq('full')].sort_values(['timestamp','symbol']).reset_index(drop=True)
            base=d[d.model.eq('without_liquidity_score')].sort_values(['timestamp','symbol']).reset_index(drop=True)
            assert full[['timestamp','symbol','target']].equals(base[['timestamp','symbol','target']])
            dates=pd.to_datetime(full.timestamp,unit='us',utc=True).dt.floor('D')
            day,unique=pd.factorize(dates,sort=True)
            caches=[ap_cache(full.target.to_numpy(),x.probability.to_numpy(),day) for x in (full,base)]
            for c,x in zip(caches,(full,base)):
                assert np.isclose(weighted_ap(c,np.ones(len(unique))),average_precision_score(x.target,x.probability))
            rng=np.random.default_rng(ew.SEED+h+(phase=='test'))
            draws=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=bootstrap_repetitions)
            result=np.array([[weighted_ap(c,w) for c in caches] for w in draws])
            difference=result[:,0]-result[:,1]
            point=[weighted_ap(c,np.ones(len(unique))) for c in caches]
            ci.append({'split':phase,'horizon_s':h,'days':len(unique),'bootstrap_repetitions':bootstrap_repetitions,
                       'full_ap':point[0],'full_ap_ci_low':np.nanquantile(result[:,0],.025),
                       'full_ap_ci_high':np.nanquantile(result[:,0],.975),
                       'ablation_ap':point[1],'ap_difference':point[0]-point[1],
                       'difference_ci_low':np.nanquantile(difference,.025),
                       'difference_ci_high':np.nanquantile(difference,.975)})
            for variant,g in [('full',full),('without_liquidity_score',base)]:
                g=g.copy()
                # Quantile bins within evaluated probabilities, for diagnostics only.
                g['calibration_bin']=pd.qcut(g.probability,10,duplicates='drop')
                c=g.groupby('calibration_bin',observed=True).agg(n=('target','size'),
                    predicted=('probability','mean'),observed=('target','mean')).reset_index(drop=True)
                c['split']=phase;c['horizon_s']=h;c['model']=variant
                calibration.append(c)
                for symbol,z in g[g.alarm].groupby('symbol'):
                    z=z.sort_values('timestamp')
                    # Retrospectively merge adjacent alerts separated by <= horizon.
                    groups=(z.timestamp.diff().fillna(np.inf)>h*1_000_000).cumsum()
                    for _,group in z.groupby(groups):
                        first=group.iloc[0]
                        alerts.append({'split':phase,'horizon_s':h,'model':variant,'symbol':symbol,
                            'first_alert_timestamp':int(first.timestamp),'last_alert_timestamp':int(group.timestamp.iloc[-1]),
                            'dc_alerts':len(group),'first_alert_target':int(first.target),
                            'any_alert_target':int(group.target.max())})
            print('Bootstrap',phase,h,flush=True)
    pd.DataFrame(ci).to_csv(ew.OUT/'paired_day_bootstrap.csv',index=False)
    pd.concat(calibration,ignore_index=True).to_csv(ew.OUT/'probability_calibration.csv',index=False)
    pd.DataFrame(alerts).to_csv(ew.OUT/'alert_episodes.csv',index=False)
    # Availability/attrition on all observed DC points, including warm-up and gaps.
    rows=[]
    for symbol in ew.lb.SYMBOLS:
        f=pd.read_parquet(ew.OUT/f'market_states_{symbol}.parquet')
        for split,g in f[f.is_dc_event].groupby('split'):
            rows.append({'symbol':symbol,'split':split,'all_dc_events':len(g),
                'nuisance_warmup':int((g.timestamp<ew.READY*1_000_000).sum()),
                'history_not_ready_after_warmup':int(((g.timestamp>=ew.READY*1_000_000)&~g.history_ready).sum()),
                'known_burst_with_ready_history':int((g.history_ready&g.burst_known_now).sum()),
                'forecast_eligible':int(g.forecast_eligible.sum())})
    pd.DataFrame(rows).to_csv(ew.OUT/'population_attrition.csv',index=False)


if __name__=='__main__':analyze()
