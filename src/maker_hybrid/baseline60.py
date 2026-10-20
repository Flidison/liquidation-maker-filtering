"""Source-backed 60s keep-with reconstruction, added after locating the slides.

V4 fitted artifacts and exact cleaning code were not supplied. The rule uses
both exchanges, raw price*amount, nonzero net flow, no count floor. It preserves
the official scorer. Budget thinning is by stable random keys, not a hindsight
ranking of liquidation sizes.
"""
from .core import *
from .regression import POLICY_START
from .policy import calibrate,ranked_groups,liq_arrays
from .evaluate import daily_stats,summarize_daily,BUDGETS


def run():
    daily=[];policies=[]
    if (OUT/'baseline60_report.csv').exists():
        if 'E1_60_strength' in pd.read_csv(OUT/'baseline60_report.csv').model.unique():return
    for period,start,end in [('train_holdout',POLICY_START,VALIDATION),('validation',VALIDATION,END)]:
        days=(end-start)//86400
        frames=[]
        for symbol in SYMBOLS:
            for day in range(start,end,86400):
                name=pd.Timestamp(day,unit='s',tz='UTC').strftime('%Y-%m-%d')+'.parquet'
                f=pd.read_parquet(OUT/'groups60'/symbol/name)
                f['symbol']=symbol;f['tie_key']=policy_hash(f.second,symbol,f.taker_sign,f.alignment)
                w=liquidation_window(liq_arrays(symbol),f.second.to_numpy()*US,60)
                strength=np.abs(sum(v[1]*(1 if k.endswith('_buy') else -1) for k,v in w.items())).astype('float32')
                for h in HORIZONS:f[f'E1_60_strength_{h}']=strength
                if period=='train_holdout':f=f[f.second+301<VALIDATION]
                frames.append(f)
        allf=pd.concat(frames,ignore_index=True)
        for scope in [*SYMBOLS,'combined']:
            f=(allf if scope=='combined' else allf[allf.symbol.eq(scope)]).reset_index(drop=True)
            for h in HORIZONS:
                native={'model':'E1_60_native','scope':scope,'tau':h,'budget_per_day':-1.,'calibration_period':period}
                daily.append(daily_stats(f,h,np.flatnonzero(f.alignment.eq(1)),None,start,days,native))
                for model in ['E1_60','E1_60_strength']:
                    rank=ranked_groups(f,model,h)
                    for budget in BUDGETS:
                        p,whole,partial=calibrate(f,model,h,budget,days,rank)
                        p.update(scope=scope,calibration_period=period);policies.append(p)
                        daily.append(daily_stats(f,h,whole,partial,start,days,p))
                print('60s baseline',period,scope,h,flush=True)
    d=pd.concat(daily,ignore_index=True)
    d.to_csv(OUT/'baseline60_daily.csv',index=False)
    summarize_daily(d).to_csv(OUT/'baseline60_report.csv',index=False)
    save_json(OUT/'baseline60_policies.json',policies)


if __name__=='__main__':run()
