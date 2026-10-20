"""Apply Jan22-31 thresholds to February final-model scores, without retuning."""
from .core import *
from .evaluate import load_period,daily_stats,summarize_daily
from .policy import full_group_keep


def run():
    policies=json.loads((OUT/'budget_policies.json').read_text());daily=[]
    for h in HORIZONS:
        allf=load_period('final',h)
        for p in policies:
            if p['calibration_period']!='train_holdout' or p['tau']!=h or p['model'] not in ['E2','E3','E7'] or p['budget_per_day'] not in [750000,10000000,250000000]:continue
            f=(allf if p['scope']=='combined' else allf[allf.symbol.eq(p['scope'])]).reset_index(drop=True)
            keep,boundary=full_group_keep(f[f"{p['model']}_{h}"],f.tie_key,f.alignment,p)
            assert not boundary.any(), 'Unexpected cross-period tie collision'
            config=p|{'calibration_period':'January_threshold_on_February'}
            daily.append(daily_stats(f,h,np.flatnonzero(keep),None,VALIDATION,28,config))
        print('Threshold transfer',h,flush=True)
    d=pd.concat(daily,ignore_index=True);d.to_csv(OUT/'threshold_transfer_daily.csv',index=False)
    summarize_daily(d).to_csv(OUT/'threshold_transfer_report.csv',index=False)


if __name__=='__main__':run()
