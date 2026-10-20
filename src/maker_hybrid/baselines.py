"""Reconstruct the full-population official keep-all/keep-with baselines first."""
from .core import *


def totals_to_metrics(a, number_of_days):
    all_w, kept_w = a['all_w'],a['kept_w']
    all_pnl=a['all_wy']/all_w if all_w else np.nan
    kept_pnl=a['kept_wy']/kept_w if kept_w else np.nan
    return {'score':kept_pnl-all_pnl,'pnl_all':all_pnl,'pnl_kept':kept_pnl,
        'pnl_filtered':(a['all_wy']-a['kept_wy'])/(all_w-kept_w) if all_w>kept_w else np.nan,
        'kept_turnover_per_day':kept_w/number_of_days,
        'kept_fraction':kept_w/all_w if all_w else np.nan,
        'fraction_trades_retained':a['kept_n']/a['all_n'] if a['all_n'] else np.nan,
        'total_weighted_marked_pnl_usd':a['kept_wy']/10000,
        'official_turnover_constraint':kept_w/number_of_days>=500000}


def run():
    rows=[];gaps=[]
    for symbol in SYMBOLS:
        paths=sorted((OUT/'groups'/symbol).glob('*.parquet'))
        if len(paths)!=90:raise RuntimeError(f'{symbol}: need all 90 prepared days')
        for path in paths:
            f=pd.read_parquet(path)
            split='validation' if path.stem>='2026-02-01' else 'train'
            for h in HORIZONS:
                w,wy,n=f[f'w_{h}'],f[f'wy_{h}'],f[f'n_{h}']
                gaps.append({'symbol':symbol,'day':path.stem,'split':split,'tau':h,
                    'valid_rows':int(n.sum()),'all_rows':int(f.trade_count.sum()),
                    'weighted_turnover':float(w.sum()),
                    'stale_quote_rows_gt2s':int(f[f'stale_n_{h}'].sum()),
                    'stale_quote_weight_gt2s':float(w.sum()-f[f'fresh_w_{h}'].sum()),
                    'stale_quote_weight_gt10s':float(f[f'stale10_w_{h}'].sum())})
                for name,mask in [('E0',np.ones(len(f),bool)),('E1',f.alignment.eq(1)),
                                  ('against',f.alignment.eq(-1)),('neutral',f.alignment.eq(0))]:
                    a={'all_w':float(w.sum()),'all_wy':float(wy.sum()),'all_n':int(n.sum()),
                       'kept_w':float(w[mask].sum()),'kept_wy':float(wy[mask].sum()),'kept_n':int(n[mask].sum())}
                    rows.append({'model':name,'symbol':symbol,'day':path.stem,'split':split,'tau':h,
                                 **a,**totals_to_metrics(a,1)})
    daily=pd.DataFrame(rows)
    sums=['all_w','all_wy','all_n','kept_w','kept_wy','kept_n']
    combined=daily.groupby(['model','day','split','tau'])[sums].sum().reset_index()
    combined['symbol']='combined'
    daily=pd.concat([daily,combined],ignore_index=True)
    for i,row in daily.loc[daily.symbol.eq('combined')].iterrows():
        for k,v in totals_to_metrics(row,1).items():daily.loc[i,k]=v
    summary=[]
    for (model,symbol,split,h),f in daily.groupby(['model','symbol','split','tau']):
        a=f[sums].sum().to_dict();days=f.day.nunique()
        summary.append({'model':model,'symbol':symbol,'split':split,'tau':h,'days':days,
            **totals_to_metrics(a,days),'minimum_daily_kept_turnover':f.kept_w.min(),
            'fraction_days_below_500k':float((f.kept_w<500000).mean())})
    daily.to_csv(OUT/'baseline_daily.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'baseline_summary.csv',index=False)
    pd.DataFrame(gaps).to_csv(OUT/'bbo_gap_daily.csv',index=False)
    print(pd.DataFrame(summary).query("split=='validation' and model in ['E0','E1']")
        [['model','symbol','tau','score','pnl_kept','kept_turnover_per_day','kept_fraction']].to_string(index=False))


if __name__=='__main__':run()
