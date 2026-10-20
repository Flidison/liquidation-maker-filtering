"""Exact sensitivity of native60s keep-with to the flagged liquidation prices.

Only affected trade intervals need relabeling; official markout calculation is
unchanged. This does not establish ML robustness to a different cleaning rule.
"""
from .core import *
from .prepare import bbo_slice,bounds


def run():
    flagged=pd.read_parquet(OUT/'flagged_liquidation_prices.parquet');rows=[]
    for s in SYMBOLS:
        flagged_s=flagged[flagged.symbol.eq(s)]
        raw={ex:pd.read_parquet(source(s,ex+'_liquidations')) for ex in ['binance','bybit']}
        clean={}
        for ex,f in raw.items():
            keys=['timestamp','side','price','amount']
            bad=flagged_s[flagged_s.exchange.eq(ex)][keys].drop_duplicates()
            f=f.merge(bad.assign(_bad=True),on=keys,how='left')
            clean[ex]=f.loc[f._bad.isna()].drop(columns='_bad')
        a=liquidation_arrays(raw['binance'],raw['bybit']);c=liquidation_arrays(clean['binance'],clean['bybit'])
        starts=flagged_s.available_timestamp.to_numpy(np.int64)
        starts=starts[(starts>=VALIDATION*US-60*US)&(starts<END*US)]
        if not len(starts):continue
        affected_days=np.unique(np.r_[starts//(86400*US),(starts+60*US)//(86400*US)])
        for d in affected_days:
            day=int(d*86400)
            if not VALIDATION<=day<END:continue
            relevant=starts[(starts<(day+86400)*US)&(starts+60*US>=day*US)]
            expression=pl.lit(False)
            for t in relevant:expression|=((pl.col('timestamp')>int(t))&(pl.col('timestamp')<=int(t+60*US)))
            f=pl.scan_parquet(source(s,'binance_trades')).filter(expression&(pl.col('timestamp')>=day*US)&(pl.col('timestamp')<(day+86400)*US)).collect().to_pandas()
            sign=np.where(f.side.eq('buy'),1,-1);t=f.timestamp.to_numpy(np.int64)
            before=alignment(a,t,sign,60)==1;after=alignment(c,t,sign,60)==1
            change=after.astype(int)-before.astype(int)
            b=bbo_slice(s,(day-1800)*US,(day+86400+300)*US)
            qt=b['timestamp'].to_numpy();mid=(b['bid_price'].to_numpy()+b['ask_price'].to_numpy())/2
            for h in HORIZONS:
                y,age=maker_labels(t,f.price.to_numpy(),sign,qt,mid,h,bounds(source(s,'binance_booktickers')))
                w=np.where(np.isfinite(y),np.minimum(f.price*f.amount,100000),0)
                wy=w*np.nan_to_num(y)
                rows.append({'symbol':s,'day':pd.Timestamp(day,unit='s',tz='UTC').strftime('%Y-%m-%d'),'tau':h,
                    'affected_trade_rows':len(f),'changed_decisions':int(np.sum(change!=0)),
                    'delta_kept_w':float(np.sum(change*w)),'delta_kept_wy':float(np.sum(change*wy))})
    pd.DataFrame(rows).to_csv(OUT/'price_cleaning_sensitivity_deltas.csv',index=False)
    print(pd.DataFrame(rows).groupby(['symbol','tau']).sum(numeric_only=True).to_string())


if __name__=='__main__':run()
