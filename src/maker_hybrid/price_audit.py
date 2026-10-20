"""Causal liquidation-price audit against prior Binance BBO (no row deletion)."""
from .core import *


def run():
    rows=[];details=[]
    for symbol in SYMBOLS:
        q=pl.scan_parquet(OUT/'quotes'/symbol/'*.parquet').select('second','quote_timestamp','bid_price','ask_price').collect().to_pandas().sort_values('second')
        for exchange in ['binance','bybit']:
            f=pd.read_parquet(source(symbol,exchange+'_liquidations'))
            available=f.timestamp.to_numpy(np.int64)+(200000 if exchange=='bybit' else 0)
            j=np.searchsorted(q.second.to_numpy(),available//US,side='right')-1
            jj=np.maximum(j,0)
            mid=(q.bid_price.to_numpy()[jj]+q.ask_price.to_numpy()[jj])/2
            age=(available-q.quote_timestamp.to_numpy()[jj])/US
            dev=(f.price.to_numpy()/mid-1)*10000
            fresh=(j>=0)&(age<=1)&(age>=0)&np.isfinite(mid)
            bad=fresh&(np.abs(dev)>300)
            rows.append({'symbol':symbol,'exchange':exchange,'rows':len(f),'fresh_reference_rows':int(fresh.sum()),
                         'flagged_300bps':int(bad.sum()),'flagged_1000bps':int((fresh&(np.abs(dev)>1000)).sum()),
                         'flagged_raw_notional':float((f.price*f.amount).to_numpy()[bad].sum())})
            x=f.loc[bad].copy();x['available_timestamp']=available[bad];x['deviation_bps']=dev[bad]
            x['reference_quote_age_s']=age[bad];x['symbol']=symbol;x['exchange']=exchange
            details.append(x)
    pd.DataFrame(rows).to_csv(OUT/'liquidation_price_audit.csv',index=False)
    pd.concat(details,ignore_index=True).to_parquet(OUT/'flagged_liquidation_prices.parquet',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':run()
