"""Deterministic artificial feeds. No original market observations are read."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def generate(root=ROOT/'data/sample'):
    rng=np.random.default_rng(20261017)
    start=int(pd.Timestamp('2026-01-01',tz='UTC').timestamp()*1e6)
    seconds=np.arange(1801,dtype=np.int64)
    for sym,base in [('btcusdt',100000.),('ethusdt',3000.)]:
        mid=base*np.exp(np.cumsum(rng.normal(0,0.00005,len(seconds))))
        trades=pd.DataFrame({'timestamp':start+seconds*1_000_000+500_000,
            'ticker':'perp:'+sym,'side':np.where(seconds%2,'buy','sell'),
            'price':mid*(1+np.where(seconds%2,1,-1)*.00001),'amount':rng.uniform(.01,.2,len(seconds))})
        bbo=pd.DataFrame({'timestamp':start+seconds*1_000_000-1,
            'ticker':'perp:'+sym,'bid_price':mid*(1-.00001),'bid_amount':rng.uniform(1,5,len(seconds)),
            'ask_price':mid*(1+.00001),'ask_amount':rng.uniform(1,5,len(seconds))})
        for kind,frame in [('binance_trades',trades),('binance_booktickers',bbo)]:
            p=root/kind/f'perp_{sym}.csv';p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False)
        for exchange in ['binance','bybit']:
            offsets=np.unique(np.r_[np.arange(30,1800,37),np.arange(1200,1220)])
            frame=pd.DataFrame({'timestamp':start+offsets*1_000_000+100_000,
                'ticker':('perp:' if exchange=='binance' else '')+sym,
                'side':np.where(offsets%3,'buy','sell'),'price':mid[offsets],
                'amount':rng.uniform(.5,20,len(offsets))})
            p=root/f'{exchange}_liquidations'/f'{"perp_" if exchange=="binance" else ""}{sym}.csv'
            p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False)


if __name__=='__main__':generate()
