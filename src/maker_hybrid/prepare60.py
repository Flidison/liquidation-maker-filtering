"""Prepare exact60s baseline groups on the held-out evaluation periods."""
from .prepare import *
from .regression import POLICY_START

if __name__=='__main__':
    for symbol in SYMBOLS:
        arrays=liquidation_arrays(pl.read_parquet(source(symbol,'binance_liquidations')),pl.read_parquet(source(symbol,'bybit_liquidations')))
        b=bounds(source(symbol,'binance_booktickers'))
        for day in range(POLICY_START,END,86400):prepare_day(symbol,day,arrays,b,window=60)
