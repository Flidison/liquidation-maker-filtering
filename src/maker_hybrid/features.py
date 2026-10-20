"""One-second held market state; all input timestamps strictly precede the boundary.

The exact trade-time liquidation alignment is added separately. No target or
future-midpoint column can enter the feature schema.
"""
import argparse
import numpy as np
import pandas as pd
import polars as pl
from .core import *

WINDOWS = (1, 5, 10, 20, 60)
AS_FEATURES = ['as_return_variance_per_s','as_price_variance_per_s',
               'as_volatility_30s_bps','as_spread_over_volatility',
               'as_maker_signed_volatility','as_maker_signed_spread_volatility']


def trailing(values, width, include_current=False):
    values = np.asarray(values, float)
    cs = np.r_[0., np.cumsum(values)]
    end = np.arange(len(values)) + int(include_current)
    start = np.maximum(end-width, 0)
    return cs[end]-cs[start]


def state_features(seconds, quotes, flows, arrays, symbol):
    """Inputs cover a contiguous second grid (at least 60 seconds of context)."""
    seconds = np.asarray(seconds, np.int64)
    assert np.all(np.diff(seconds) == 1)
    q = quotes.set_index('second').reindex(seconds)
    qt = q.quote_timestamp.to_numpy()
    age = np.where(qt >= 0, (seconds*US-qt)/US, 1e6)
    bid, ask = q.bid_price.to_numpy(), q.ask_price.to_numpy()
    mid = (bid+ask)/2
    qb, qa = q.bid_amount.to_numpy(), q.ask_amount.to_numpy()
    valid = np.isfinite(mid) & (mid > 0) & (age <= 2)
    features = {'symbol_eth':np.full(len(seconds), int(symbol == 'ethusdt')),
                'quote_stale':(~valid).astype(int), 'log_quote_age':np.log1p(np.clip(age,0,1e6)),
                'spread_bps':np.where(valid,(ask-bid)/mid*10000,0),
                'book_imbalance':np.divide(qb-qa,qb+qa,out=np.zeros(len(seconds)),
                                          where=valid & (qb+qa>0))}
    returns = np.zeros(len(seconds))
    good = valid[1:] & valid[:-1]
    returns[1:] = np.where(good,np.log(mid[1:]/mid[:-1]),0)
    r2 = returns**2
    features['bad_quote_seconds_60s'] = trailing(~valid,60,True)
    for w in [5,10,20,60]:
        prior = np.maximum(np.arange(len(mid))-w,0)
        complete = (np.arange(len(mid))>=w) & valid & valid[prior]
        features[f'return_{w}s_bps'] = np.where(complete,np.log(mid/mid[prior])*10000,0)
        if w != 60:
            features[f'rv_{w}s_bps'] = np.sqrt(trailing(r2,w,True))*10000
    # A rolling variance proxy, not a fitted conditional-volatility forecast.
    var = trailing(r2,60,True)/60
    features['as_return_variance_per_s'] = var
    features['as_price_variance_per_s'] = np.where(valid,mid**2*var,0)
    features['as_volatility_30s_bps'] = np.sqrt(var*30)*10000
    features['as_spread_over_volatility'] = features['spread_bps']/np.maximum(features['as_volatility_30s_bps'],.01)
    f = flows.groupby(['second','taker_sign'])[['trade_count','notional']].sum()
    for sign,label in [(1,'buy'),(-1,'sell')]:
        ff = f.xs(sign,level='taker_sign') if sign in f.index.get_level_values('taker_sign') else pd.DataFrame(columns=['trade_count','notional'])
        ff = ff.reindex(seconds).fillna(0)
        for w in [5,20,60]:
            features[f'trade_{label}_log_count_{w}s'] = np.log1p(trailing(ff.trade_count,w))
            features[f'trade_{label}_log_notional_{w}s'] = np.log1p(trailing(ff.notional,w))
    features['trade_flow_imbalance_20s'] = (
        np.expm1(features['trade_buy_log_notional_20s'])-np.expm1(features['trade_sell_log_notional_20s']))/np.maximum(
        np.expm1(features['trade_buy_log_notional_20s'])+np.expm1(features['trade_sell_log_notional_20s']),1)
    windows = {w:liquidation_window(arrays,seconds*US,w) for w in WINDOWS}
    base = liquidation_window(arrays,seconds*US-5*US,1200)
    for name,(ts,_) in arrays.items():
        j = np.searchsorted(ts,seconds*US,side='left')-1
        elapsed = np.where(j>=0,(seconds*US-ts[np.maximum(j,0)])/US,86400) if len(ts) else np.full(len(seconds),86400.)
        features[f'liq_{name}_log_age'] = np.log1p(np.minimum(elapsed,86400))
        for w in WINDOWS:
            count,notional = windows[w][name]
            features[f'liq_{name}_log_count_{w}s'] = np.log1p(count)
            if w in [5,20,60]:
                features[f'liq_{name}_log_notional_{w}s'] = np.log1p(notional)
        c5,v5 = windows[5][name]
        features[f'liq_{name}_log_count_ratio'] = np.log1p(c5/np.maximum(base[name][0]/240,1))
        features[f'liq_{name}_log_notional_ratio'] = np.log1p(v5/np.maximum(base[name][1]/240,1000))
        features[f'liq_{name}_acceleration'] = (2*c5-windows[10][name][0])/5
    for w in [5,20,60]:
        buy = sum(v[1] for k,v in windows[w].items() if k.endswith('_buy'))
        sell = sum(v[1] for k,v in windows[w].items() if k.endswith('_sell'))
        features[f'liq_imbalance_{w}s'] = (buy-sell)/np.maximum(buy+sell,1)
    frame = pd.DataFrame(features)
    # Fixed, declared missing-value policy; no validation-fitted statistics.
    frame = frame.replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    frame.insert(0,'second',seconds)
    return frame


def feature_columns(kind):
    schema = json.loads((OUT/'feature_schema.json').read_text())
    market = schema['market']
    base = [x for x in market if not x.startswith('as_')] + ['taker_sign','alignment','signed_return_5s','signed_flow_20s']
    if kind == 'LR':
        return [x for x in market if not x.startswith('as_')]
    if kind == 'E4':
        return [x for x in base if not x.startswith('liq_') and x != 'alignment']
    if kind in ['E5','E7']:
        base += ['p_buy_burst_5s','p_sell_burst_5s','p_aligned_burst_5s','p_opposing_burst_5s']
    if kind in ['E6','E7']:
        base += AS_FEATURES
    return base


def trade_features(groups, states, probabilities=None):
    result = groups[['second','taker_sign','alignment']].merge(states,on='second',how='left',validate='many_to_one',sort=False)
    s = result.taker_sign.to_numpy()
    result['signed_return_5s'] = s*result.return_5s_bps
    result['signed_flow_20s'] = s*result.trade_flow_imbalance_20s
    result['as_maker_signed_volatility'] = -s*result.as_volatility_30s_bps
    result['as_maker_signed_spread_volatility'] = -s*result.as_spread_over_volatility
    if probabilities is not None:
        result = result.merge(probabilities,on='second',how='left',validate='many_to_one',sort=False)
        result['p_aligned_burst_5s'] = np.where(s>0,result.p_buy_burst_5s,result.p_sell_burst_5s)
        result['p_opposing_burst_5s'] = np.where(s>0,result.p_sell_burst_5s,result.p_buy_burst_5s)
    return result


def build(symbol):
    arrays=liquidation_arrays(pl.read_parquet(source(symbol,'binance_liquidations')),
                              pl.read_parquet(source(symbol,'bybit_liquidations')))
    folder=OUT/'states'/symbol
    folder.mkdir(parents=True,exist_ok=True)
    for day in range(START,END,86400):
        name=pd.Timestamp(day,unit='s',tz='UTC').strftime('%Y-%m-%d')+'.parquet'
        path=folder/name
        if path.exists(): continue
        if not (OUT/'groups'/symbol/name).with_suffix('.json').exists():
            print('Waiting for prepared day',symbol,name,flush=True)
            break
        prior=pd.Timestamp(day-86400,unit='s',tz='UTC').strftime('%Y-%m-%d')+'.parquet'
        quote_files=[OUT/'quotes'/symbol/name]
        group_files=[OUT/'groups'/symbol/name]
        if day>START:
            quote_files.insert(0,OUT/'quotes'/symbol/prior)
            group_files.insert(0,OUT/'groups'/symbol/prior)
        seconds=np.arange(max(START,day-61),day+86400)
        quotes=pl.scan_parquet(quote_files).filter(pl.col('second')>=seconds[0]).collect().to_pandas()
        flows=pl.scan_parquet(group_files).filter(pl.col('second')>=seconds[0]).select(
            'second','taker_sign','trade_count','notional').collect().to_pandas()
        state=state_features(seconds,quotes,flows,arrays,symbol)
        state=state.loc[state.second>=day]
        temp=path.with_suffix('.tmp')
        state.to_parquet(temp,index=False)
        temp.replace(path)
        if not (OUT/'feature_schema.json').exists():
            save_json(OUT/'feature_schema.json',{'market':state.columns.drop('second').tolist(),
                'feature_clock':'strictly before floor(trade_timestamp / 1e6), held for one second',
                'exact_trade_feature':'5s signed-notional liquidation alignment, strictly before trade timestamp',
                'missing_value_policy':'fixed zero with quote_stale and log_quote_age; no imputer fitting',
                'AS_risk_horizon_s':30,'AS_inventory_used':False})
        print('State features',symbol,name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--symbol',choices=SYMBOLS)
    a=p.parse_args()
    for s in ((a.symbol,) if a.symbol else SYMBOLS):build(s)
