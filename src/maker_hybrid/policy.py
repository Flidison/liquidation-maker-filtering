"""Fixed score thresholds and deterministic ties, calibrated without future PnL.

Matched turnover is rounded upward by at most one clipped trade ($100k over the
entire evaluation period), never by an entire high-volume second. The score and
filter themselves remain the official binary ones.
"""
from functools import lru_cache
from .core import *
from .prepare import bounds, bbo_slice


def trade_hash(frame, symbol):
    """Identical trades retain separate deterministic occurrence numbers.

    Prefixes at a timestamp retain all equal-time rows. Adding later timestamps
    cannot change these within-identical-message occurrence numbers.
    """
    columns=['timestamp','side','price','amount']
    rank=frame.groupby(columns,sort=False,dropna=False).cumcount().to_numpy(dtype=np.uint64)
    t=frame.timestamp.to_numpy(dtype=np.uint64)
    price=frame.price.to_numpy(dtype=np.float64).view(np.uint64)
    amount=frame.amount.to_numpy(dtype=np.float64).view(np.uint64)
    x=t ^ (price*np.uint64(0x9e3779b97f4a7c15)) ^ (amount*np.uint64(0xbf58476d1ce4e5b9))
    x=x ^ (rank*np.uint64(0x94d049bb133111eb)) ^ np.uint64(SEED+(12345 if symbol=='ethusdt' else 0))
    x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return ((x^(x>>np.uint64(31)))>>np.uint64(11)).astype(float)/2**53


@lru_cache(maxsize=2)
def liq_arrays(symbol):
    return liquidation_arrays(pl.read_parquet(source(symbol,'binance_liquidations')),
                               pl.read_parquet(source(symbol,'bybit_liquidations')))


@lru_cache(maxsize=2)
def bbo_bounds(symbol):
    return bounds(source(symbol,'binance_booktickers'))


def boundary_trades(symbol,second,sign,align,window=5):
    path=OUT/('boundary_trades' if window==5 else 'boundary_trades60')/f'{symbol}_{second}_{sign}_{align}.parquet'
    if path.exists():return pd.read_parquet(path)
    f=pl.scan_parquet(source(symbol,'binance_trades')).filter(
        (pl.col('timestamp')>=second*US)&(pl.col('timestamp')<(second+1)*US)&
        (pl.col('side')==('buy' if sign==1 else 'sell'))).collect(engine='streaming').to_pandas()
    current=alignment(liq_arrays(symbol),f.timestamp.to_numpy(),np.full(len(f),sign),window=window)
    f=f.loc[current==align].copy()
    f['u_trade']=trade_hash(f,symbol)
    f['weight']=np.minimum(f.price*f.amount,100000)
    b=bbo_slice(symbol,(second-1800)*US,(second+301)*US)
    qt=b['timestamp'].to_numpy();mid=(b['bid_price'].to_numpy()+b['ask_price'].to_numpy())/2
    for h in HORIZONS:
        y,age=maker_labels(f.timestamp.to_numpy(),f.price.to_numpy(),np.full(len(f),sign),qt,mid,h,bbo_bounds(symbol))
        valid=np.isfinite(y)
        f[f'w_{h}']=np.where(valid,f.weight,0)
        f[f'wy_{h}']=f[f'w_{h}']*np.nan_to_num(y)
        f[f'n_{h}']=valid.astype(int)
        f[f'fresh_w_{h}']=np.where(age<=2,f[f'w_{h}'],0)
        f[f'fresh_wy_{h}']=np.where(age<=2,f[f'wy_{h}'],0)
        f[f'future_age_{h}']=age
    path.parent.mkdir(parents=True,exist_ok=True)
    f.to_parquet(path,index=False)
    return f


def ranked_groups(frame,model,horizon):
    if model=='E0_random':score=np.zeros(len(frame),dtype=np.float32)
    elif model in ['E1','E1_60']:score=np.ones(len(frame),dtype=np.float32)
    else:score=frame[f'{model}_{horizon}'].to_numpy(dtype=np.float32)
    eligible=np.isfinite(score)&(frame[f'w_{horizon}'].to_numpy()>0)
    if model in ['E1','E1_60','E1_60_strength']:eligible &= frame.alignment.to_numpy()==1
    indices=np.flatnonzero(eligible)
    order=np.lexsort((frame.tie_key.to_numpy()[indices],-score[indices]))
    indices=indices[order]
    return score,indices,np.cumsum(frame[f'w_{horizon}'].to_numpy()[indices])


def calibrate(frame,model,horizon,budget_per_day,days,ranking=None):
    score,order,cumulative=ranking or ranked_groups(frame,model,horizon)
    target=budget_per_day*days
    total=cumulative[-1] if len(cumulative) else 0
    if total<target:
        return {'model':model,'tau':horizon,'budget_per_day':budget_per_day,'feasible':False,
                'candidate_turnover':float(total),'score_threshold':None},order,None
    k=int(np.searchsorted(cumulative,target,side='left'))
    boundary=int(order[k]);row=frame.iloc[boundary]
    before=float(cumulative[k-1]) if k else 0.
    remaining=target-before
    raw=boundary_trades(row.symbol,int(row.second),int(row.taker_sign),int(row.alignment),window=60 if model.startswith('E1_60') else 5)
    assert np.isclose(raw[f'w_{horizon}'].sum(),row[f'w_{horizon}'],rtol=1e-10,atol=.001)
    # Crucially: this ranking uses only deterministic hashes and weights, not labels.
    raw=raw.sort_values('u_trade',kind='stable').reset_index(drop=True)
    cw=raw[f'w_{horizon}'].cumsum().to_numpy()
    j=min(int(np.searchsorted(cw,remaining,side='left')),len(raw)-1)
    cut=float(raw.u_trade.iloc[j])
    part=raw.loc[raw.u_trade<=cut]
    extra=before+part[f'w_{horizon}'].sum()-target
    assert -.001<=extra<=100000.001,extra
    config={'model':model,'tau':horizon,'budget_per_day':budget_per_day,'feasible':True,
        'score_threshold':float(score[boundary]),'group_tie_cut':int(row.tie_key),
        'trade_tie_cut':cut,'boundary_symbol':row.symbol,'boundary_second':int(row.second),
        'boundary_taker_sign':int(row.taker_sign),'boundary_alignment':int(row.alignment),
        'target_total_turnover':target,'actual_total_turnover':float(before+part[f'w_{horizon}'].sum()),
        'extra_turnover_over_budget':float(extra),
        'parameter_selection':'scores, clipped weights and deterministic ties only; no markout values',
        'policy_type':'fixed threshold; no ranking of future days at inference'}
    # The executable policy must exactly recover the calibration selection.
    kept_check,boundary_check=full_group_keep(score,frame.tie_key,frame.alignment,config)
    # Target availability restricts evaluation only, never the public signal.
    kept_check &= frame[f'w_{horizon}'].to_numpy()>0
    assert np.array_equal(np.flatnonzero(boundary_check),np.array([boundary]))
    assert np.array_equal(np.flatnonzero(kept_check),np.sort(order[:k]))
    return config,order[:k],(boundary,part)


def full_group_keep(scores,group_hash,alignment_value,policy):
    scores=np.asarray(scores,dtype=np.float32)
    group_hash=np.asarray(group_hash,dtype=np.uint64)
    eligible=np.ones(len(scores),bool)
    if policy['model'] in ['E1','E1_60','E1_60_strength']:eligible &= np.asarray(alignment_value)==1
    threshold=policy['score_threshold']
    if threshold is None:return eligible,np.zeros(len(scores),bool)
    keep=eligible&((scores>threshold)|((scores==threshold)&(group_hash<policy['group_tie_cut'])))
    boundary=eligible&(scores==threshold)&(group_hash==policy['group_tie_cut'])
    return keep,boundary
