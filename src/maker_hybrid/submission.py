"""Deterministic causal inference. No file reads of market data or target generation.

Input must include the available history (>=1,205 seconds for complete liquidation
baselines and >=61 seconds for quotes/trades). Short histories use fixed zero/
staleness conventions, exactly as at the start of the research sample.
"""
from functools import lru_cache
import warnings
import joblib
from threadpoolctl import threadpool_limits
from .core import *
from .features import state_features, trade_features
from .policy import trade_hash, full_group_keep

RULES=('E1','E1_60','E1_60_strength')


def flow_strength(arrays, timestamps):
    windows=liquidation_window(arrays,(np.asarray(timestamps,np.int64)//US)*US,60)
    return np.abs(sum(v[1]*(1 if k.endswith('_buy') else -1) for k,v in windows.items())).astype('float32')


def _empty_liq():
    return pd.DataFrame({'timestamp':pd.Series(dtype='int64'),'side':pd.Series(dtype=str),
                         'price':pd.Series(dtype=float),'amount':pd.Series(dtype=float)})


def _symbol(frame, symbol):
    if frame is None or len(frame)==0:return frame
    if 'ticker' not in frame.columns:return frame
    ticker=frame['ticker'].astype(str).str.lower().str.replace('perp:','',regex=False)
    return frame.loc[ticker.eq(symbol)]


def _quotes(seconds, bbo):
    q={'second':seconds}
    if len(bbo):
        b=bbo.sort_values('timestamp',kind='stable')
        tt=b.timestamp.to_numpy(np.int64)
        j=np.searchsorted(tt,seconds*US,side='left')-1
        good=j>=0;jj=np.maximum(j,0)
        q['quote_timestamp']=np.where(good,tt[jj],-1)
        for c in ['bid_price','ask_price','bid_amount','ask_amount']:
            q[c]=np.where(good,b[c].to_numpy(float)[jj],np.nan)
    else:
        q['quote_timestamp']=np.full(len(seconds),-1)
        for c in ['bid_price','ask_price','bid_amount','ask_amount']:q[c]=np.full(len(seconds),np.nan)
    return pd.DataFrame(q)


def market_batches(trades,bbo,binance,bybit,symbol):
    """Yield original row indices, exact-alignment groups, and causal market states.

    Only a day's trade rows and quote context are copied at a time. Sorting of
    timestamp indices is stable; equal-time market messages are excluded.
    """
    t=trades.timestamp.to_numpy(np.int64)
    order=np.argsort(t,kind='stable') if np.any(np.diff(t)<0) else np.arange(len(t))
    ts=t[order]
    arrays=liquidation_arrays(binance,bybit)
    qt=bbo.timestamp.to_numpy(np.int64) if len(bbo) else np.array([],np.int64)
    for day in np.unique(ts//(86400*US))*86400:
        lo,hi=np.searchsorted(ts,[day*US,(day+86400)*US],side='left')
        positions=order[lo:hi];current=trades.iloc[positions]
        secs=t[positions]//US
        # >=61s of context, excluding the incomplete current second in flow.
        grid=np.arange(int(secs.min())-61,int(secs.max())+1,dtype=np.int64)
        a,b=np.searchsorted(ts,[grid[0]*US,(grid[-1]+1)*US],side='left')
        past=trades.iloc[order[a:b]]
        flow=pd.DataFrame({'second':past.timestamp.to_numpy(np.int64)//US,
            'taker_sign':np.where(past.side.eq('buy'),1,-1),
            'trade_count':np.ones(len(past),np.int64),'notional':(past.price*past.amount).to_numpy()})
        flow=flow.groupby(['second','taker_sign'],as_index=False)[['trade_count','notional']].sum()
        # Include the final quote before the context starts; no backward fill.
        if len(qt):
            qmask=(qt>=grid[0]*US)&(qt<=(grid[-1]+1)*US)
            before=np.flatnonzero(qt<grid[0]*US)
            if len(before):qmask[before[np.argmax(qt[before])]]=True
            local_bbo=bbo.loc[qmask]
        else:local_bbo=bbo
        state=state_features(grid,_quotes(grid,local_bbo),flow,arrays,symbol)
        sign=np.where(current.side.eq('buy'),1,-1).astype(np.int8)
        align=alignment(arrays,t[positions],sign)
        keys=pd.DataFrame({'second':secs,'taker_sign':sign,'alignment':align})
        # Factorization preserves the exact original row-to-group mapping.
        inverse,unique=pd.factorize(pd.MultiIndex.from_frame(keys),sort=False)
        groups=pd.DataFrame(list(unique),columns=keys.columns)
        yield positions,current,groups,inverse,state,arrays


@lru_cache(maxsize=1)
def _configuration():
    path=OUT/'submission_config.json'
    return json.loads(path.read_text())


@lru_cache(maxsize=16)
def _model(name):
    return joblib.load(OUT/'models'/name)


def signal_with_config(trades,bbo,liq_binance,liq_bybit,config):
    trades=as_pandas(trades)
    result={h:np.zeros(len(trades),dtype=np.uint8) for h in HORIZONS}
    if not len(trades):return result
    bbo=as_pandas(bbo) if bbo is not None else pd.DataFrame()
    binance=as_pandas(liq_binance) if liq_binance is not None else _empty_liq()
    bybit=as_pandas(liq_bybit) if liq_bybit is not None else _empty_liq()
    if 'ticker' not in trades:
        raise ValueError('Official input requires ticker; refusing to guess BTC versus ETH')
    symbols=trades.ticker.astype(str).str.lower().str.replace('perp:','',regex=False).to_numpy()
    for symbol in SYMBOLS:
        positions=np.flatnonzero(symbols==symbol)
        if not len(positions):continue
        tr=trades.iloc[positions]
        bn=_symbol(binance,symbol);by=_symbol(bybit,symbol);bb=_symbol(bbo,symbol)
        if bn is None or not len(bn):bn=_empty_liq()
        if by is None or not len(by):by=_empty_liq()
        policies={h:config['policies'][str(h)].get(symbol,config['policies'][str(h)].get('combined')) for h in HORIZONS}
        active={h:p for h,p in policies.items() if p and p.get('model')!='keep_all'}
        # Fixed fallback: keep-all for unsupported symbol, absent model, or absent BBO.
        if not active:continue
        needs_ml=any(p['model'].startswith('E') and p['model'] not in RULES for p in active.values())
        if not needs_ml:
            arrays=liquidation_arrays(bn,by)
            sign=np.where(tr.side.eq('buy'),1,-1)
            for h,p in active.items():
                align=alignment(arrays,tr.timestamp.to_numpy(np.int64),sign,window=p.get('window',60 if p['model'].startswith('E1_60') else 5))
                if p.get('native',False):keep=align==1
                else:
                    scores=flow_strength(arrays,tr.timestamp) if p['model']=='E1_60_strength' else np.ones(len(tr),np.float32)
                    u=policy_hash(tr.timestamp.to_numpy(np.int64)//US,symbol,sign,align)
                    keep,boundary=full_group_keep(scores,u,align,p)
                    if boundary.any():keep[boundary]=trade_hash(tr.loc[boundary],symbol)<=p['trade_tie_cut']
                result[h][positions]=(~keep).astype(np.uint8)
            continue
        if bb is None or not len(bb):continue
        loaded={};features=config['features']
        for h,p in list(active.items()):
            try:
                if p['model'] not in RULES:
                    loaded[h]=_model(f"{p['model']}_{h}_final.joblib")
                    if p['model'] in ['E5','E7']:
                        loaded['buy']=_model('lr_buy_final.joblib');loaded['sell']=_model('lr_sell_final.joblib')
            except (FileNotFoundError, OSError, ValueError) as exc:
                warnings.warn(f'{symbol} {h}s: model unavailable; fixed keep-all fallback ({type(exc).__name__})',RuntimeWarning)
                del active[h]
        if not active:continue
        for ix,raw,groups,inverse,states,arrays in market_batches(tr,bb,bn,by,symbol):
            prob=None
            if any(p['model'] in ['E5','E7'] for p in active.values()):
                prob=states[['second']].copy()
                with threadpool_limits(limits=4):
                    for side in ['buy','sell']:
                        prob[f'p_{side}_burst_5s']=loaded[side].predict_proba(states[features['LR']])[:,1].astype('float32')
            x=trade_features(groups,states,prob)
            u=policy_hash(groups.second,symbol,groups.taker_sign,groups.alignment)
            for h,p in active.items():
                kind=p['model']
                if kind in RULES:
                    sign=np.where(raw.side.eq('buy'),1,-1)
                    align=alignment(arrays,raw.timestamp.to_numpy(np.int64),sign,window=60 if kind.startswith('E1_60') else 5)
                    scores=flow_strength(arrays,raw.timestamp) if kind=='E1_60_strength' else np.ones(len(raw),np.float32)
                    uu=policy_hash(raw.timestamp.to_numpy(np.int64)//US,symbol,sign,align)
                    if p.get('native'):keep=align==1;boundary=np.zeros(len(raw),bool)
                    else:keep,boundary=full_group_keep(scores,uu,align,p)
                else:
                    with threadpool_limits(limits=4):scores=loaded[h].predict(x[features[kind]].astype('float32')).astype('float32')
                    kg,bg=full_group_keep(scores,u,groups.alignment,p)
                    keep,boundary=kg[inverse],bg[inverse]
                if boundary.any():keep[boundary]=trade_hash(raw.loc[boundary],symbol)<=p['trade_tie_cut']
                result[h][positions[ix]]=(~keep).astype(np.uint8)
    return result


def signal(trades,bbo,liq_binance,liq_bybit):
    """Official interface. 0 keeps a fill; 1 filters it. Never trains."""
    try:config=_configuration()
    except (FileNotFoundError, OSError, ValueError):
        warnings.warn('Configuration unavailable; fixed keep-all fallback',RuntimeWarning)
        return {h:np.zeros(len(trades),np.uint8) for h in HORIZONS}
    return signal_with_config(trades,bbo,liq_binance,liq_bybit,config)


def hybrid_signal(trades,bbo,liq_binance,liq_bybit):
    """The complete LR -> LightGBM + AS-risk research architecture, frozen E7."""
    try:config=json.loads((OUT/'hybrid_config.json').read_text())
    except (FileNotFoundError,OSError,ValueError):
        warnings.warn('Hybrid configuration unavailable; fixed keep-all fallback',RuntimeWarning)
        return {h:np.zeros(len(trades),np.uint8) for h in HORIZONS}
    return signal_with_config(trades,bbo,liq_binance,liq_bybit,config)
