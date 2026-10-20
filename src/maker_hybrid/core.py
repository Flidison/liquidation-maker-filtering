from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import polars as pl

from research_paths import ROOT, resolve_legacy, DATA_ROOT
OUT = ROOT / 'artifacts/results/maker'
DATA = DATA_ROOT
HORIZONS = (30, 120, 300)
SYMBOLS = ('btcusdt', 'ethusdt')
SEED = 20261017
US = 1_000_000
START = int(pd.Timestamp('2025-12-01', tz='UTC').timestamp())
READY = int(pd.Timestamp('2025-12-15', tz='UTC').timestamp())
VALIDATION = int(pd.Timestamp('2026-02-01', tz='UTC').timestamp())
END = int(pd.Timestamp('2026-03-01', tz='UTC').timestamp())


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def uniform_hash(second, symbol, taker_sign=0, align=0):
    """Stable pseudo-random tie break/sample key; no future ranks or row indices."""
    x = np.asarray(second,dtype=np.uint64)*np.uint64(16)
    x = x + np.asarray(taker_sign,dtype=np.int64).astype(np.uint64) * np.uint64(3)
    x = x + np.asarray(align,dtype=np.int64).astype(np.uint64) + np.uint64(SEED + (12345 if symbol=='ethusdt' else 0))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / 2**53


def policy_hash(second, symbol, taker_sign, align):
    """Bijective uint64 tie keys with disjoint symbol namespaces.

    Do not use the floating sampling hash for policy ties: additive ticker salts
    can collide with another timestamp. Keys stay integers in parquet and JSON.
    The input encoding is injective for supported symbols, signs and alignments.
    """
    raw=np.asarray(second,dtype=np.uint64)*np.uint64(16)
    raw+=((np.asarray(taker_sign,dtype=np.int64)+1)*3+np.asarray(align,dtype=np.int64)+1).astype(np.uint64)
    x=(raw*np.uint64(2)+np.uint64(symbol=='ethusdt')) ^ np.uint64(SEED)
    x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x^(x>>np.uint64(31))


def source(symbol, kind):
    return DATA / kind / (('' if kind.startswith('bybit') else 'perp_') + symbol + '.parquet')


def as_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    return frame.to_pandas()


def liquidation_arrays(binance, bybit):
    result = {}
    for exchange, frame in [('binance', binance), ('bybit', bybit)]:
        f = as_pandas(frame)
        t = f.timestamp.to_numpy(dtype=np.int64) + (200_000 if exchange == 'bybit' else 0)
        notional = (f.price * f.amount).to_numpy(float)
        for side in ['buy', 'sell']:
            mask = f.side.eq(side).to_numpy()
            order = np.argsort(t[mask], kind='stable')
            tt, vv = t[mask][order], notional[mask][order]
            result[f'{exchange}_{side}'] = (tt, np.r_[0., np.cumsum(vv)])
    return result


def liquidation_window(arrays, queries_us, window):
    out = {}
    for name, (ts, sums) in arrays.items():
        right = np.searchsorted(ts, queries_us, side='left')
        left = np.searchsorted(ts, queries_us - window * US, side='left')
        out[name] = (right-left, sums[right]-sums[left])
    return out


def alignment(arrays, queries_us, taker_sign, window=5):
    info = liquidation_window(arrays, np.asarray(queries_us, np.int64), window)
    net = sum(v[1] * (1 if k.endswith('_buy') else -1) for k, v in info.items())
    # Counts are exact; floating cancellation at an empty window is exactly zero.
    return (np.sign(net) * taker_sign).astype(np.int8)


def maker_labels(t, price, taker_sign, quote_t, midpoint, horizon, global_bounds=None):
    """Official backward as-of, including an equal-timestamp future quote."""
    target = np.asarray(t, np.int64) + horizon * US
    if not len(quote_t):
        return np.full(len(target),np.nan), np.full(len(target),np.nan)
    j = np.searchsorted(quote_t, target, side='right')-1
    lower, upper = global_bounds if global_bounds is not None else (quote_t[0], quote_t[-1])
    valid = (j >= 0) & (target >= lower) & (target <= upper)
    jj = np.clip(j, 0, len(quote_t)-1)
    age = (target - quote_t[jj])/US
    pnl = -np.asarray(taker_sign) * (midpoint[jj] - price)/price * 10000 + .5
    return np.where(valid, pnl, np.nan), np.where(valid, age, np.nan)


def official_score(pnl, weight, filtered, number_of_days):
    pnl, weight, filtered = np.asarray(pnl), np.asarray(weight), np.asarray(filtered)
    if not np.isin(filtered, [0, 1]).all():
        raise ValueError('Filter must be binary: 0 keep, 1 filter')
    valid = np.isfinite(pnl) & np.isfinite(weight) & (weight >= 0)
    p, w, keep = pnl[valid], weight[valid], filtered[valid] == 0
    all_w, kept_w, removed_w = w.sum(), w[keep].sum(), w[~keep].sum()
    all_pnl = np.sum(p*w)/all_w if all_w else np.nan
    kept_pnl = np.sum(p[keep]*w[keep])/kept_w if kept_w else np.nan
    removed_pnl = np.sum(p[~keep]*w[~keep])/removed_w if removed_w else np.nan
    return dict(score=kept_pnl-all_pnl, pnl_all=all_pnl, pnl_kept=kept_pnl,
                pnl_filtered=removed_pnl, kept_turnover_per_day=kept_w/number_of_days,
                kept_fraction=kept_w/all_w if all_w else np.nan,
                fraction_trades_retained=float(np.mean(keep)),
                total_weighted_marked_pnl_usd=float(np.sum(p[keep]*w[keep])/10000))


def as_quotes(mid, inventory, gamma, sigma_price_squared, horizon_seconds, k):
    """Theoretical A-S approximation, never used by the deployed filter.

    Fix one inventory lot: mid is its USD price, inventory is a number of lots,
    sigma_price_squared is USD^2/second, horizon is seconds, gamma and k are
    reciprocal USD in this fixed-lot convention. No identifiable inventory or k
    is available in the supplied feeds; arbitrary values are not calibrated risk.
    """
    if gamma <= 0 or k <= 0 or sigma_price_squared < 0 or horizon_seconds < 0:
        raise ValueError('Invalid A-S parameters')
    risk = gamma * sigma_price_squared * horizon_seconds
    reservation = mid - inventory * risk
    spread = risk + 2/gamma * np.log1p(gamma/k)
    return reservation, reservation-spread/2, reservation+spread/2
