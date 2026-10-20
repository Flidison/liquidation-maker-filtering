"""Full-stream official labels, reduced by causal feature-equivalence groups.

Grouping is exact for policies based on (last completed second, taker side,
exact 5s liquidation alignment). No score sampling or trade deduplication.
"""
import argparse
import time
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from .core import *


def bounds(path):
    f = pq.ParquetFile(path)
    i = f.schema_arrow.get_field_index('timestamp')
    stats = [f.metadata.row_group(j).column(i).statistics for j in range(f.metadata.num_row_groups)]
    return min(s.min for s in stats), max(s.max for s in stats)


def bbo_slice(symbol, start, end):
    return pl.scan_parquet(source(symbol, 'binance_booktickers')).filter(
        (pl.col('timestamp') >= start) & (pl.col('timestamp') <= end)).collect(engine='streaming')


def prepare_day(symbol, day, arrays, global_bbo_bounds, window=5):
    day_text = pd.Timestamp(day, unit='s', tz='UTC').strftime('%Y-%m-%d')
    target = OUT / ('groups' if window==5 else 'groups60') / symbol / f'{day_text}.parquet'
    state_file = OUT / 'quotes' / symbol / f'{day_text}.parquet'
    sample_file = OUT / 'trade_samples' / symbol / f'{day_text}.parquet'
    receipt = target.with_suffix('.json')
    if receipt.exists():
        return
    started = time.monotonic()
    for path in [target, state_file, sample_file]:
        path.parent.mkdir(parents=True, exist_ok=True)
    bbo = bbo_slice(symbol, (day-1800)*US, (day+86400+300)*US)
    qt = bbo['timestamp'].to_numpy()
    mid = (bbo['bid_price'].to_numpy()+bbo['ask_price'].to_numpy())/2
    # Strictly preceding quote at each second boundary, never within that second.
    seconds = np.arange(day, day+86400, dtype=np.int64)
    j = np.searchsorted(qt, seconds*US, side='left')-1
    good, jj = j >= 0, np.maximum(j, 0)
    states = {'second': seconds, 'quote_timestamp': np.where(good, qt[jj], -1)}
    for name in ['bid_price', 'ask_price', 'bid_amount', 'ask_amount']:
        states[name] = np.where(good, bbo[name].to_numpy()[jj], np.nan)
    if window==5: pl.DataFrame(states).write_parquet(state_file, compression='zstd')
    scanner = ds.dataset(source(symbol, 'binance_trades'), format='parquet').scanner(
        columns=['timestamp','side','price','amount'],
        filter=(ds.field('timestamp') >= day*US) & (ds.field('timestamp') < (day+86400)*US),
        batch_size=500_000, use_threads=False)
    pieces, samples, nrows = [], [], 0
    key = ['second','taker_sign','alignment']
    for batch in scanner.to_batches():
        t = batch.column('timestamp').to_numpy()
        price = batch.column('price').to_numpy()
        amount = batch.column('amount').to_numpy()
        sign = np.where(batch.column('side').to_numpy(zero_copy_only=False) == 'buy', 1, -1).astype(np.int8)
        notional = price*amount
        weight = np.minimum(notional, 100000.)
        align = alignment(arrays, t, sign, window=window)
        f = {'second':t//US, 'taker_sign':sign, 'alignment':align,
             'trade_count':np.ones(len(t), np.int64), 'notional':notional, 'weight':weight}
        for h in HORIZONS:
            pnl, age = maker_labels(t, price, sign, qt, mid, h, global_bbo_bounds)
            valid = np.isfinite(pnl)
            w = np.where(valid, weight, 0.)
            yy = np.nan_to_num(pnl)
            f.update({f'w_{h}':w, f'wy_{h}':w*yy, f'wy2_{h}':w*yy**2,
                      f'n_{h}':valid.astype(np.int64),
                      f'fresh_w_{h}':np.where(age <= 2, w, 0.),
                      f'fresh_wy_{h}':np.where(age <= 2, w*yy, 0.),
                      f'stale_n_{h}':(age > 2).astype(np.int64),
                      f'stale10_w_{h}':np.where(age > 10, w, 0.)})
        frame = pl.DataFrame(f)
        pieces.append(frame.group_by(key).agg(pl.exclude(key).sum()))
        # Fixed systematic row sampling for distribution plots only, never Score.
        mask = (np.arange(nrows, nrows+len(t)) % 1000) == 17
        sample = frame.filter(pl.Series(mask))
        if sample.height:
            samples.append(sample.with_columns(pl.Series('timestamp', t[mask]),
                           pl.Series('row_in_day', np.flatnonzero(mask)+nrows)))
        nrows += len(t)
    grouped = pl.concat(pieces).group_by(key).agg(pl.exclude(key).sum()).sort(key)
    assert grouped['trade_count'].sum() == nrows
    grouped.write_parquet(target, compression='zstd')
    if window==5: pl.concat(samples).write_parquet(sample_file, compression='zstd')
    save_json(receipt, {'symbol':symbol, 'day':day_text, 'raw_trade_rows':nrows,
                       'groups':grouped.height, 'source_order_preserved_by_sampling':True,
                       'seconds_elapsed':time.monotonic()-started,
                       'group_sha256':digest(target),'quote_sha256':digest(state_file),
                       'score_uses_all_rows':True})
    print(f'{symbol} {day_text}: {nrows:,} trades -> {grouped.height:,} groups; '
          f'{time.monotonic()-started:.1f}s', flush=True)


def run(symbols=SYMBOLS, first=START, last=END):
    for symbol in symbols:
        arrays = liquidation_arrays(pl.read_parquet(source(symbol,'binance_liquidations')),
                                    pl.read_parquet(source(symbol,'bybit_liquidations')))
        b = bounds(source(symbol,'binance_booktickers'))
        for day in range(first, last, 86400):
            prepare_day(symbol, day, arrays, b)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', choices=SYMBOLS)
    p.add_argument('--first',default='2025-12-01')
    p.add_argument('--last',default='2026-03-01')
    args = p.parse_args()
    run((args.symbol,) if args.symbol else SYMBOLS,
        int(pd.Timestamp(args.first,tz='UTC').timestamp()),
        int(pd.Timestamp(args.last,tz='UTC').timestamp()))
