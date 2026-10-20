"""Full-row quality audit, extracted from Data Quality.ipynb.
The formulas and thresholds are preserved; paths and orchestration are portable.
Raw files are read only. Run explicitly with python -m data_quality_audit.
"""


from pathlib import Path
from collections import Counter, defaultdict
import gc, hashlib, json, math, os, pickle, platform, shutil, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display, Markdown, HTML


from research_paths import ROOT, DATA_ROOT as DATA, artifact_dir
DATA_ROOT = DATA.parent
OUTPUT = artifact_dir("quality")
WORK = OUTPUT / "_temporary"


BATCH_SIZE = 1_000_000
BYBIT_DELAY_US = 200_000
BBO_GAP_US = 10_000_000
TOP_N = 10
WINDOW_SECONDS = 5
REUSE_VERIFIED_CACHE = True  # False: пересчитать; True: только при совпадении SHA256 и версии кода.
AUDIT_VERSION = '2026-10-04-v1'
SEVERITIES = ['CLEAR_DATA_ERROR', 'SUSPICIOUS', 'PLAUSIBLE_MARKET_EVENT']
ACTIONS = {'CLEAR_DATA_ERROR': 'REMOVED_FROM_CLEAN_COPY',
           'SUSPICIOUS': 'KEPT_FOR_INVESTIGATION',
           'PLAUSIBLE_MARKET_EVENT': 'KEPT_AS_MARKET_EVENT'}
plt.rcParams.update({'figure.figsize': (11, 3.5), 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .18, 'figure.dpi': 110})
COLORS = {'bbo': '#2364AA', 'trades': '#6A737D', 'buy': '#CB8A20',
          'sell': '#7D4C99', 'bybit': '#C55D32'}
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)

def heading(text, level=3):
    display(Markdown('#' * level + ' ' + str(text)))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def utc(ts):
    return pd.to_datetime(ts, unit='us', utc=True, errors='coerce')

def symbol(ticker):
    return str(ticker).lower().removeprefix('perp:')

def file_kind(path):
    s = str(path).lower()
    if 'bookticker' in s or 'bbo' in s:
        return 'bbo'
    if 'liquidation' in path.parent.name.lower():
        return 'liquidations'
    if 'trade' in path.parent.name.lower():
        return 'trades'
    cols = pq.ParquetFile(path).schema_arrow.names
    if {'bid_price', 'ask_price'} <= set(cols):
        return 'bbo'
    raise ValueError(f'Нераспознанный рыночный файл: {path}; классифицируйте явно')

def file_id(path):
    return ('data/' + str(path.relative_to(DATA))).replace('/', '__').replace('.parquet', '')

def stat_quantiles(values):
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {}
    return {'minimum': s.min(), 'p05': s.quantile(.05), 'p25': s.quantile(.25),
            'median': s.median(), 'p75': s.quantile(.75), 'p95': s.quantile(.95),
            'p99': s.quantile(.99), 'p99.9': s.quantile(.999),
            'maximum': s.max(), 'mean': s.mean(), 'count': len(s)}

class ParquetSink:
    """Пишет только отдельные производные артефакты; source parquet не является target."""
    def __init__(self, path):
        self.path, self.writer = Path(path), None
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def write(self, frame):
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='zstd')
        self.writer.write_table(table.cast(self.writer.schema), row_group_size=250_000)
    def close(self):
        if self.writer is not None:
            self.writer.close()

class ExactDistribution:
    """Точные quantile_cont на дисковой копии float64; никаких sampled percentiles."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(self.path, 'wb')
        self.n, self.total = 0, 0.
        self.minimum, self.maximum = np.inf, -np.inf
        self.zero_count = 0
    def add(self, values):
        a = np.asarray(values, dtype=np.float64)
        a = a[np.isfinite(a)]
        if not len(a):
            return
        a.tofile(self.handle)
        self.n += len(a)
        self.total += a.sum(dtype=np.float64)
        self.minimum = min(self.minimum, float(a.min()))
        self.maximum = max(self.maximum, float(a.max()))
        self.zero_count += int((a == 0).sum())
    def finish(self):
        self.handle.close()
        if not self.n:
            self.path.unlink(missing_ok=True)
            return {'count': 0, 'hist_counts': [], 'hist_edges': []}
        qs = np.array([0, .05, .25, .5, .75, .95, .99, .999, 1.])
        pos = (self.n - 1) * qs
        lo, hi = np.floor(pos).astype('int64'), np.ceil(pos).astype('int64')
        data = np.memmap(self.path, dtype='float64', mode='r+', shape=(self.n,))
        # partition in place изменяет ТОЛЬКО временный вектор, не исходные наблюдения.
        data.partition(np.unique(np.r_[lo, hi]))
        quantiles = data[lo] + (data[hi] - data[lo]) * (pos - lo)
        ans = dict(zip(['minimum', 'p05', 'p25', 'median', 'p75', 'p95',
                        'p99', 'p99.9', 'maximum'], map(float, quantiles)))
        ans.update(count=self.n, mean=self.total / self.n, zero_count=self.zero_count)
        # Signed log1p сохраняет нули и отрицательные значения; histogram по ВСЕМ строкам.
        transform = lambda a: np.sign(a) * np.log10(1 + np.abs(a))
        a, b = transform(np.array([self.minimum, self.maximum]))
        edges = np.linspace(a, b if b > a else a + 1, 65)
        counts = np.zeros(64, dtype='int64')
        above = 0
        for start in range(0, self.n, BATCH_SIZE):
            chunk = data[start:start+BATCH_SIZE]
            counts += np.histogram(transform(chunk), edges)[0]
            above += int((chunk > ans['p99.9']).sum())
        ans.update(hist_edges=edges.tolist(), hist_counts=counts.tolist(), above_p999=above)
        del data
        self.path.unlink()
        return ans

def emit_issue(sink, rows, issue_type, reason, severity='SUSPICIOUS'):
    if rows.empty:
        return
    out = pd.DataFrame({'source_file': rows['source_file'].astype(str),
                        'source_row': rows['source_row'].astype('int64'),
                        'ticker': rows['ticker'].astype(str),
                        'timestamp': pd.array(rows['timestamp'], dtype='Int64')})
    out['datetime_utc'] = utc(out['timestamp'])
    out['issue_type'], out['severity'], out['reason'] = issue_type, severity, reason
    out['action'] = ACTIONS[severity]
    sink.write(out.reset_index(drop=True))

def raw_rows(path, row_ids, radius=0):
    """Точное извлечение исходных строк по 0-based physical row id, без полного read_parquet."""
    pf = pq.ParquetFile(path)
    ids = np.unique(np.concatenate([np.arange(max(0, int(i)-radius),
                    min(pf.metadata.num_rows, int(i)+radius+1)) for i in row_ids])) if len(row_ids) else []
    offsets = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows
                                 for i in range(pf.metadata.num_row_groups)])]
    frames = []
    for group in np.unique(np.searchsorted(offsets[1:], ids, side='right')):
        selected = np.asarray(ids)[(np.asarray(ids) >= offsets[group]) &
                                  (np.asarray(ids) < offsets[group+1])]
        tab = pf.read_row_group(int(group)).take(pa.array(selected-offsets[group]))
        f = tab.to_pandas()
        f.insert(0, 'source_row', selected)
        f['datetime_utc'] = utc(f['timestamp'])
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def show_context(result, row_ids, radius=3):
    for row_id in row_ids:
        heading(f"{result['id']} — source_row={row_id}", 4)
        display(raw_rows(result['path'], [row_id], radius).style.apply(
            lambda r: ['background-color: #fff0ce' if r.source_row == row_id else ''] * len(r), axis=1))

def original_batches(path):
    pf, offset = pq.ParquetFile(path), 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        f = batch.to_pandas()
        f['source_row'] = np.arange(offset, offset+len(f), dtype=np.int64)
        f['source_file'] = str(path)
        offset += len(f)
        yield f

def schema_audit(frame, original_columns):
    raw = frame[original_columns]
    missing = raw.isna()  # эквивалент df.isna().sum(), суммируется по всем batches.
    blank = pd.DataFrame({c: (raw[c].astype('string').str.strip().eq('').fillna(False)
                              if pd.api.types.is_string_dtype(raw[c]) else False)
                          for c in original_columns}, index=raw.index)
    return missing.sum(), blank.sum(), (missing | blank).all(axis=1), missing.any(axis=1)

def timestamp_audit(frame, previous_timestamp):
    timestamp = frame['timestamp']
    numeric = pd.to_numeric(timestamp, errors='coerce')
    invalid = numeric.isna() | ~np.isfinite(numeric) | numeric.mod(1).ne(0)
    # Для int64 диапазона данных pandas datetime64[us] не ограничен 2262 годом.
    dt = utc(numeric)
    invalid |= dt.isna()
    timestamp_diff = numeric.diff()
    if len(frame) and previous_timestamp is not None:
        timestamp_diff.iloc[0] = numeric.iloc[0] - previous_timestamp
    return invalid, timestamp_diff

def bbo_audit(frame):
    out = {}
    for col in ['bid_price', 'ask_price']:
        x = frame[col]
        out[f'INVALID_{col.upper()}'] = (x.notna() & ((x <= 0) | ~np.isfinite(x)),
            'CLEAR_DATA_ERROR', f'{col} <= 0 либо non-finite: непригодная цена')
    for col in ['bid_amount', 'ask_amount']:
        x = frame[col]
        out[f'INVALID_{col.upper()}'] = (x.notna() & ((x < 0) | ~np.isfinite(x)),
            'CLEAR_DATA_ERROR', f'{col} < 0 либо non-finite: непригодный size')
        out[f'ZERO_{col.upper()}'] = (x.eq(0), 'SUSPICIOUS', f'{col}=0; нужен контекст, не доказанная ошибка')
    out['CROSSED_BBO'] = (frame.ask_price < frame.bid_price, 'SUSPICIOUS',
                         'ask < bid: crossed book / feed ordering / asynchronous update')
    out['MISSING_BID'] = (frame.bid_price.isna() | frame.bid_amount.isna(), 'SUSPICIOUS',
                          'Неполная bid сторона; missing price делает mid неопределённым')
    out['MISSING_ASK'] = (frame.ask_price.isna() | frame.ask_amount.isna(), 'SUSPICIOUS',
                          'Неполная ask сторона; missing price делает mid неопределённым')
    return out

def trade_audit(frame):
    return {'INVALID_PRICE': (frame.price.notna() & ((frame.price <= 0) | ~np.isfinite(frame.price)),
                              'CLEAR_DATA_ERROR', 'price <= 0 либо non-finite'),
            'INVALID_AMOUNT': (frame.amount.notna() & ((frame.amount < 0) | ~np.isfinite(frame.amount)),
                              'CLEAR_DATA_ERROR', 'amount < 0 либо non-finite'),
            'INVALID_SIDE': (~frame.side.isin(['buy', 'sell']), 'CLEAR_DATA_ERROR',
                             'side отсутствует или не принадлежит {buy, sell}')}

def liquidation_audit(frame):
    # Валидность схемы та же; величина amount / notional НЕ является правилом удаления.
    return trade_audit(frame)

def validation_masks(frame, kind, previous_timestamp=None):
    columns = [c for c in frame if c not in ['source_row', 'source_file']]
    nan, blanks, empty, any_missing = schema_audit(frame, columns)
    invalid_ts, diff = timestamp_audit(frame, previous_timestamp)
    empty_strings = pd.DataFrame({c: (
        frame[c].astype('string').str.strip().eq('').fillna(False)
        if pd.api.types.is_string_dtype(frame[c]) else pd.Series(False,index=frame.index,dtype=bool))
        for c in columns}, index=frame.index).any(axis=1)
    rules = {'FULLY_EMPTY_ROW': (empty, 'CLEAR_DATA_ERROR', 'Все поля пусты'),
             'INVALID_TIMESTAMP': (invalid_ts, 'CLEAR_DATA_ERROR', 'Отсутствующий/нецелый/нечитаемый timestamp'),
             'MISSING_VALUES': (any_missing, 'SUSPICIOUS', 'Есть NaN; missing price/size не восстанавливаются'),
             'EMPTY_STRING': (empty_strings,
                         'SUSPICIOUS', 'Пустая строка в поле'),
             'TIMESTAMP_ORDERING_ISSUE': (diff < 0, 'SUSPICIOUS', 'timestamp[i] < timestamp[i-1] в исходном порядке')}
    rules.update(bbo_audit(frame) if kind == 'bbo' else
                 liquidation_audit(frame) if kind == 'liquidations' else trade_audit(frame))
    remove = pd.Series(False, index=frame.index)
    for mask, severity, _ in rules.values():
        if severity == 'CLEAR_DATA_ERROR':
            remove |= mask
    return rules, remove, nan, blanks, diff

def iter_cleaned(path):
    """Временные pandas DataFrame; очищенные source файлы НЕ сохраняются."""
    for frame in original_batches(Path(path)):
        _, remove, _, _, _ = validation_masks(frame, file_kind(Path(path)))
        cleaned_df = frame.loc[~remove].copy()
        cleaned_df['datetime_utc'] = utc(cleaned_df.timestamp)
        if 'bybit' in str(path).lower():
            cleaned_df['raw_timestamp'] = cleaned_df.timestamp
            cleaned_df['effective_timestamp'] = cleaned_df.timestamp + BYBIT_DELAY_US
        yield cleaned_df

FEATURE_COLS = ['second', 'ticker', 'count', 'amount', 'notional', 'buy_notional',
                'sell_notional', 'first_price', 'last_price', 'min_price', 'max_price',
                'max_spread_bps']

def second_features(frame, kind, delay):
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLS)
    f = pd.DataFrame({'second': (frame.timestamp.astype('int64') + delay) // 1_000_000,
                      'ticker': frame.ticker.astype(str)})
    if kind == 'bbo':
        p = (frame.bid_price + frame.ask_price) / 2
        f['amount'], f['notional'] = 0., 0.
        f['buy_notional'], f['sell_notional'] = 0., 0.
        f['spread'] = (frame.ask_price-frame.bid_price) / p * 10000
    else:
        p = frame.price
        f['amount'], f['notional'] = frame.amount, frame.price * frame.amount
        f['buy_notional'] = f.notional.where(frame.side.eq('buy'), 0.)
        f['sell_notional'] = f.notional.where(frame.side.eq('sell'), 0.)
        f['spread'] = np.nan
    f['price'] = p.where(np.isfinite(p) & (p > 0))
    out = f.groupby(['ticker', 'second'], sort=False, dropna=False).agg(
        count=('price', 'size'), amount=('amount', 'sum'), notional=('notional', 'sum'),
        buy_notional=('buy_notional', 'sum'), sell_notional=('sell_notional', 'sum'),
        first_price=('price', 'first'), last_price=('price', 'last'),
        min_price=('price', 'min'), max_price=('price', 'max'), max_spread_bps=('spread', 'max')).reset_index()
    return out[FEATURE_COLS]

def gap_analysis(frame, previous, kind):
    ts = frame.timestamp.to_numpy(dtype='int64')
    if not len(ts):
        return pd.DataFrame(), np.array([])
    delta = np.diff(ts, prepend=previous if previous is not None else ts[0]).astype('float64')
    if previous is None:
        delta[0] = np.nan
    ix = np.flatnonzero(np.isfinite(delta))
    top = ix[np.argpartition(delta[ix], -min(TOP_N, len(ix)))[-TOP_N:]] if len(ix) else ix
    selected = np.union1d(top, np.flatnonzero(delta > BBO_GAP_US) if kind == 'bbo' else [])
    selected = selected.astype('int64')
    out = frame.iloc[selected][['source_row', 'ticker', 'timestamp']].copy()
    out['start_timestamp'] = ts[selected] - delta[selected].astype('int64')
    out['end_timestamp'] = ts[selected]
    out['gap_us'] = delta[selected].astype('int64')
    return out, delta

def price_anomaly_analysis(frame, previous_price, previous_ts, kind):
    p = ((frame.bid_price+frame.ask_price)/2 if kind == 'bbo' else frame.price).astype(float)
    p = p.where(np.isfinite(p) & (p > 0))
    prev = p.shift()
    prev_ts = frame.timestamp.shift()
    if len(p) and previous_price is not None:
        prev.iloc[0], prev_ts.iloc[0] = previous_price, previous_ts
    ret = (p / prev - 1) * 10000
    out = frame.loc[ret.notna() & ret.ne(0), ['source_row', 'ticker', 'timestamp']].copy()
    out['price'], out['previous_price'] = p.loc[out.index], prev.loc[out.index]
    out['previous_timestamp'] = prev_ts.loc[out.index].astype('int64')
    out['return_bps'] = ret.loc[out.index]
    out['abs_return_bps'] = ret.loc[out.index].abs()
    return out, ret, p

def merge_second_features(path):
    # Batches peuvent couper une seconde: count/notional addition, first/last in physical order.
    df = pl.scan_parquet(path).group_by(['ticker', 'second'], maintain_order=True).agg(
        pl.col('count').sum(), pl.col('amount').sum(), pl.col('notional').sum(),
        pl.col('buy_notional').sum(), pl.col('sell_notional').sum(),
        pl.col('first_price').drop_nulls().first(), pl.col('last_price').drop_nulls().last(),
        pl.col('min_price').min(), pl.col('max_price').max(), pl.col('max_spread_bps').max()
    ).sort(['ticker', 'second']).collect(engine='streaming')
    target = path.with_name('seconds.parquet')
    df.write_parquet(target, compression='zstd')
    path.unlink()
    return target

def audit_file(path, digest):
    ident, kind = file_id(path), file_kind(path)
    dest = OUTPUT / ident
    dest.mkdir(exist_ok=True)
    cache = dest / 'result.pkl'
    if REUSE_VERIFIED_CACHE and cache.exists():
        with open(cache, 'rb') as f:
            cached = pickle.load(f)
        if cached['sha256'] == digest and cached['version'] == AUDIT_VERSION and cached['path'] == str(path):
            print(f'Проверенный cache: {ident}', flush=True)
            return cached
    # Удаляем только прежние собственные артефакты этого source, не исходные данные.
    for old in dest.glob('*.parquet'):
        old.unlink()
    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    required = {'timestamp', 'ticker'} | ({'bid_price','bid_amount','ask_price','ask_amount'}
                if kind == 'bbo' else {'side','price','amount'})
    if not required <= set(columns):
        raise ValueError(f'{path}: missing columns {required-set(columns)}')
    result = {'id': ident, 'path': str(path), 'kind': kind,
              'exchange': 'bybit' if 'bybit' in str(path).lower() else 'binance',
              'sha256': digest, 'version': AUDIT_VERSION, 'rows': pf.metadata.num_rows,
              'columns': len(columns), 'schema': {f.name: str(f.type) for f in pf.schema_arrow},
              'preview': next(pf.iter_batches(batch_size=5)).to_pandas(),
              'tickers': {}, 'nan': Counter(), 'blanks': Counter(), 'side_counts': Counter(),
              'counts': Counter(), 'removed_rows': 0, 'examples': defaultdict(list)}
    base_sink = ParquetSink(dest / 'base_audit.parquet')
    removal_sink = ParquetSink(dest / 'removed_observations.parquet')
    duplicate_sink = ParquetSink(dest / 'duplicate_audit.parquet')
    missing_sink = ParquetSink(dest / 'missing_observations.parquet')
    ret_sink = ParquetSink(dest / 'nonzero_returns.parquet')
    second_sink = ParquetSink(dest / 'seconds_parts.parquet')
    distributions, gaps, largest, previous = {}, defaultdict(list), defaultdict(list), {}
    pending = pd.DataFrame()
    raw_previous = None
    started = time.time()
    metrics = ['delta_us', 'abs_return_bps', 'spread', 'spread_bps'] if kind == 'bbo' else [
        'delta_us', 'abs_return_bps', 'amount', 'notional']
    def temporal_chunk(frame):
        nonlocal pending
        if frame.empty:
            return
        # Carry последней timestamp-группы позволяет точно считать дубликаты на границах batches.
        combined = pd.concat([pending, frame], ignore_index=True) if len(pending) else frame
        last = combined.timestamp.iloc[-1]
        split = len(combined)-int(combined.timestamp.eq(last).iloc[::-1].cumprod().sum())
        pending = combined.iloc[split:].copy()
        work = combined.iloc[:split]
        consume_sorted(work)
    def consume_sorted(frame):
        if frame.empty:
            return
        for ticker, raw in frame.groupby('ticker', sort=False, dropna=False):
            key = str(ticker)
            state = previous.setdefault(key, {'ts': None, 'price': None, 'price_ts': None})
            tstats = result['tickers'].setdefault(key, {'rows': 0, 'counts': Counter(), 'nan': Counter()})
            tsdup = raw.timestamp.duplicated(keep=False)
            exact = raw.duplicated(subset=columns, keep=False)
            tstats['counts']['duplicate_timestamps'] += int(raw.timestamp.duplicated().sum())
            tstats['counts']['duplicate_timestamp_participants'] += int(tsdup.sum())
            tstats['counts']['exact_duplicate_rows'] += int(raw.duplicated(subset=columns).sum())
            tstats['counts']['exact_duplicate_participants'] += int(exact.sum())
            emit_issue(duplicate_sink, raw.loc[exact], 'EXACT_DUPLICATE_ROW',
                       'Полностью одинаковые исходные поля; без event ID ошибочный дубль не доказан')
            gap_rows, dt = gap_analysis(raw, state['ts'], kind)
            if len(gap_rows):
                gaps[key].append(gap_rows)
            distributions[key]['delta_us'].add(dt)
            state['ts'] = int(raw.timestamp.iloc[-1])
            rules, remove, _, _, _ = validation_masks(raw, kind)
            cleaned_df = raw.loc[~remove].copy()  # единственное удаление наблюдений, только temporary dataframe.
            if cleaned_df.empty:
                continue
            returns, ret, p = price_anomaly_analysis(cleaned_df, state['price'], state['price_ts'], kind)
            ret_sink.write(returns.reset_index(drop=True))
            distributions[key]['abs_return_bps'].add(ret.abs())
            state['price'], state['price_ts'] = float(p.iloc[-1]), int(cleaned_df.timestamp.iloc[-1])
            if kind == 'bbo':
                spread = cleaned_df.ask_price-cleaned_df.bid_price
                spread_bps = spread/p*10000
                distributions[key]['spread'].add(spread)
                distributions[key]['spread_bps'].add(spread_bps)
                top = cleaned_df[['source_row','ticker','timestamp','bid_price','ask_price','bid_amount','ask_amount']].copy()
                top['spread'], top['spread_bps'] = spread, spread_bps
                largest[key].append(top.nlargest(TOP_N, 'spread_bps'))
            else:
                cleaned_df['notional'] = cleaned_df.price * cleaned_df.amount
                distributions[key]['amount'].add(cleaned_df.amount)
                distributions[key]['notional'].add(cleaned_df.notional)
                largest[key].append(cleaned_df.nlargest(TOP_N, 'notional'))
            delay = BYBIT_DELAY_US if result['exchange']=='bybit' else 0
            second_sink.write(second_features(cleaned_df, kind, delay))
    # Pass 1: полная валидация исходного порядка; temporal предполагает monotonic и проверяет это.
    seen_reversal = False
    for number, frame in enumerate(original_batches(path)):
        rules, remove, nan, blanks, diff = validation_masks(frame, kind, raw_previous)
        raw_previous = frame.timestamp.iloc[-1]
        result['nan'].update(nan.to_dict())
        result['blanks'].update(blanks.to_dict())
        result['counts']['consecutive_equal_timestamps'] += int(diff.eq(0).sum())
        if 'side' in frame:
            result['side_counts'].update({str(k):int(v) for k,v in frame.side.value_counts(dropna=False).items()})
        for ticker, group in frame.groupby('ticker', sort=False, dropna=False):
            key = str(ticker)
            if key not in distributions:
                distributions[key] = {m: ExactDistribution(WORK / f'{ident}_{key.replace(":", "_")}_{m}.bin') for m in metrics}
            tstats = result['tickers'].setdefault(key, {'rows': 0, 'counts': Counter(), 'nan': Counter()})
            tstats['rows'] += len(group)
            tstats['nan'].update(group[columns].isna().sum().to_dict())
            good = pd.to_numeric(group.timestamp, errors='coerce').dropna()
            if len(good):
                tstats['min_timestamp'] = min(tstats.get('min_timestamp', int(good.min())), int(good.min()))
                tstats['max_timestamp'] = max(tstats.get('max_timestamp', int(good.max())), int(good.max()))
            for issue,(mask,_,_) in rules.items():
                tstats['counts'][issue] += int(mask.loc[group.index].sum())
        for issue, (mask, severity, reason) in rules.items():
            count = int(mask.sum())
            result['counts'][issue] += count
            if count:
                selected = frame.loc[mask]
                emit_issue(base_sink, selected, issue, reason, severity)
                if issue != 'MISSING_VALUES':
                    result['examples'][issue].extend(selected.source_row.iloc[:max(0, 50-len(result['examples'][issue]))].tolist())
        if rules['MISSING_VALUES'][0].any():
            missing_sink.write(frame.loc[rules['MISSING_VALUES'][0]].reset_index(drop=True))
        if remove.any():
            removed = frame.loc[remove].copy()
            removed['removal_reason'] = ['; '.join(issue for issue,(mask,sev,_) in rules.items()
                if sev=='CLEAR_DATA_ERROR' and bool(mask.loc[i])) for i in removed.index]
            removal_sink.write(removed.reset_index(drop=True))
        result['removed_rows'] += int(remove.sum())
        seen_reversal |= bool((diff < 0).any())
        valid = frame.loc[~rules['INVALID_TIMESTAMP'][0]].copy()
        if not seen_reversal:
            temporal_chunk(valid)
        if number % 20 == 0:
            print(f'{ident}: {min((number+1)*BATCH_SIZE, result["rows"]):,}/{result["rows"]:,}; {time.time()-started:.0f}s', flush=True)
    if seen_reversal:
        # External stable sorting в отдельную временную копию; raw parquet всегда read-only.
        for sink in [duplicate_sink, ret_sink, second_sink]:
            sink.close()
            sink.path.unlink(missing_ok=True)
            sink.writer = None
        for values in distributions.values():
            for dist in values.values():
                dist.handle.close()
                dist.path.unlink(missing_ok=True)
        for key in distributions:
            distributions[key] = {m: ExactDistribution(WORK / f'{ident}_{key.replace(":", "_")}_{m}.bin') for m in metrics}
            for c in ['duplicate_timestamps','duplicate_timestamp_participants','exact_duplicate_rows','exact_duplicate_participants']:
                result['tickers'][key]['counts'][c] = 0
        gaps, largest, previous, pending = defaultdict(list), defaultdict(list), {}, pd.DataFrame()
        sorted_path = WORK / f'{ident}_sorted.parquet'
        con = duckdb.connect()
        con.execute("SET memory_limit='4GB'")
        con.execute('SET temp_directory=?', [str(WORK / 'duckdb')])
        qpath = str(path).replace("'", "''")
        qtarget = str(sorted_path).replace("'", "''")
        con.execute(f"COPY (SELECT *, file_row_number AS source_row FROM read_parquet('{qpath}', file_row_number=true) WHERE timestamp IS NOT NULL ORDER BY timestamp, file_row_number) TO '{qtarget}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.close()
        for batch in pq.ParquetFile(sorted_path).iter_batches(batch_size=BATCH_SIZE):
            f = batch.to_pandas().drop(columns='file_row_number')
            f['source_file'] = str(path)
            temporal_chunk(f)
        sorted_path.unlink()
    consume_sorted(pending)
    for sink in [base_sink, removal_sink, duplicate_sink, missing_sink, ret_sink, second_sink]:
        sink.close()
    result['distributions'], result['gaps'], result['largest'] = {}, {}, {}
    for key, values in distributions.items():
        result['distributions'][key] = {}
        for metric, dist in values.items():
            print(f'{ident}/{key}: exact quantiles {metric} ({dist.n:,})', flush=True)
            result['distributions'][key][metric] = dist.finish()
        if gaps[key]:
            all_gaps = pd.concat(gaps[key], ignore_index=True).drop_duplicates('source_row')
            result['gaps'][key] = pd.concat([all_gaps.nlargest(TOP_N, 'gap_us'),
                all_gaps.loc[all_gaps.gap_us > BBO_GAP_US] if kind=='bbo' else all_gaps.iloc[:0]]).drop_duplicates('source_row')
        else:
            result['gaps'][key] = pd.DataFrame(columns=['source_row','ticker','timestamp','start_timestamp','end_timestamp','gap_us'])
        result['largest'][key] = pd.concat(largest[key], ignore_index=True).nlargest(
            TOP_N, 'spread_bps' if kind=='bbo' else 'notional') if largest[key] else pd.DataFrame()
    result['seconds_path'] = str(merge_second_features(second_sink.path)) if second_sink.path.exists() else None
    result['elapsed_seconds'] = time.time()-started
    with open(cache, 'wb') as f:
        pickle.dump(result, f)
    gc.collect()
    print(f'Завершён {ident}: {result["elapsed_seconds"]:.1f}s', flush=True)
    return result

def read_seconds(result, ticker=None):
    f = pl.read_parquet(result['seconds_path']).to_pandas()
    return f.loc[f.ticker.eq(ticker)].reset_index(drop=True) if ticker is not None else f

def cross_dataset_validation(event_seconds, feature_frame, prefix):
    """Exact observed totals in [floor(event)-5s, floor(event)+6s), effective Bybit axis."""
    centers = np.asarray(event_seconds, dtype=np.int64)
    f = feature_frame.sort_values('second').reset_index(drop=True)
    out = pd.DataFrame({'center_second': centers})
    fields = ['count','amount','notional','buy_notional','sell_notional']
    if f.empty:
        for col in fields:
            out[prefix+'_'+col] = np.nan
        out[prefix+'_coverage'] = False
        for col in ['move_bps','min_price','max_price','max_spread_bps','before_price','after_price']:
            out[prefix+'_'+col] = np.nan
        return out
    seconds = f.second.to_numpy(dtype=np.int64)
    lo = np.searchsorted(seconds, centers-WINDOW_SECONDS, side='left')
    hi = np.searchsorted(seconds, centers+WINDOW_SECONDS+1, side='left')
    for col in fields:
        cumulative = np.r_[0., f[col].fillna(0).to_numpy(dtype=float).cumsum()]
        out[prefix+'_'+col] = cumulative[hi]-cumulative[lo]
    out[prefix+'_coverage'] = ((centers-WINDOW_SECONDS >= seconds.min()) &
                              (centers+WINDOW_SECONDS <= seconds.max()))
    # 11 фиксированных секунд, без forward-fill через отсутствующие наблюдения.
    offsets = np.arange(-WINDOW_SECONDS, WINDOW_SECONDS+1)
    wanted = centers[:,None]+offsets
    idx = np.searchsorted(seconds, wanted)
    clipped = np.minimum(idx, len(seconds)-1)
    exists = (idx < len(seconds)) & (seconds[clipped] == wanted)
    def matrix(col):
        return np.where(exists, f[col].to_numpy(dtype=float)[clipped], np.nan)
    last_price = matrix('last_price')
    # Median наблюдаемых секунд до/после: устойчивость, а не только single tick.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        before = np.nanmedian(last_price[:,:WINDOW_SECONDS], axis=1)
        after = np.nanmedian(last_price[:,WINDOW_SECONDS+1:], axis=1)
        out[prefix+'_before_price'], out[prefix+'_after_price'] = before, after
        out[prefix+'_move_bps'] = (after/before-1)*10000
        out[prefix+'_min_price'] = np.nanmin(matrix('min_price'), axis=1)
        out[prefix+'_max_price'] = np.nanmax(matrix('max_price'), axis=1)
        out[prefix+'_max_spread_bps'] = np.nanmax(matrix('max_spread_bps'), axis=1)
    out[prefix+'_active_seconds'] = exists.sum(axis=1)
    return out

def discover_events(result):
    """Все returns > точного p99.9 + top 10; все liquidation extremes/episodes + top 10 spreads."""
    dest = OUTPUT / result['id']
    events = []
    for ticker, ds in result['distributions'].items():
        path = dest/'nonzero_returns.parquet'
        if path.exists():
            lazy = pl.scan_parquet(path).filter(pl.col('ticker')==ticker)
            cut = ds['abs_return_bps'].get('p99.9', np.inf)
            extreme = lazy.filter(pl.col('abs_return_bps') > cut).collect(engine='streaming').to_pandas()
            top = lazy.top_k(TOP_N, by='abs_return_bps').collect(engine='streaming').to_pandas()
            rows = pd.concat([extreme,top], ignore_index=True).drop_duplicates('source_row')
            rows['issue_type'], rows['threshold_bps'] = 'EXTREME_PRICE_MOVE', cut
            events.append(rows)
        if result['kind']=='bbo':
            cut = ds['spread_bps'].get('p99.9', np.inf)
            rows = (pl.scan_parquet(result['path']).with_row_index('source_row')
                .filter(pl.col('ticker')==ticker).with_columns(
                    ((pl.col('ask_price')+pl.col('bid_price'))/2).alias('price'),
                    ((pl.col('ask_price')-pl.col('bid_price')) /
                     ((pl.col('ask_price')+pl.col('bid_price'))/2)*10000).alias('spread_bps'))
                .filter((pl.col('spread_bps')>cut) & (pl.col('bid_price')>0) & (pl.col('ask_price')>0))
                .select(['source_row','ticker','timestamp','price','spread_bps'])
                .collect(engine='streaming').to_pandas())
            top = result['largest'][ticker].copy()
            top['price'] = (top.bid_price+top.ask_price)/2
            rows = pd.concat([rows,top], ignore_index=True).drop_duplicates('source_row')
            rows['issue_type'] = 'EXTREME_SPREAD'
            events.append(rows)
        elif result['kind']=='liquidations':
            frame = pd.read_parquet(result['path'])
            frame['source_row'] = np.arange(len(frame), dtype='int64')
            frame['source_file'] = result['path']
            _, remove, _, _, _ = validation_masks(frame, result['kind'])
            f = frame.loc[~remove & frame.ticker.eq(ticker)].copy()
            f['notional'] = f.price*f.amount
            big = f.loc[f.notional>ds['notional'].get('p99.9',np.inf)]
            big = pd.concat([big, f.nlargest(TOP_N,'notional')]).drop_duplicates('source_row')
            big['issue_type'] = 'LARGE_LIQUIDATION'
            events.append(big)
            delay = BYBIT_DELAY_US if result['exchange']=='bybit' else 0
            f['center_second'] = (f.timestamp+delay)//1_000_000
            f['episode_start_second'] = (f.center_second//10)*10
            bins = f.groupby('episode_start_second').agg(count=('price','size'), notional=('notional','sum'))
            # Кластер: верхняя 0.1% активных 10-секундных bins по notional/count, минимум 2 события.
            # Это также выявляет серии Binance liquidation observations, идущие раз в секунду.
            clusters = bins.loc[((bins.notional>bins.notional.quantile(.999)) |
                                  (bins['count']>bins['count'].quantile(.999))) & (bins['count']>=2)]
            clustered = f.loc[f.episode_start_second.isin(clusters.index)].copy()
            clustered['episode_notional'] = clustered.episode_start_second.map(clusters.notional)
            clustered['episode_count'] = clustered.episode_start_second.map(clusters['count'])
            clustered['diagnostic_center_timestamp'] = (clustered.episode_start_second+5)*1_000_000
            clustered['issue_type'] = 'LIQUIDATION_CLUSTER'
            events.append(clustered)
            bins.sort_values('notional',ascending=False).head(100).to_parquet(dest/f'{symbol(ticker)}_episodes.parquet')
        else:
            big = result['largest'][ticker].copy()
            big['issue_type'] = 'LARGE_TRADE_NOTIONAL'
            events.append(big)
    if not events:
        return pd.DataFrame()
    e = pd.concat(events, ignore_index=True)
    e['source_row'] = e.source_row.astype('int64')
    e['source_file'], e['source_id'] = result['path'], result['id']
    e['symbol'] = e.ticker.map(symbol)
    e['raw_timestamp'] = e.timestamp.astype('int64')
    e['effective_timestamp'] = e.raw_timestamp + (BYBIT_DELAY_US if result['exchange']=='bybit' else 0)
    e['diagnostic_center_timestamp'] = e.get('diagnostic_center_timestamp',e.effective_timestamp).fillna(e.effective_timestamp).astype('int64')
    e['center_second'] = e.diagnostic_center_timestamp//1_000_000
    e['datetime_utc'] = utc(e.timestamp)
    e['effective_datetime_utc'] = utc(e.effective_timestamp)
    return e.drop_duplicates(['source_row','issue_type'])

def classify_events(e):
    """Консервативная evidence-based эвристика; она НЕ удаляет статистические аномалии."""
    e = e.copy()
    b, t = e.bbo_move_bps, e.trades_move_bps
    direction = np.sign(e.get('return_bps',pd.Series(np.nan,index=e.index)))
    # Для liquidation episode направление задаёт суммарное order-side давление.
    pressure = (e.binance_liq_buy_notional-e.binance_liq_sell_notional +
                e.bybit_liq_buy_notional-e.bybit_liq_sell_notional)
    is_liq = e.issue_type.isin(['LARGE_LIQUIDATION','LIQUIDATION_CLUSTER'])
    direction = direction.where(~is_liq, np.sign(pressure))
    direction = direction.fillna(np.sign(b))
    threshold = np.maximum(.1, e.get('return_bps',pd.Series(0.,index=e.index)).abs().fillna(0)*.1)
    # Уровень подозрительной цены должен согласоваться с независимо наблюдаемым BBO.
    # Для liquidation order price допускаем более широкое отклонение (250 bps);
    # это diagnostic tolerance, не верхняя граница допустимой рыночной цены.
    anchor = (e.bbo_min_price+e.bbo_max_price)/2
    distance = np.maximum(e.bbo_min_price-e.price, e.price-e.bbo_max_price).clip(lower=0)/anchor*10000
    liquidation_source = e.source_id.str.contains('liquidations')
    tolerance = np.maximum(np.where(liquidation_source,250.,20.), b.abs().fillna(0)*2)
    e['price_distance_from_bbo_bps'] = distance
    e['price_level_supported'] = distance <= tolerance
    if 'previous_price' in e:
        previous_distance = np.maximum(e.bbo_min_price-e.previous_price,
            e.previous_price-e.bbo_max_price).clip(lower=0)/anchor*10000
        # Большой elapsed time между liquidation observations делает previous price нерелевантной
        # для локального окна; сравниваем её только если она внутри окна.
        recent = e.previous_timestamp >= (e.center_second-WINDOW_SECONDS)*1_000_000
        e['previous_price_level_supported'] = (~recent) | (previous_distance<=tolerance)
    else:
        e['previous_price_level_supported'] = True
    e['liquidation_to_trade_notional_ratio'] = e.get('notional',pd.Series(np.nan,index=e.index))/e.trades_notional.replace(0,np.nan)
    unsupported_size = is_liq & (e.liquidation_to_trade_notional_ratio>100)
    agreement = ((b*direction > threshold) & (t*direction > threshold) &
                 (e.trades_notional > 0) & e.bbo_coverage & e.trades_coverage &
                 e.price_level_supported & e.previous_price_level_supported & ~unsupported_size)
    # Достаточно согласованного BBO + trades; отсутствие редких Bybit liquidation не опровергает событие.
    e['cross_market_support'] = e.bybit_liq_count.fillna(0)>0
    e['severity'] = np.where(agreement, 'PLAUSIBLE_MARKET_EVENT', 'SUSPICIOUS')
    e['action'] = e.severity.map(ACTIONS)
    e['reason'] = np.where(agreement,
        'Согласованное устойчивое направление BBO/trades и наблюдаемый turnover; вероятное рыночное событие',
        'Недостаточно согласованных подтверждений или движение неустойчиво; нужна проверка контекста, не доказанная ошибка')
    e.loc[~e.price_level_supported,'reason'] = 'Уровень цены не подтверждён диапазоном BBO в окне; возможен bad tick / особенность liquidation order price; сохранить для проверки'
    e.loc[unsupported_size,'reason'] = 'Liquidation notional превышает наблюдаемый trade turnover окна более чем в 100 раз; недостаточно подтверждений размера'
    e['confidence'] = np.where(agreement & e.cross_market_support, 'medium: cross-exchange liquidation support',
                      np.where(agreement, 'medium: Binance BBO + trades', 'low: unresolved'))
    e['window_start_utc'] = utc((e.center_second-WINDOW_SECONDS)*1_000_000)
    e['window_end_utc_exclusive'] = utc((e.center_second+WINDOW_SECONDS+1)*1_000_000)
    return e

def build_event_evidence(results):
    evidence_paths = []
    for sym in sorted({symbol(t) for r in results for t in r['tickers']}):
        group = [r for r in results if any(symbol(t)==sym for t in r['tickers'])]
        frames = [discover_events(r) for r in group]
        events = pd.concat([f.loc[f.symbol==sym] for f in frames if not f.empty],ignore_index=True)
        centers = np.sort(events.center_second.unique())
        evidence = pd.DataFrame({'center_second':centers})
        for prefix,exchange,kind in [('bbo','binance','bbo'),('trades','binance','trades'),
                                   ('binance_liq','binance','liquidations'),('bybit_liq','bybit','liquidations')]:
            source = next((r for r in group if r['exchange']==exchange and r['kind']==kind), None)
            f = read_seconds(source) if source else pd.DataFrame()
            if len(f):
                f = f.loc[f.ticker.map(symbol)==sym]
            v = cross_dataset_validation(centers, f, prefix)
            evidence = evidence.merge(v,on='center_second',validate='1:1')
            del f, v
            gc.collect()
        events = events.merge(evidence,on='center_second',validate='m:1')
        events = classify_events(events)
        out = OUTPUT/f'{sym}_event_evidence.parquet'
        events.to_parquet(out,index=False)
        evidence_paths.append(out)
        for ident, f in events.groupby('source_id'):
            audit = f[['source_file','source_row','ticker','timestamp','datetime_utc',
                       'issue_type','severity','reason','action']]
            audit.to_parquet(OUTPUT/ident/'event_audit.parquet',index=False)
        print(f'{sym}: {len(events):,} event/issue rows, {len(centers):,} диагностических окон',flush=True)
        del events, evidence
        gc.collect()
    return evidence_paths

def exact_interval_activity(source, intervals, delay=0):
    """All rows, exact open interval (left,right); no resampling boundary approximation."""
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute('SET threads=4')
    con.register('intervals', intervals)
    path = source['path'].replace("'", "''")
    sql = f"""
      SELECT i.gap_id, count(p.timestamp) AS count,
             count(DISTINCT p.timestamp) AS distinct_timestamps,
             coalesce(sum(p.price*p.amount),0) AS notional,
             count(p.timestamp) FILTER (WHERE p.price IS NULL OR p.amount IS NULL) AS unknown_notional_count,
             coalesce(sum(p.price*p.amount) FILTER (WHERE p.side='buy'),0) AS buy_notional,
             coalesce(sum(p.price*p.amount) FILTER (WHERE p.side='sell'),0) AS sell_notional
      FROM intervals i LEFT JOIN read_parquet('{path}') p
        ON p.timestamp > i.start_timestamp - {delay}
        AND p.timestamp < i.end_timestamp - {delay}
        AND replace(lower(p.ticker),'perp:','') = i.symbol
        AND (p.price IS NULL OR (isfinite(p.price) AND p.price>0))
        AND (p.amount IS NULL OR (isfinite(p.amount) AND p.amount>=0))
        AND p.side IN ('buy','sell')
      GROUP BY i.gap_id
    """
    out = con.execute(sql).df()
    con.close()
    lo = min(v['min_timestamp'] for v in source['tickers'].values())+delay
    hi = max(v['max_timestamp'] for v in source['tickers'].values())+delay
    bounds = intervals.set_index('gap_id')
    out['coverage'] = [(bounds.loc[i,'start_timestamp']>=lo and bounds.loc[i,'end_timestamp']<=hi)
                       for i in out.gap_id]
    return out

def investigate_bbo_gaps(results):
    all_rows = []
    for source in [r for r in results if r['kind']=='bbo']:
        gaps = pd.concat(list(source['gaps'].values()),ignore_index=True)
        if gaps.empty:
            continue
        gaps['symbol'] = gaps.ticker.map(symbol)
        gaps['source_file'], gaps['source_id'] = source['path'],source['id']
        gaps['gap_id'] = [f'{source["id"]}:{row}' for row in gaps.source_row]
        gaps['gap_seconds'] = gaps.gap_us/1e6
        intervals = gaps[['gap_id','symbol','start_timestamp','end_timestamp']]
        for prefix,exchange,kind,delay in [('trades','binance','trades',0),
                ('binance_liq','binance','liquidations',0),
                ('bybit_liq','bybit','liquidations',BYBIT_DELAY_US),
                ('bybit_raw','bybit','liquidations',0)]:
            matches = [r for r in results if r['exchange']==exchange and r['kind']==kind and
                       any(symbol(t) in set(gaps.symbol) for t in r['tickers'])]
            if not matches:
                for name in ['count','notional','unknown_notional_count','buy_notional','sell_notional']:
                    gaps[prefix+'_'+name] = np.nan
                gaps[prefix+'_coverage'] = False
                continue
            values = pd.concat([exact_interval_activity(r,intervals.loc[intervals.symbol.isin(
                [symbol(t) for t in r['tickers']])],delay) for r in matches])
            values = values.rename(columns={c:prefix+'_'+c for c in values if c!='gap_id'})
            gaps = gaps.merge(values,on='gap_id',validate='1:1')
        rates = {}
        for r in results:
            if r['kind']=='trades':
                for ticker,v in r['tickers'].items():
                    rates[symbol(ticker)] = v['rows']/max(1,(v['max_timestamp']-v['min_timestamp'])/1e6)
        gaps['trades_per_second'] = gaps.trades_count/gaps.gap_seconds
        gaps['trade_rate_vs_full_period'] = gaps.trades_per_second/gaps.symbol.map(rates)
        covered = gaps[['trades_coverage','binance_liq_coverage','bybit_liq_coverage']].all(axis=1)
        strong = covered & (gaps.trades_count>=100) & (gaps.trades_distinct_timestamps>=10) & (gaps.trade_rate_vs_full_period>=.1)
        quiet = covered & gaps[['trades_count','binance_liq_count','bybit_liq_count']].eq(0).all(axis=1)
        gaps['finding'] = np.select([strong,quiet],['BBO DATA / FEED ISSUE',
                            'PLAUSIBLE QUIET INTERVAL'],default='UNRESOLVED GAP')
        gaps['severity'] = np.where(quiet,'PLAUSIBLE_MARKET_EVENT','SUSPICIOUS')
        gaps['reason'] = np.select([strong,quiet],[
            'Внутри отсутствующего BBO есть существенная активность Binance trades; сильный признак проблемы BBO feed',
            'В покрываемом источниками интервале нет наблюдаемых trades/liquidations; отсутствие updates возможно, не доказано'],
            default='Длительный gap; активность или покрытие недостаточны для уверенного вывода')
        gaps['action'] = gaps.severity.map(ACTIONS)
        gaps['issue_type'] = 'BBO_GAP'
        gaps['datetime_utc'] = utc(gaps.timestamp)
        gaps['start_utc'],gaps['end_utc'] = utc(gaps.start_timestamp),utc(gaps.end_timestamp)
        gaps[['source_file','source_row','ticker','timestamp','datetime_utc','issue_type','severity',
              'reason','action']].to_parquet(OUTPUT/source['id']/'gap_audit.parquet',index=False)
        all_rows.append(gaps)
        print(f'{source["id"]}: exact cross-source comparison for {len(gaps)} gaps',flush=True)
    out = pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    out.to_parquet(OUTPUT/'bbo_gap_evidence.parquet',index=False)
    return out

def investigate_trade_gaps(results, bbo_gaps):
    records=[]
    for source in [r for r in results if r['kind']=='trades']:
        for ticker,gaps in source['gaps'].items():
            threshold=source['distributions'][ticker]['delta_us'].get('p99.9',np.inf)
            bbo=next((r for r in results if r['kind']=='bbo' and
                      any(symbol(t)==symbol(ticker) for t in r['tickers'])),None)
            for _,gap in gaps.loc[gaps.gap_us>threshold].iterrows():
                row=gap.to_dict()
                row.update(source_file=source['path'],source_id=source['id'],
                           issue_type='TRADES_GAP',severity='SUSPICIOUS',action='KEPT_FOR_INVESTIGATION',
                           empirical_p999_gap_us=threshold)
                if bbo:
                    quotes=pd.read_parquet(bbo['path'],columns=['timestamp','ticker'],filters=[
                        ('timestamp','>',int(gap.start_timestamp)),('timestamp','<',int(gap.end_timestamp))])
                    row['bbo_updates_inside_gap']=int(quotes.ticker.map(symbol).eq(symbol(ticker)).sum())
                else:
                    row['bbo_updates_inside_gap']=np.nan
                row['reason']=('Top gap trades относительно собственного распределения; одновременно нет наблюдаемых BBO updates: quiet interval / общая проблема покрытия не различены'
                    if row['bbo_updates_inside_gap']==0 else
                    'Top gap trades относительно собственного распределения; BBO updates продолжаются. Пауза исполнения или проблема trade feed требуют проверки')
                row['datetime_utc']=utc(gap.timestamp)
                row['start_utc'],row['end_utc']=utc(gap.start_timestamp),utc(gap.end_timestamp)
                records.append(row)
    out=pd.DataFrame(records)
    out.to_parquet(OUTPUT/'trade_gap_evidence.parquet',index=False)
    if len(out):
        for ident,f in out.groupby('source_id'):
            f[['source_file','source_row','ticker','timestamp','datetime_utc','issue_type','severity',
               'reason','action']].to_parquet(OUTPUT/ident/'trade_gap_audit.parquet',index=False)
    return out

def audit_page(source_file=None, issue_type=None, severity=None, offset=0, limit=100):
    q = audit_log
    for field,value in [('source_file',source_file),('issue_type',issue_type),('severity',severity)]:
        if value is not None:
            q = q.filter(pl.col(field)==value)
    return q.slice(offset,limit).collect(engine='streaming').to_pandas()

def evidence_page(source_id=None, issue_type=None, limit=20):
    q = event_evidence
    if source_id:
        q = q.filter(pl.col('source_id')==source_id)
    if issue_type:
        q = q.filter(pl.col('issue_type')==issue_type)
    return q.limit(limit).collect(engine='streaming').to_pandas()

def severity_counts(result):
    # Exact unique observation counts без гигантского group_by по всем duplicate rows.
    dest = OUTPUT/result['id']
    small_paths = [p for p in dest.glob('*_audit.parquet') if p.name!='duplicate_audit.parquet']
    small = (pl.scan_parquet(small_paths).select(['source_row','severity']).unique().collect(engine='streaming')
             if small_paths else pl.DataFrame(schema={'source_row':pl.Int64,'severity':pl.String}))
    counts = {severity:small.filter(pl.col('severity')==severity).height for severity in SEVERITIES}
    duplicate_path = dest/'duplicate_audit.parquet'
    if duplicate_path.exists():
        ndup = pq.ParquetFile(duplicate_path).metadata.num_rows
        suspects = small.filter(pl.col('severity')=='SUSPICIOUS').select('source_row').lazy()
        overlap = pl.scan_parquet(duplicate_path).select('source_row').join(
            suspects,on='source_row',how='semi').select(pl.len()).collect(engine='streaming').item()
        counts['SUSPICIOUS'] += ndup-overlap
    return counts

def summary_generation(results):
    rows = []
    for r in results:
        c = r['counts']
        low = min(v['min_timestamp'] for v in r['tickers'].values())
        high = max(v['max_timestamp'] for v in r['tickers'].values())
        row = {'file':r['path'],'rows':r['rows'],'columns':r['columns'],
            'date range':f'{utc(low)} → {utc(high)}','NaN count':sum(r['nan'].values()),
            'fully empty rows':c['FULLY_EMPTY_ROW'],'invalid timestamp count':c['INVALID_TIMESTAMP'],
            'timestamp reversals':c['TIMESTAMP_ORDERING_ISSUE'],
            'duplicate timestamps':sum(v['counts']['duplicate_timestamps'] for v in r['tickers'].values()),
            'exact duplicate rows':sum(v['counts']['exact_duplicate_rows'] for v in r['tickers'].values()),
            'consecutive equal timestamps':c['consecutive_equal_timestamps'],
            'invalid prices':sum(c[k] for k in ['INVALID_PRICE','INVALID_BID_PRICE','INVALID_ASK_PRICE']),
            'invalid amounts':sum(c[k] for k in ['INVALID_AMOUNT','INVALID_BID_AMOUNT','INVALID_ASK_AMOUNT']),
            'invalid side values':c['INVALID_SIDE'],'crossed BBO count':c['CROSSED_BBO'],
            'zero size BBO count':0,
            'largest time gap':max(d['delta_us'].get('maximum',np.nan)/1e6 for d in r['distributions'].values()),
            'number of suspicious large gaps':0,
            'number of extreme price moves':int(event_evidence.filter((pl.col('source_id')==r['id']) &
                 (pl.col('issue_type')=='EXTREME_PRICE_MOVE')).select(pl.len()).collect().item()),
            'number of removed observations':r['removed_rows']}
        p = OUTPUT/r['id']/'base_audit.parquet'
        if p.exists():
            row['zero size BBO count'] = (pl.scan_parquet(p).filter(pl.col('issue_type').is_in(
                ['ZERO_BID_AMOUNT','ZERO_ASK_AMOUNT'])).select(pl.col('source_row').n_unique()).collect().item())
            for name,issues in [('invalid prices',['INVALID_PRICE','INVALID_BID_PRICE','INVALID_ASK_PRICE']),
                                ('invalid amounts',['INVALID_AMOUNT','INVALID_BID_AMOUNT','INVALID_ASK_AMOUNT'])]:
                row[name] = (pl.scan_parquet(p).filter(pl.col('issue_type').is_in(issues))
                    .select(pl.col('source_row').n_unique()).collect().item())
        if not gap_evidence.empty:
            row['number of suspicious large gaps'] = int(((gap_evidence.source_id==r['id']) &
                (gap_evidence.gap_us>BBO_GAP_US) & (gap_evidence.severity=='SUSPICIOUS')).sum())
        if r['kind']=='trades' and not trade_gap_evidence.empty:
            row['number of suspicious large gaps']=int(trade_gap_evidence.source_id.eq(r['id']).sum())
        for severity,n in severity_counts(r).items():
            row[f'number of {severity} observations'] = n
        rows.append(row)
    return pd.DataFrame(rows)

def histogram(result, ticker, metrics):
    fig, axes = plt.subplots(1,len(metrics),figsize=(6*len(metrics),3.3),squeeze=False)
    for ax,metric in zip(axes[0],metrics):
        d = result['distributions'][ticker][metric]
        edges,counts = np.array(d['hist_edges']),np.array(d['hist_counts'])
        if len(counts):
            ax.stairs(counts,edges,fill=True,color=COLORS['bbo'],alpha=.8)
            ax.set_yscale('log')
        unit={'delta_us':'microseconds','abs_return_bps':'bps','spread':'USDT',
              'spread_bps':'bps','amount':'base-asset units','notional':'USDT'}[metric]
        ax.set_xlabel(f'sign(x) × log10(1+|x|); x in {unit}')
        ax.set_ylabel('Observations (log scale)')
        ax.set_title(f'{ticker} · {metric}')
    fig.suptitle(f'{result["exchange"]} / {result["kind"]} — full population',y=1.03)
    fig.tight_layout()
    (OUTPUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUTPUT/'figures'/f'{result["id"]}_{symbol(ticker)}_{"_".join(metrics)}.png',dpi=130,bbox_inches='tight')
    display(fig)
    plt.close(fig)

def render_schema(result):
    heading(result['id'],2)
    display(Markdown(f'**Полный путь:** `{result["path"]}`  \n'
                     f'**Размер:** {result["rows"]:,} строк × {result["columns"]} колонок. '
                     f'**Уникальных ticker:** {len(result["tickers"])}.'))
    display(pd.DataFrame({'parquet_dtype':pd.Series(result['schema']),
                          'pandas_dtype':result['preview'].dtypes.astype(str)}))
    display(result['preview'])
    display(pd.DataFrame([{'ticker':t,'rows':v['rows'], 'min_timestamp':v['min_timestamp'],
         'max_timestamp':v['max_timestamp'],'min_datetime_utc':utc(v['min_timestamp']),
         'max_datetime_utc':utc(v['max_timestamp'])} for t,v in result['tickers'].items()]))
    if result['side_counts']:
        display(pd.Series(result['side_counts'],name='side_count').to_frame())

def render_checks(result):
    heading(f'Проверки: {result["id"]}',2)
    nan = pd.Series(result['nan'],name='NaN_count')
    display(pd.DataFrame({'NaN_count':nan,'NaN_share':nan/result['rows'],
                          'empty_string_count':pd.Series(result['blanks'])}))
    display(pd.Series(result['counts'],name='count').to_frame())
    display(Markdown('Timestamp монотонно не убывает в исходном порядке: **'+
                     str(result['counts']['TIMESTAMP_ORDERING_ISSUE']==0)+'**.'))
    for ticker,v in result['tickers'].items():
        heading(ticker,3)
        display(pd.Series(v['counts'],name='count').to_frame())
        display(pd.DataFrame({'NaN_count':pd.Series(v['nan']),
                              'NaN_share':pd.Series(v['nan'])/v['rows']}))
        ds = result['distributions'][ticker]
        d = {k:v for k,v in ds['delta_us'].items() if not k.startswith('hist_')}
        display(pd.DataFrame({'microseconds':d,'milliseconds':{k:v/1000 for k,v in d.items()},
                             'seconds':{k:v/1e6 for k,v in d.items()}}).drop(
                                 index=['count','zero_count','above_p999'],errors='ignore'))
        gaps = result['gaps'][ticker].copy()
        gaps['start_utc'],gaps['end_utc'] = utc(gaps.start_timestamp),utc(gaps.end_timestamp)
        display(gaps.nlargest(TOP_N,'gap_us'))
        histogram(result,ticker,['delta_us'])
        metrics = ['spread','spread_bps'] if result['kind']=='bbo' else ['amount','notional']
        display(pd.DataFrame({m:{k:v for k,v in ds[m].items() if not k.startswith('hist_')}
                               for m in metrics}))
        histogram(result,ticker,metrics)
        display(result['largest'][ticker])
        display(pd.Series({k:v for k,v in ds['abs_return_bps'].items() if not k.startswith('hist_')},
                          name='absolute consecutive return (bps)').to_frame())
    # Все reversals отдельно, все нарушения BBO отдельно; не скрываем за aggregate counts.
    base = OUTPUT/result['id']/'base_audit.parquet'
    if base.exists():
        q = pl.read_parquet(base).to_pandas()
        for issue,f in q.groupby('issue_type'):
            heading(issue,3)
            display(f[AUDIT_COLUMNS].head(20))
            selected = f.source_row.unique()
            # Полные контексты сохраняются для КАЖДОЙ строки, интерфейс notebook ограничивает только preview.
            context_path = OUTPUT/result['id']/f'contexts_{issue}.html'
            with open(context_path,'w') as handle:
                handle.write('<meta charset="utf-8"><h1>'+issue+'</h1>')
                for row_id in selected:
                    handle.write(f'<h2>source_row={row_id}</h2>'+raw_rows(result['path'],[row_id],3).to_html(index=False))
            display(HTML(f'<a href="{context_path.relative_to(OUTPUT.parent)}">Все {len(selected):,} контекстов {issue}</a>'))
            if issue=='TIMESTAMP_ORDERING_ISSUE':
                show_context(result,selected)
            elif result['kind']=='bbo':
                if len(selected)<=30:
                    show_context(result,selected)
                else:
                    show_context(result,selected[:5])
    if result['counts']['MISSING_VALUES']:
        display(pd.read_parquet(OUTPUT/result['id']/'missing_observations.parquet'))
    duplicates = OUTPUT/result['id']/'duplicate_audit.parquet'
    if duplicates.exists():
        heading('Примеры полностью одинаковых строк — сохранены',3)
        ids = pl.scan_parquet(duplicates).select('source_row').head(3).collect().to_series().to_list()
        display(raw_rows(result['path'],ids,radius=1))

def render_liquidations(result):
    heading(f'Liquidation pressure: {result["id"]}',2)
    f = pd.concat(list(iter_cleaned(Path(result['path']))),ignore_index=True)
    f['notional'] = f.price*f.amount
    display(f.groupby(['ticker','side'],dropna=False).agg(count=('side','size'),
            amount=('amount','sum'),notional=('notional','sum')))
    for ticker,g in f.groupby('ticker'):
        display(g.nlargest(TOP_N,'notional'))
        episodes = pd.read_parquet(OUTPUT/result['id']/f'{symbol(ticker)}_episodes.parquet')
        episodes['effective_datetime_utc'] = utc(episodes.index.to_numpy()*1_000_000)
        display(episodes.head(20))
        timeline = f.loc[f.ticker==ticker].copy()
        timeline['datetime_utc'] = utc(timeline.timestamp+(BYBIT_DELAY_US if result['exchange']=='bybit' else 0))
        daily = timeline.set_index('datetime_utc').groupby('side').resample('1D').notional.sum().unstack(0).fillna(0)
        fig,ax=plt.subplots(figsize=(11,3.4))
        for side in ['buy','sell']:
            if side in daily:
                ax.plot(daily.index,daily[side],label=side,color=COLORS[side],linestyle='-' if side=='sell' else '--')
        ax.set(title=f'{result["exchange"]} {ticker} · daily liquidation notional',ylabel='USDT',xlabel='UTC (effective for Bybit)')
        ax.legend(); fig.tight_layout(); display(fig); plt.close(fig)

def window_rows(result, sym, start_us, end_us, causal=True):
    delay = BYBIT_DELAY_US if causal and result['exchange']=='bybit' else 0
    f = pd.read_parquet(result['path'], filters=[('timestamp','>=',int(start_us-delay)),
                                                ('timestamp','<',int(end_us-delay))])
    f = f.loc[f.ticker.map(symbol)==sym].copy()
    f['raw_timestamp'] = f.timestamp
    f['effective_timestamp'] = f.timestamp+(BYBIT_DELAY_US if result['exchange']=='bybit' else 0)
    f['datetime_utc'] = utc(f.effective_timestamp if causal else f.timestamp)
    return f.sort_values('timestamp',kind='stable')

def diagnostic_window(sym, effective_timestamp, seconds=5, title='', save_name=None):
    start,end = int(effective_timestamp-seconds*1e6),int(effective_timestamp+(seconds*1e6)+1)
    frames = {}
    for r in results:
        if any(symbol(t)==sym for t in r['tickers']):
            frames[(r['exchange'],r['kind'])] = window_rows(r,sym,start,end)
    fig, axes=plt.subplots(5,1,figsize=(12,13),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1,1]})
    bbo=frames.get(('binance','bbo'),pd.DataFrame())
    trades=frames.get(('binance','trades'),pd.DataFrame())
    if not bbo.empty:
        mid=(bbo.bid_price+bbo.ask_price)/2
        mid=mid.where((bbo.bid_price>0)&(bbo.ask_price>0))
        # Разрыв линии при >1s без нового BBO: не рисуем выдуманный continuous mid.
        line_mid=mid.copy(); line_mid.loc[bbo.timestamp.diff()>1_000_000]=np.nan
        axes[0].plot(bbo.datetime_utc,line_mid,color=COLORS['bbo'],label='Binance mid',lw=1)
        plotted_spread=(bbo.ask_price-bbo.bid_price)/mid*10000
        # Spread тоже не интерполируется через отсутствующие BBO updates.
        plotted_spread=plotted_spread.copy()
        plotted_spread.loc[bbo.timestamp.diff()>1_000_000]=np.nan
        axes[4].plot(bbo.datetime_utc,plotted_spread,color=COLORS['bbo'])
        if plotted_spread.notna().any():
            axes[4].set_ylim(bottom=min(0.,float(plotted_spread.min())*1.1))
    else:
        axes[0].text(.01,.94,'No BBO observations in this window',transform=axes[0].transAxes)
        axes[4].text(.01,.8,'No BBO observations',transform=axes[4].transAxes)
    if not trades.empty:
        axes[0].scatter(trades.datetime_utc,trades.price,s=3,alpha=.3,color=COLORS['trades'],label='Binance trades')
        trades['notional']=trades.price*trades.amount
        bars=trades.set_index('datetime_utc').resample('200ms').agg(amount=('amount','sum'),notional=('notional','sum'))
        axes[1].bar(bars.index,bars.notional,width=.16/86400,color=COLORS['trades'],label='Trade notional')
        a2=axes[1].twinx(); a2.plot(bars.index,bars.amount,color=COLORS['buy'],lw=1,label='Trade volume')
        a2.set_ylabel('Base-asset amount')
    else:
        axes[1].text(.01,.8,'No trade observations in this window',transform=axes[1].transAxes)
    totals=[]
    for ax,exchange in [(axes[2],'binance'),(axes[3],'bybit')]:
        liq=frames.get((exchange,'liquidations'),pd.DataFrame())
        if not liq.empty:
            liq['notional']=liq.price*liq.amount
            for side,marker in [('buy','^'),('sell','v')]:
                v=liq.loc[liq.side==side]
                ax.scatter(v.datetime_utc,v.notional*(1 if side=='buy' else -1),s=22,
                           marker=marker,color=COLORS[side],label=side)
            totals.append({'source':exchange+' liquidations','count':len(liq),
                'notional':liq.notional.sum(),'buy_notional':liq.loc[liq.side=='buy','notional'].sum(),
                'sell_notional':liq.loc[liq.side=='sell','notional'].sum(),
                'first_price':liq.price.iloc[0],'last_price':liq.price.iloc[-1],
                'min_price':liq.price.min(),'max_price':liq.price.max()})
            ax.legend(loc='upper right')
        else:
            ax.text(.01,.8,'No observed liquidation events',transform=ax.transAxes)
        ax.axhline(0,color='#999999',lw=.6)
        ax.set_ylabel(f'{exchange}\n± USDT')
    axes[0].set_ylabel('Price (USDT)'); axes[0].legend(loc='upper right')
    axes[1].set_ylabel('Trade USDT / 200ms'); axes[4].set_ylabel('Spread (bps)')
    for ax in axes:
        ax.axvline(utc(effective_timestamp),color='#777777',ls=':',lw=.8)
        ax.set_xlim(utc(start),utc(end))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S',tz=__import__('datetime').timezone.utc))
    axes[-1].set_xlabel('UTC; Bybit effective = raw + 200ms; buy positive / sell negative liquidation orders')
    fig.suptitle(f'{sym.upper()} · {utc(effective_timestamp)}\n{title}',y=.998)
    fig.tight_layout(rect=[0,0,1,.96])
    if save_name:
        (OUTPUT/'figures').mkdir(exist_ok=True)
        fig.savefig(OUTPUT/'figures'/f'{save_name}.png',dpi=130,bbox_inches='tight')
    display(fig); plt.close(fig)
    if totals:
        display(pd.DataFrame(totals))
    for exchange in ['binance','bybit']:
        liq=frames.get((exchange,'liquidations'),pd.DataFrame())
        if not liq.empty:
            display(Markdown(f'{exchange}: liquidation observations; raw и effective оси указаны явно.'))
            display(liq[['raw_timestamp','effective_timestamp','datetime_utc','ticker','side','price','amount']].head(20))
    return frames

def render_event_diagnostics(result, plots_per_type=2):
    heading(f'Cross-source evidence: {result["id"]}',2)
    events = event_evidence.filter(pl.col('source_id')==result['id']).collect(engine='streaming').to_pandas()
    for issue,f in events.groupby('issue_type'):
        heading(issue,3)
        order = ('abs_return_bps' if issue=='EXTREME_PRICE_MOVE' else 'spread_bps' if issue=='EXTREME_SPREAD'
                 else 'episode_notional' if issue=='LIQUIDATION_CLUSTER' else 'notional')
        f=f.sort_values(order,ascending=False)
        cols=['source_row','ticker','timestamp','raw_timestamp','effective_timestamp',order,'severity',
              'bbo_move_bps','trades_move_bps','trades_amount','trades_notional',
              'binance_liq_count','binance_liq_buy_notional','binance_liq_sell_notional',
              'binance_liq_move_bps','bybit_liq_count','bybit_liq_buy_notional','bybit_liq_sell_notional',
              'bybit_liq_move_bps','bbo_max_spread_bps','price_distance_from_bbo_bps','reason']
        display(f[list(dict.fromkeys(cols))].head(TOP_N))
        display(f.groupby('severity').size().rename('observation_issue_count').to_frame())
        for j,(_,event) in enumerate(f.drop_duplicates('center_second').head(plots_per_type).iterrows()):
            diagnostic_window(event.symbol,event.diagnostic_center_timestamp,
                 title=f'{issue} · {event.severity}',save_name=f'{result["id"]}_{issue}_{j}')
            if issue=='EXTREME_PRICE_MOVE':
                # Непосредственный следующий tick: отдельный тест мгновенного возврата.
                ctx=raw_rows(result['path'],[event.source_row],2)
                p=(ctx.bid_price+ctx.ask_price)/2 if result['kind']=='bbo' else ctx.price
                ctx['diagnostic_price']=p
                display(ctx)
                current=np.flatnonzero(ctx.source_row.to_numpy()==event.source_row)
                if len(current) and current[0]>0 and current[0]+1<len(ctx):
                    k=current[0]; old,now,nxt=p.iloc[k-1:k+2]
                    ratio=abs(nxt-old)/abs(now-old) if now!=old else np.nan
                    display(Markdown(f'Доля исходного скачка, оставшаяся на следующем observation: **{ratio:.3f}**. '
                        'Значение около 0 означает возврат; само по себе не доказывает bad tick. '
                        'Порядок внутри равного timestamp не является доказанным causal order.'))

def dataset_conclusion(result):
    row=summary.loc[summary.file==result['path']].iloc[0]
    gaps=gap_evidence.loc[gap_evidence.source_id==result['id']] if not gap_evidence.empty else pd.DataFrame()
    nfeed=int((gaps.finding=='BBO DATA / FEED ISSUE').sum()) if len(gaps) else 0
    nclear=int(row['number of CLEAR_DATA_ERROR observations'])
    nmarket=int(row['number of PLAUSIBLE_MARKET_EVENT observations'])
    nexact=int(row['exact duplicate rows'])
    events = event_evidence.filter(pl.col('source_id')==result['id'])
    unsupported = events.filter(~pl.col('price_level_supported')).select(pl.col('source_row').n_unique()).collect().item()
    text=(f'**{result["id"]}.** После исключения {int(row["number of removed observations"]):,} строк '
          f'с явными нарушениями данные '+('пригодны только с локальной маской проблемных интервалов' if nfeed else
          'в целом пригодны для исследовательского анализа при учёте перечисленных ограничений')+'. '
          f'CLEAR DATA ERROR: {nclear:,}; timestamp reversals: {int(row["timestamp reversals"]):,}; '
          f'лишних копий полностью совпадающих строк: {nexact:,} (сохранены, event ID отсутствует). '
          f'Обнаружено интервалов с сильными признаками BBO feed issue: {nfeed:,}. '
          f'Наблюдений с поддержкой PLAUSIBLE MARKET EVENT: {nmarket:,}; они сохранены. ')
    text+=f'Уровень цены не подтверждён локальным BBO у {unsupported:,} отмеченных строк: это требует отдельного исследования перед построением price-based признаков. '
    if nclear:
        reasons=pl.scan_parquet(OUTPUT/result['id']/'removed_observations.parquet').select(
            pl.col('removal_reason').unique()).collect().to_series().to_list()
        text+='Причины удаления: '+', '.join(reasons)+'. '
    else:
        text+='Удалений не потребовалось. '
    text+='Все statistical tails, crossed/zero-size BBO и неоднозначные совпадения остаются доступными; SUSPICIOUS не означает доказанную ошибку. '
    if result['kind']=='bbo':
        text+='Для markout не следует безусловно forward-fill mid через отмеченные gaps или missing-side участки; проверяйте возраст последней котировки.'
    if result['exchange']=='bybit':
        text+='Для causal признаков используйте только effective_timestamp = raw_timestamp + 200 ms. Отсутствие liquidation не доказывает отсутствие торговли на Bybit.'
    if result['kind']=='liquidations':
        text+=' Большие inter-arrival gaps здесь ожидаемы для редких условных событий и сами по себе не считаются ошибкой.'
    display(Markdown(text))
    if len(gaps):
        display(gaps[['start_utc','end_utc','gap_seconds','finding','trades_count','trades_notional']])
    if result['kind']=='trades' and not trade_gap_evidence.empty:
        display(Markdown('Наибольшие паузы исполнения также требуют осторожности; отсутствие наблюдений не доказывает физическое отсутствие торговли.'))
        display(trade_gap_evidence.loc[trade_gap_evidence.source_id.eq(result['id']),
            ['start_utc','end_utc','gap_us','bbo_updates_inside_gap','reason']])
    caution = (events.filter(pl.col('severity')=='SUSPICIOUS').sort('abs_return_bps',descending=True,nulls_last=True)
        .select(['source_row','issue_type','window_start_utc','window_end_utc_exclusive',
                 'price_distance_from_bbo_bps','reason']).head(10).collect(engine='streaming').to_pandas())
    if len(caution):
        display(Markdown('Приоритетные интервалы для дополнительной осторожности (полный список в `event_evidence`):'))
        display(caution)

def validate_audit():
    for r in results:
        assert sum(v['rows'] for v in r['tickers'].values()) == r['rows']
        actual = summary.loc[summary.file==r['path'],'number of CLEAR_DATA_ERROR observations'].iloc[0]
        assert actual==r['removed_rows'], (r['id'],'removals do not reconcile')
        removed_path = OUTPUT/r['id']/'removed_observations.parquet'
        assert (pq.ParquetFile(removed_path).metadata.num_rows if removed_path.exists() else 0)==r['removed_rows']
        for ticker,ds in r['distributions'].items():
            for metric,d in ds.items():
                if d['count']:
                    assert sum(d['hist_counts'])==d['count'],(r['id'],metric,'histogram count')
                    qs=[d[k] for k in ['minimum','p05','p25','median','p75','p95','p99','p99.9','maximum']]
                    assert all(a<=b for a,b in zip(qs,qs[1:])),(r['id'],metric,'quantile order')
    # Независимое восстановление нескольких наиболее важных окон непосредственно из raw parquet.
    reconciled=[]
    for sym in sorted({symbol(t) for r in results for t in r['tickers']}):
        for severity in ['SUSPICIOUS','PLAUSIBLE_MARKET_EVENT']:
            chosen=(event_evidence.filter((pl.col('symbol')==sym) & (pl.col('severity')==severity) &
                    (pl.col('issue_type')=='EXTREME_PRICE_MOVE')).top_k(1,by='abs_return_bps')
                    .collect(engine='streaming').to_pandas())
            for _,e in chosen.iterrows():
                for exchange,kind,prefix in [('binance','trades','trades'),
                        ('binance','liquidations','binance_liq'),('bybit','liquidations','bybit_liq')]:
                    source=next((r for r in results if r['exchange']==exchange and r['kind']==kind and
                                 any(symbol(t)==sym for t in r['tickers'])),None)
                    if source is None:
                        continue
                    raw=window_rows(source,sym,(e.center_second-5)*1_000_000,(e.center_second+6)*1_000_000)
                    _,remove,_,_,_=validation_masks(raw[list(source['schema'])],kind)
                    clean=raw.loc[~remove]
                    value=(clean.price*clean.amount).sum()
                    assert len(clean)==e[prefix+'_count'],(sym,prefix,'window count')
                    assert np.isclose(value,e[prefix+'_notional'],rtol=1e-7,atol=.05),(sym,prefix,'window notional')
                    reconciled.append({'symbol':sym,'severity':severity,'source':prefix,
                        'center_second':e.center_second,'count':len(clean),
                        'raw_notional':value,'aggregate_notional':e[prefix+'_notional'],
                        'absolute_difference':abs(value-e[prefix+'_notional'])})
    pd.DataFrame(reconciled).to_csv(OUTPUT/'window_reconciliation.csv',index=False)
    # Сохраняем результат контроля неизменности byte-for-byte.
    checked=[]
    for row in source_manifest.itertuples():
        after=sha256(row.source_file)
        assert after==row.sha256, f'Source changed: {row.source_file}'
        checked.append({'source_file':row.source_file,'sha256_before':row.sha256,'sha256_after':after,'unchanged':True})
    verification=pd.DataFrame(checked)
    verification.to_csv(OUTPUT/'source_integrity_verification.csv',index=False)
    return verification


def run():
    global FILES, source_manifest, results, results_by_id, event_evidence_paths
    global gap_evidence, trade_gap_evidence, audit_log, removed_observations, event_evidence, summary
    FILES = sorted(DATA.rglob('*.parquet'))
    if not FILES:
        raise FileNotFoundError(f'No raw Parquet files in {DATA}; see data/README.md')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)


    source_manifest = pd.DataFrame([{'source_file':str(p), 'bytes':p.stat().st_size,
        'rows':pq.ParquetFile(p).metadata.num_rows, 'sha256':sha256(p)} for p in FILES])
    source_manifest.to_csv(OUTPUT / 'source_manifest.csv', index=False)
    results = [audit_file(Path(row.source_file), row.sha256) for row in source_manifest.itertuples()]
    results_by_id = {r['id']:r for r in results}
    print(f'Проверено файлов: {len(results)}; строк: {sum(r["rows"] for r in results):,}')

    event_evidence_paths = build_event_evidence(results)
    gap_evidence = investigate_bbo_gaps(results)
    trade_gap_evidence = investigate_trade_gaps(results, gap_evidence)

    AUDIT_COLUMNS = ['source_file','source_row','ticker','timestamp','datetime_utc',
                     'issue_type','severity','reason','action']
    audit_paths = sorted(p for r in results for p in (OUTPUT/r['id']).glob('*_audit.parquet'))
    audit_log = pl.scan_parquet(audit_paths)  # Полный lazy DataFrame, НЕ sample: может содержать сотни миллионов строк.
    removed_paths = sorted(p for r in results for p in (OUTPUT/r['id']).glob('removed_observations.parquet'))
    removed_observations = (pl.concat([pl.scan_parquet(p) for p in removed_paths],how='diagonal_relaxed')
                            if removed_paths else pl.DataFrame(schema={'source_file':pl.String,
                              'source_row':pl.Int64,'removal_reason':pl.String}).lazy())
    event_evidence = pl.scan_parquet(event_evidence_paths)

    summary = summary_generation(results)
    summary.to_csv(OUTPUT/'summary.csv',index=False)

    verification = validate_audit()
    return summary, verification

if __name__ == "__main__":
    run()
