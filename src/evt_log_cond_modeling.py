"""Causal DC features, validation-only selection, and a sealed test evaluation.

No training code reads the test feature partition. The final stage freezes all
five horizons and both models before it creates or reads test observations.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_squared_error, log_loss, precision_score,
                             recall_score, f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import liquidation_bursts as lb
import liquidation_early_warning as ew

from research_paths import ROOT, resolve_legacy
OUT = ROOT / 'artifacts/results/evt' / 'modeling'
START, END, READY = ew.START, ew.END, ew.READY
VAL_START, TEST_START = ew.VAL_START, ew.TEST_START
WINDOWS = (5, 10, 20)
HORIZONS = (5, 20, 30, 120, 300)
LAG_COUNTS = tuple(range(5, 11))
C_GRID = (.01, .1, 1., 10., 100.)
TREES = tuple(range(6, 16))
LEAVES = (15, 20, 30)
LEARNING_RATES = (.01, .05, .1)
SEED = 20261012
PURGE_SECONDS = max(HORIZONS) + 5


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def verify_inputs():
    OUT.mkdir(exist_ok=True, parents=True)
    catalog = pd.read_csv(ROOT / 'artifacts/results/evt/source_catalog.csv')
    for row in catalog.itertuples():
        assert digest(resolve_legacy(row.source)) == row.sha256, row.source
    old = json.loads((lb.OUT / 'run_manifest.json').read_text())
    required = [f'seconds_{s}.parquet' for s in lb.SYMBOLS]
    required += [f'features_{s}_{x}_{side}.parquet' for s in lb.SYMBOLS
                 for x in lb.EXCHANGES for side in lb.SIDES]
    required += ['frozen_config.json']
    verified = {}
    for name in required:
        actual = digest(lb.OUT / name)
        assert actual == old['artifacts'][name], name
        verified[str((lb.OUT / name).relative_to(ROOT))] = actual
    bounds = json.loads((ROOT / 'artifacts/results/evt/split_specification.json').read_text())
    assert bounds['split_bounds_us']['train'] == [START*10**6, VAL_START*10**6]
    assert bounds['split_bounds_us']['validation'] == [VAL_START*10**6, TEST_START*10**6]
    assert bounds['split_bounds_us']['test'] == [TEST_START*10**6, END*10**6]
    dump(OUT / 'verified_inputs.json', {'source_hashes':dict(zip(catalog.source, catalog.sha256)),
                                       'derived_hashes':verified})


def asof_quotes(batches, query_us):
    """Latest quote strictly before each sorted query, including batch boundaries."""
    query_us = np.asarray(query_us, dtype=np.int64)
    if np.any(np.diff(query_us) < 0):
        raise ValueError('Queries must be sorted')
    names = ['bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    answer = {name:np.full(len(query_us), np.nan) for name in names}
    answer['quote_timestamp'] = np.full(len(query_us), -1, dtype=np.int64)
    pos = 0
    last_ts = -1
    last = dict.fromkeys(names, np.nan)
    for batch in batches:
        if isinstance(batch, dict):
            values = batch
        else:
            values = {name:batch.column(batch.schema.get_field_index(name)).to_numpy()
                      for name in ['timestamp', *names]}
        ts = np.asarray(values['timestamp'], dtype=np.int64)
        if not len(ts):
            continue
        if ts[0] < last_ts or np.any(np.diff(ts) < 0):
            raise ValueError('BBO timestamps are not sorted')
        stop = np.searchsorted(query_us, ts[-1], side='right')
        j = np.searchsorted(ts, query_us[pos:stop], side='left') - 1
        good = j >= 0
        safe = np.maximum(j, 0)
        answer['quote_timestamp'][pos:stop] = np.where(good, ts[safe], last_ts)
        for name in names:
            answer[name][pos:stop] = np.where(good, values[name][safe], last[name])
            last[name] = values[name][-1]
        last_ts = int(ts[-1])
        pos = stop
        if pos == len(query_us):
            break
    answer['quote_timestamp'][pos:] = last_ts
    for name in names:
        answer[name][pos:] = last[name]
    return pd.DataFrame(answer)


def raw_quote_snapshots(symbol, query_us):
    path = lb.source_path(symbol, 'binance_booktickers')
    # The scanner filters on the timestamp before materializing query data.
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, format='parquet')
    scanner = dataset.scanner(columns=['timestamp','bid_price','ask_price','bid_amount','ask_amount'],
        filter=ds.field('timestamp') < int(query_us[-1]), batch_size=500_000, use_threads=False)
    return asof_quotes(scanner.to_batches(), query_us)


def queue_features(bid, ask):
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask >= 0)
    difference = np.where(valid, bid-ask, np.nan)
    total = bid + ask
    imbalance = np.divide(difference, total, out=np.full(len(bid), np.nan),
                          where=valid & (total > 0))
    return difference, imbalance


def intensity(event_us, query_us, window):
    """Number of messages per second, exactly [t-window, t)."""
    event_us = np.sort(np.asarray(event_us, dtype=np.int64))
    query_us = np.asarray(query_us, dtype=np.int64)
    upper = np.searchsorted(event_us, query_us, side='left')
    lower = np.searchsorted(event_us, query_us-window*1_000_000, side='left')
    return (upper-lower) / float(window)


def lagged_returns(mid, age, q, n_lags=10):
    """Lag j is log(m(t-5*j)/m(t-5*(j+1))); quotes are strictly past t."""
    result = {}
    for j in range(n_lags):
        a, b = q-1-5*j, q-1-5*(j+1)
        valid = (b >= 0) & (a < len(mid))
        aa, bb = np.clip(a, 0, len(mid)-1), np.clip(b, 0, len(mid)-1)
        valid &= (age[aa] <= 2) & (age[bb] <= 2) & (mid[aa] > 0) & (mid[bb] > 0)
        result[f'log_return_5s_lag{j}_bps'] = np.where(valid, np.log(mid[aa]/mid[bb])*10000, np.nan)
    return result


def feature_names(n_lags):
    if n_lags not in LAG_COUNTS:
        raise ValueError('Lag count must be 5..10')
    return (['symbol_eth', 'queue_difference_base', 'queue_imbalance']
        + [f'{ex}_{side}_intensity_{w}s' for ex in lb.EXCHANGES for side in lb.SIDES for w in WINDOWS]
        + [f'rv_{w}s_bps' for w in WINDOWS]
        + [f'log_return_5s_lag{j}_bps' for j in range(n_lags)])


def period_labels(t, burst):
    starts = np.flatnonzero(burst & ~np.r_[False, burst[:-1]])
    onsets = START + starts*5
    result, _ = ew.future_labels(t, onsets, horizons=HORIZONS, ends=ew.partition_end(t)-5)
    return result


def build_features(period='development'):
    if period not in ('development','test'):
        raise ValueError(period)
    if period == 'test' and not (OUT/'frozen_selection.json').exists():
        raise RuntimeError('Test remains sealed until all selections are frozen')
    end = TEST_START if period == 'development' else END
    frames, attrition = [], []
    for symbol in lb.SYMBOLS:
        print(f'Features {period}: {symbol}', flush=True)
        sec = pl.scan_parquet(lb.OUT/f'seconds_{symbol}.parquet').filter(pl.col('second') < end).collect()
        mid = sec['mid_close'].to_numpy()
        age = sec['quote_age_upper_s'].to_numpy()
        fresh = np.isfinite(mid) & (mid > 0) & (age <= 2)
        _, _, event, progress, overshoot = ew.directional_changes(np.log(mid), fresh,
                                                np.array(ew.THETA_BPS)/10000)
        del progress, overshoot
        q = np.flatnonzero(event) + 1
        t = START + q
        use = (t >= READY) & (t < end)
        if period == 'test':use &= t >= TEST_START
        q, t = q[use], t[use]
        n_dc = len(t)
        f = pd.DataFrame({'timestamp':t*1_000_000, 'symbol':symbol,
                          'symbol_eth':int(symbol == 'ethusdt'), 'split':ew.partition(t)})
        for window in WINDOWS:
            _, rv = ew.trailing_price_features(mid, fresh, q, window)
            f[f'rv_{window}s_bps'] = rv
        for name, values in lagged_returns(mid, age, q).items():f[name] = values
        quotes = raw_quote_snapshots(symbol, f.timestamp.to_numpy())
        f['quote_timestamp'] = quotes.quote_timestamp.to_numpy()
        f['quote_age_s'] = (f.timestamp - f.quote_timestamp)/1_000_000
        f['queue_bid_base'] = quotes.bid_amount.to_numpy()
        f['queue_ask_base'] = quotes.ask_amount.to_numpy()
        f['queue_difference_base'], f['queue_imbalance'] = queue_features(f.queue_bid_base, f.queue_ask_base)
        # Raw audited message semantics, matching Data Loading (no new deduplication).
        for ex in lb.EXCHANGES:
            delay = 200_000 if ex == 'bybit' else 0
            liq = pl.scan_parquet(lb.source_path(symbol, f'{ex}_liquidations')).filter(
                pl.col('timestamp')+delay < end*1_000_000).select('timestamp','side').collect().to_pandas()
            for side in lb.SIDES:
                times = liq.loc[liq.side.eq(side),'timestamp'].to_numpy()+delay
                for window in WINDOWS:
                    f[f'{ex}_{side}_intensity_{window}s'] = intensity(times, f.timestamp.to_numpy(), window)
        # Only previously frozen labels, not newly tuned detector thresholds.
        burst = np.zeros(len(sec)//5, dtype=bool)
        for ex in lb.EXCHANGES:
            for side in lb.SIDES:
                a = pl.scan_parquet(lb.OUT/f'features_{symbol}_{ex}_{side}.parquet').filter(
                    pl.col('timestamp') <= end*1_000_000).select('burst').collect()['burst'].to_numpy()
                assert len(a) == len(burst)
                burst |= a
        current_bin = q//5-1
        f['burst_known_now'] = burst[current_bin]
        for name, value in period_labels(t, burst).items():f[name] = value
        finite = np.isfinite(f[feature_names(10)].to_numpy()).all(axis=1)
        fresh_quote = (f.quote_timestamp > 0) & f.quote_age_s.between(0, 2, inclusive='right')
        not_active = ~f.burst_known_now
        complete = t + PURGE_SECONDS < ew.partition_end(t)
        f['usable'] = finite & fresh_quote & not_active & complete
        attrition.append({'period':period,'symbol':symbol,'dc_after_calibration':n_dc,
            'nonfinite_feature_rows':int((~finite).sum()), 'stale_quote_rows':int((~fresh_quote).sum()),
            'active_burst_rows':int((~not_active).sum()),'boundary_purge_rows':int((~complete).sum()),
            'usable_rows':int(f.usable.sum())})
        floats = f.select_dtypes('float64').columns
        f[floats] = f[floats].astype('float32')
        frames.append(f)
        del sec, event, mid, age, burst, quotes
        gc.collect()
    all_rows = pd.concat(frames, ignore_index=True).sort_values(['timestamp','symbol']).reset_index(drop=True)
    all_rows.to_parquet(OUT/f'features_{period}.parquet', index=False)
    pd.DataFrame(attrition).to_csv(OUT/f'attrition_{period}.csv', index=False)
    print(period, 'usable', int(all_rows.usable.sum()), '/', len(all_rows), flush=True)
    return all_rows


def parameters():
    return {'windows_s':WINDOWS,'horizons_s':HORIZONS,'n_return_lags':LAG_COUNTS,
            'logistic_C':C_GRID,'n_trees':TREES,'num_leaves':LEAVES,'learning_rate':LEARNING_RATES,
            'split_days':[60,15,15],'selection_metric':'validation MSE of predict_proba[:,1]',
            'objective':'binary log loss','class_weight':None,'downsampling':False,
            'purge_seconds_all_horizons':PURGE_SECONDS,'sampling':'directional change',
            'dc_threshold_bps':ew.THETA_BPS,'test_previously_used_in_other_notebooks':True,
            'seed':SEED}


def evaluate(y, probability, threshold):
    alarm = probability >= threshold
    return {'n':len(y), 'positives':int(np.sum(y)), 'prevalence':float(np.mean(y)),
            'mse':float(mean_squared_error(y, probability)),
            'log_loss':float(log_loss(y, probability, labels=[0,1])),
            'average_precision':float(average_precision_score(y, probability)),
            'threshold':float(threshold), 'precision':float(precision_score(y,alarm,zero_division=0)),
            'recall':float(recall_score(y,alarm,zero_division=0)),
            'f1':float(f1_score(y,alarm,zero_division=0)),
            'f1_macro':float(f1_score(y,alarm,average='macro',zero_division=0))}


def mse_eval(y, p):
    return 'probability_mse', float(np.mean((y-p)**2)), False


def tree_model(leaves, learning_rate, n_trees=15):
    return lgb.LGBMClassifier(objective='binary', n_estimators=n_trees,
        num_leaves=leaves, learning_rate=learning_rate, min_child_samples=100,
        max_bin=127, reg_lambda=1., random_state=SEED, n_jobs=4,
        deterministic=True, force_col_wise=True, verbosity=-1)


def tune():
    if (OUT/'test_evaluation_receipt.json').exists():
        raise RuntimeError('Test already evaluated: do not retune this experiment against that result')
    dump(OUT/'search_specification.json', parameters())
    data = pd.read_parquet(OUT/'features_development.parquet')
    assert not data.split.eq('test').any()
    data = data[data.usable].reset_index(drop=True)
    train = data.split.eq('train').to_numpy()
    val = data.split.eq('validation').to_numpy()
    names = feature_names(10)
    xtrain = data.loc[train, names].astype('float64')
    xval = data.loc[val, names].astype('float64')
    scaler = StandardScaler().fit(xtrain)
    xs_train = scaler.transform(xtrain)
    xs_val = scaler.transform(xval)
    joblib.dump(scaler, OUT/'train_only_scaler.joblib')
    trials, choices, val_predictions, metric_rows = [], [], [], []
    for h in HORIZONS:
        print(f'Tuning horizon {h}s, train={train.sum()}, validation={val.sum()}', flush=True)
        ytrain = data.loc[train,f'target_burst_{h}s'].to_numpy(dtype=int)
        yval = data.loc[val,f'target_burst_{h}s'].to_numpy(dtype=int)
        assert set(np.unique(ytrain)) == {0,1} and set(np.unique(yval)) == {0,1}
        prior = float(ytrain.mean())
        best_lr, best_gb = None, None
        for n_lags in LAG_COUNTS:
            count = len(feature_names(n_lags))
            tr, va = xs_train[:,:count], xs_val[:,:count]
            for c in C_GRID:
                model = LogisticRegression(C=c, solver='lbfgs', max_iter=2000,
                                           tol=1e-6, random_state=SEED)
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(tr, ytrain)
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                p = model.predict_proba(va)[:,1]
                mse = float(mean_squared_error(yval, p))
                row = {'model':'logistic','horizon_s':h,'n_lags':n_lags,'C':c,
                       'validation_mse':mse,'converged':converged,'iterations':int(model.n_iter_[0])}
                trials.append(row)
                if converged and (best_lr is None or mse < best_lr['validation_mse']):
                    best_lr = row.copy()
            cols = feature_names(n_lags)
            for leaves in LEAVES:
                for rate in LEARNING_RATES:
                    model = tree_model(leaves, rate)
                    model.fit(xtrain[cols], ytrain, eval_set=[(xval[cols], yval)], eval_metric=mse_eval)
                    scores = model.evals_result_['valid_0']['probability_mse']
                    for trees in TREES:
                        row = {'model':'lightgbm','horizon_s':h,'n_lags':n_lags,'num_leaves':leaves,
                               'learning_rate':rate,'n_estimators':trees,
                               'validation_mse':float(scores[trees-1]),'converged':True}
                        trials.append(row)
                        if best_gb is None or row['validation_mse'] < best_gb['validation_mse']:
                            best_gb = row.copy()
            pd.DataFrame(trials).to_csv(OUT/'validation_search.csv', index=False)
            print(f'  {h}s: tested {n_lags} return lags', flush=True)
        if best_lr is None:raise RuntimeError('No converged logistic candidate')
        for chosen in (best_lr, best_gb):
            cols = feature_names(chosen['n_lags'])
            if chosen['model'] == 'logistic':
                model = LogisticRegression(C=chosen['C'], solver='lbfgs', max_iter=2000,
                                           tol=1e-6, random_state=SEED)
                with threadpool_limits(limits=4):model.fit(xs_train[:,:len(cols)], ytrain)
                p = model.predict_proba(xs_val[:,:len(cols)])[:,1]
            else:
                model = tree_model(chosen['num_leaves'], chosen['learning_rate'], chosen['n_estimators'])
                model.fit(xtrain[cols], ytrain)
                p = model.predict_proba(xval[cols])[:,1]
            assert np.isclose(mean_squared_error(yval,p), chosen['validation_mse'], rtol=1e-8, atol=1e-10)
            chosen.update({'features':cols, 'threshold':ew.choose_threshold(yval,p),
                           'train_prevalence':prior})
            filename = f"model_{chosen['model']}_{h}s.joblib"
            joblib.dump(model, OUT/filename)
            chosen['model_file'] = filename
            chosen['model_sha256'] = digest(OUT/filename)
            choices.append(chosen)
            pred = data.loc[val,['timestamp','symbol']].copy()
            pred['horizon_s'], pred['model'], pred['target'], pred['probability'] = h, chosen['model'], yval, p
            pred['threshold'] = chosen['threshold']
            val_predictions.append(pred)
            metric_rows.append({'split':'validation','model':chosen['model'],'horizon_s':h,
                                **evaluate(yval,p,chosen['threshold'])})
        metric_rows.append({'split':'validation','model':'constant_train_prevalence','horizon_s':h,
                            **evaluate(yval,np.full(len(yval),prior),.5)})
        pd.DataFrame(choices).drop(columns='features').to_csv(OUT/'selected_parameters.csv',index=False)
        pd.DataFrame(metric_rows).to_csv(OUT/'validation_metrics.csv',index=False)
        print(f'  selected {h}s: LR mse={best_lr["validation_mse"]:.6f}; GB mse={best_gb["validation_mse"]:.6f}',flush=True)
    pd.concat(val_predictions,ignore_index=True).to_parquet(OUT/'predictions_validation.parquet',index=False)
    population = data.groupby('split').agg(rows=('timestamp','size'),first=('timestamp','min'),last=('timestamp','max'))
    population.to_csv(OUT/'development_population.csv')
    dump(OUT/'frozen_selection.json', {
        'frozen_utc':pd.Timestamp.now(tz='UTC').isoformat(), 'models':choices,
        'specification_sha256':digest(OUT/'search_specification.json'),
        'code_sha256':digest(Path(__file__)),
        'development_sha256':digest(OUT/'features_development.parquet'),
        'scaler_sha256':digest(OUT/'train_only_scaler.joblib'),
        'test_evaluations_before_freeze':0})


def evaluate_test_once():
    receipt = OUT/'test_evaluation_receipt.json'
    if receipt.exists():
        print('Test already evaluated; returning saved metrics without fitting/predicting again.',flush=True)
        return pd.read_csv(OUT/'test_metrics.csv')
    frozen_path = OUT/'frozen_selection.json'
    frozen = json.loads(frozen_path.read_text())
    assert frozen['code_sha256'] == digest(Path(__file__))
    assert frozen['scaler_sha256'] == digest(OUT/'train_only_scaler.joblib')
    assert len(frozen['models']) == len(HORIZONS)*2
    for chosen in frozen['models']:
        assert digest(OUT/chosen['model_file']) == chosen['model_sha256']
    # Development selection is now over. Test data are first built/read here.
    data = build_features('test')
    data = data[data.usable].reset_index(drop=True)
    assert data.split.eq('test').all()
    scaler = joblib.load(OUT/'train_only_scaler.joblib')
    full = data[feature_names(10)].astype('float64')
    scaled = scaler.transform(full)
    predictions, rows = [], []
    for chosen in frozen['models']:
        h, name = chosen['horizon_s'], chosen['model']
        y = data[f'target_burst_{h}s'].to_numpy(dtype=int)
        model = joblib.load(OUT/chosen['model_file'])
        x = scaled[:,:len(chosen['features'])] if name == 'logistic' else full[chosen['features']]
        p = model.predict_proba(x)[:,1]
        rows.append({'split':'test','model':name,'horizon_s':h,**evaluate(y,p,chosen['threshold'])})
        out = data[['timestamp','symbol']].copy()
        out['horizon_s'], out['model'], out['target'], out['probability'] = h, name, y, p
        out['threshold'] = chosen['threshold']
        predictions.append(out)
        if name == 'logistic':
            rows.append({'split':'test','model':'constant_train_prevalence','horizon_s':h,
                         **evaluate(y,np.full(len(y),chosen['train_prevalence']),.5)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT/'test_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictions_test.parquet',index=False)
    dump(receipt, {'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),
         'frozen_selection_sha256':digest(frozen_path),'test_evaluation_passes':1,
         'models':len(frozen['models']),'rows_per_model':len(data),
         'predictions_sha256':digest(OUT/'predictions_test.parquet'),
         'metrics_sha256':digest(OUT/'test_metrics.csv'),
         'historical_test_previously_seen_in_other_notebooks':True})
    return metrics


def verify_results():
    frozen = json.loads((OUT/'frozen_selection.json').read_text())
    receipt = json.loads((OUT/'test_evaluation_receipt.json').read_text())
    assert receipt['test_evaluation_passes'] == 1
    assert receipt['frozen_selection_sha256'] == digest(OUT/'frozen_selection.json')
    assert frozen['code_sha256'] == digest(Path(__file__))
    assert frozen['specification_sha256'] == digest(OUT/'search_specification.json')
    assert frozen['scaler_sha256'] == digest(OUT/'train_only_scaler.joblib')
    assert frozen['development_sha256'] == digest(OUT/'features_development.parquet')
    assert receipt['predictions_sha256'] == digest(OUT/'predictions_test.parquet')
    assert receipt['metrics_sha256'] == digest(OUT/'test_metrics.csv')
    for chosen in frozen['models']:
        assert chosen['model_sha256'] == digest(OUT/chosen['model_file'])
    return receipt


def summarize_probabilities():
    """Diagnostics of frozen predictions; no fitting or selection."""
    rows, daily, uncertainty = [], [], []
    for split in ('validation','test'):
        p = pd.read_parquet(OUT/f'predictions_{split}.parquet')
        p['day'] = pd.to_datetime(p.timestamp,unit='us',utc=True).dt.strftime('%Y-%m-%d')
        p['squared_error'] = (p.probability-p.target)**2
        d = p.groupby(['horizon_s','model','day']).squared_error.agg(['sum','size']).reset_index()
        d['split'] = split
        daily.append(d)
        for (h, model), f in p.groupby(['horizon_s','model']):
            f = f.copy()
            # Probability quantiles, not arbitrary equal row blocks that split ties.
            bins = np.unique(np.quantile(f.probability,np.linspace(0,1,11)))
            f['bin'] = np.searchsorted(bins[1:-1],f.probability,side='right')
            c = f.groupby('bin').agg(n=('target','size'),predicted=('probability','mean'),observed=('target','mean')).reset_index()
            c['horizon_s'],c['model'],c['split'] = h,model,split
            rows.append(c)
        for h in HORIZONS:
            a = d[d.horizon_s.eq(h)].pivot(index='day',columns='model',values='sum')
            sizes = d[d.horizon_s.eq(h)&d.model.eq('logistic')].set_index('day')['size'].reindex(a.index).to_numpy()
            delta = (a.lightgbm-a.logistic).to_numpy()
            rng = np.random.default_rng(SEED+h)
            draws = rng.integers(0,len(a),size=(1000,len(a)))
            values = delta[draws].sum(axis=1)/sizes[draws].sum(axis=1)
            uncertainty.append({'split':split,'horizon_s':h,'days':len(a),
                'mse_lightgbm_minus_logistic':float(delta.sum()/sizes.sum()),
                'ci_low':float(np.quantile(values,.025)),'ci_high':float(np.quantile(values,.975))})
    pd.concat(rows,ignore_index=True).to_csv(OUT/'probability_calibration.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(OUT/'daily_squared_errors.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(OUT/'paired_day_mse_intervals.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['features','tune','test','summary','all'],default='all')
    args = parser.parse_args()
    if args.stage in ('features','all'):
        verify_inputs();build_features('development')
    if args.stage in ('tune','all'):tune()
    if args.stage in ('test','all'):evaluate_test_once()
    if args.stage in ('summary','all'):summarize_probabilities()
