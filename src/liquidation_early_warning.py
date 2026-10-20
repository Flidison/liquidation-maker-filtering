"""Reproducible event-time liquidation warning experiment; no execution/PnL claim.

All feature boundaries use records strictly before timestamp (microseconds UTC).
Dec 1--7 fits transition probabilities; Dec 8--14 calibrates surprise scores.
These nuisance estimators then remain frozen. No outcome-dependent sampling.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import warnings

import joblib
import lightgbm as lgb
import numba
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy.special import ndtr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             brier_score_loss, precision_recall_curve)

import liquidation_bursts as lb

from research_paths import ROOT, resolve_legacy
OUT = ROOT / 'artifacts/results/early_warning'
START, END = lb.START, lb.END
FIT_END = START + 7 * 86400
READY = START + 14 * 86400
VAL_START = START + 60 * 86400
TEST_START = START + 75 * 86400
LAGS = (5, 10, 20, 30, 60)
HORIZONS = (5, 20, 30, 120, 300)
THETA_BPS = (2.5, 5., 10.)
K = 20
SEED = 20261011


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def partition(seconds):
    return np.where(seconds < VAL_START, 'train',
                    np.where(seconds < TEST_START, 'validation', 'test'))


def partition_end(seconds):
    return np.where(seconds < VAL_START, VAL_START,
                    np.where(seconds < TEST_START, TEST_START, END))


@numba.njit(cache=True)
def directional_changes(log_price, valid, thresholds):
    """Causal joint state at each completed second; reset on stale/missing BBO.

Each threshold tracks a running extreme. A reversal >= theta confirms a DC.
Multiple thresholds can flip in one observation: one atomic joint-state event.
"""
    n, m = len(log_price), len(thresholds)
    states = np.full(n, -1, np.int16)
    segments = np.zeros(n, np.int32)
    events = np.zeros(n, np.bool_)
    progress = np.full((n, m), np.nan, np.float32)
    overshoot = np.full((n, m), np.nan, np.float32)
    mode = np.zeros(m, np.int8)
    high = np.zeros(m); low = np.zeros(m); extreme = np.zeros(m)
    confirm = np.zeros(m)
    initialized = False
    segment = 0
    previous = -1
    for i in range(n):
        segments[i] = segment
        if not valid[i]:
            if initialized:
                segment += 1
            initialized = False
            previous = -1
            mode[:] = 0
            continue
        p = log_price[i]
        if not initialized:
            high[:] = p; low[:] = p; extreme[:] = p; confirm[:] = p
            initialized = True
        segments[i] = segment
        ready = True
        state = 0
        for j in range(m):
            theta = thresholds[j]
            if mode[j] == 0:
                high[j] = max(high[j], p); low[j] = min(low[j], p)
                if p - low[j] >= theta:
                    mode[j] = 1; extreme[j] = p; confirm[j] = p
                elif high[j] - p >= theta:
                    mode[j] = -1; extreme[j] = p; confirm[j] = p
            elif mode[j] == 1:
                extreme[j] = max(extreme[j], p)
                if extreme[j] - p >= theta:
                    mode[j] = -1; extreme[j] = p; confirm[j] = p
            else:
                extreme[j] = min(extreme[j], p)
                if p - extreme[j] >= theta:
                    mode[j] = 1; extreme[j] = p; confirm[j] = p
            if mode[j] == 0:
                ready = False
            else:
                if mode[j] == 1:
                    state += 1 << j
                progress[i, j] = mode[j] * (extreme[j] - p) / theta
                overshoot[i, j] = mode[j] * (extreme[j] - confirm[j]) / theta
        if ready:
            states[i] = state
            events[i] = previous >= 0 and state != previous
            previous = state
    return states, segments, events, progress, overshoot


def fit_transition_matrix(previous, current, n_states=8):
    """Multinomial logit, one-hot previous state. Counts compress identical rows.

Half a pseudo-observation per non-self cell provides explicit smoothing.
Self transitions are impossible on the event clock and are masked/renormalized.
"""
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (previous, current), 1)
    a, b = np.where(~np.eye(n_states, dtype=bool))
    x = np.eye(n_states)[a]
    model = LogisticRegression(C=1., max_iter=2000, tol=1e-9, random_state=SEED)
    model.fit(x, b, sample_weight=counts[a, b] + .5)
    matrix = model.predict_proba(np.eye(n_states))
    np.fill_diagonal(matrix, 0.)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return model, matrix, counts


def rolling_surprise(probabilities, segment, k=K):
    surprise = -np.log(np.clip(probabilities, 1e-12, 1.))
    w = pd.Series(surprise).rolling(k, min_periods=k).mean().to_numpy().copy()
    same = np.zeros(len(w), bool)
    if len(w) >= k:
        same[k-1:] = segment[k-1:] == segment[:len(segment)-k+1]
    w[~same] = np.nan
    return w


def liquidity_scores(w, reference):
    reference = np.sort(np.asarray(reference)[np.isfinite(reference)])
    if len(reference) < 100:
        raise ValueError('Insufficient calibration transitions for phi')
    mu, sd = float(reference.mean()), float(reference.std(ddof=1))
    if not sd > 1e-10:
        raise ValueError('Degenerate surprise calibration')
    gaussian = ndtr(-(w - mu) / sd)
    # Smoothed empirical survival, preserving nonzero tail mass.
    empirical = (1 + len(reference) - np.searchsorted(reference, w, side='right')) / (len(reference) + 1)
    empirical[~np.isfinite(w)] = np.nan
    return gaussian, empirical, {'mean_W': mu, 'std_W': sd, 'n_reference': len(reference)}


def trailing_sum(values, boundary_indices, window):
    """values[i] describes [i,i+1); query at q includes values[q-window:q]."""
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0., np.cumsum(values)]
    q = np.asarray(boundary_indices, dtype=np.int64)
    good = (q >= window) & (q <= len(values))
    out = np.full(len(q), np.nan)
    out[good] = cumulative[q[good]] - cumulative[q[good] - window]
    return out


def trailing_price_features(mid, fresh, q, window):
    """Exactly window 1-second returns; all window+1 boundary quotes required."""
    log_mid = np.log(mid)
    r = np.r_[np.nan, np.diff(log_mid)]
    valid_r = fresh & np.r_[False, fresh[:-1]] & np.isfinite(r)
    variance = trailing_sum(np.where(valid_r, r*r, 0.), q, window)
    missing = trailing_sum((~valid_r).astype(float), q, window)
    valid = (q > window) & (q <= len(mid)) & (missing == 0)
    ret = np.full(len(q), np.nan)
    ret[valid] = (log_mid[q[valid]-1] - log_mid[q[valid]-window-1]) * 10000
    rv = np.where(valid, np.sqrt(np.maximum(variance, 0.)) * 10000, np.nan)
    return ret, rv


def future_labels(times, detections, horizons=HORIZONS, ends=None):
    """Next supplied event timestamp in (t,t+h]; boundary-crossers unknown.

The caller supplies retrospective onset labels and an end reduced by the
5-second detection delay, so label availability also stays in its partition.
"""
    times = np.asarray(times, dtype=np.int64)
    detections = np.sort(np.unique(detections))
    j = np.searchsorted(detections, times, side='right')
    next_time = np.full(len(times), np.iinfo(np.int64).max, dtype=np.int64)
    ok = j < len(detections)
    next_time[ok] = detections[j[ok]]
    ends = partition_end(times) if ends is None else np.asarray(ends)
    result = {}
    for h in horizons:
        # Strict < preserves disjoint UTC-date partitions at midnight.
        available = times + h < ends
        result[f'target_burst_{h}s'] = np.where(available, (next_time <= times+h).astype(np.int8), -1)
    return result, next_time


def build_features(symbol):
    print('Building', symbol, flush=True)
    sec = pl.read_parquet(lb.OUT / f'seconds_{symbol}.parquet')
    mid = sec['mid_close'].to_numpy()
    age = sec['quote_age_upper_s'].to_numpy()
    fresh = (age <= 2) & np.isfinite(mid) & (mid > 0)
    state, segment, event, progress, overshoot = directional_changes(
        np.log(mid), fresh, np.array(THETA_BPS) / 10000)
    e = np.flatnonzero(event)
    event_time = START + e + 1
    previous = state[e-1].astype(int)
    current = state[e].astype(int)
    fit = event_time < FIT_END
    model, matrix, counts = fit_transition_matrix(previous[fit], current[fit])
    joblib.dump(model, OUT / f'transition_logistic_{symbol}.joblib')
    probabilities = matrix[previous, current]
    w = rolling_surprise(probabilities, segment[e])
    reference = w[(event_time >= FIT_END) & (event_time < READY)]
    score, empirical, calibration = liquidity_scores(w, reference)
    (OUT / f'phi_{symbol}.json').write_text(json.dumps(calibration, indent=2))
    np.save(OUT / f'phi_reference_{symbol}.npy', reference[np.isfinite(reference)])
    a, b = np.indices(matrix.shape)
    pd.DataFrame({'symbol': symbol, 'previous_state': a.ravel(), 'next_state': b.ravel(),
                  'fit_count': counts.ravel(), 'probability': matrix.ravel()}).to_csv(
                      OUT / f'transition_matrix_{symbol}.csv', index=False)
    transitions = pd.DataFrame({'timestamp': event_time*1_000_000, 'symbol': symbol,
                                'segment': segment[e], 'previous_state': previous,
                                'state': current, 'transition_probability': probabilities,
                                'W_20': w, 'liquidity_L': score, 'L_empirical': empirical})
    # Never present fitted nuisance scores as causally available before freeze.
    transitions.loc[event_time < READY, ['transition_probability','W_20','liquidity_L','L_empirical']] = np.nan
    transitions.to_parquet(OUT / f'dc_events_{symbol}.parquet', index=False)

    episodes = pd.read_parquet(lb.OUT / 'episodes.parquet')
    episodes = episodes[episodes.symbol.eq(symbol)]
    snapshots = episodes[['episode_id','symbol','exchange','side','classification',
                          'onset_timestamp','first_detection_timestamp','first_cascade_detection_timestamp']].copy()
    snap1 = snapshots.assign(anchor='first_detection', timestamp=snapshots.first_detection_timestamp)
    snap2 = snapshots[snapshots.first_cascade_detection_timestamp.notna()].copy()
    snap2['anchor'] = 'cascade_detection'; snap2['timestamp'] = snap2.first_cascade_detection_timestamp.astype('int64')
    snapshot_meta = pd.concat([snap1, snap2], ignore_index=True)
    snapshot_meta['class_known_at_anchor'] = np.where(snapshot_meta.anchor.eq('cascade_detection'),
        'Potential liquidation cascade', 'Liquidation burst')
    q = np.unique(np.r_[e + 1, snapshot_meta.timestamp.to_numpy(dtype=np.int64)//1_000_000-START])
    q = q[(q > 0) & (q < len(mid))]
    t = START + q
    frame = pd.DataFrame({'timestamp': t*1_000_000, 'symbol': symbol,
                          'split': partition(t), 'symbol_eth': int(symbol == 'ethusdt'),
                          'is_dc_event': event[q-1], 'dc_state': state[q-1],
                          'midpoint': mid[q-1], 'bbo_age_upper_s': age[q-1]})
    for j, theta in enumerate(THETA_BPS):
        name = str(theta).replace('.', 'p')
        frame[f'dc_up_{name}bps'] = np.where(state[q-1] >= 0, (state[q-1] >> j) & 1, np.nan)
        frame[f'dc_reversal_progress_{name}bps'] = progress[q-1,j]
        frame[f'dc_overshoot_{name}bps'] = overshoot[q-1,j]
    del progress, overshoot
    last = np.searchsorted(event_time, t, side='right')-1
    good = (last >= 0) & (t >= READY)
    safe = np.maximum(last, 0)
    good &= segment[q-1] == segment[e[safe]]
    for name, values in [('W_20',w), ('liquidity_L',score), ('L_empirical',empirical),
                         ('last_transition_probability', probabilities)]:
        frame[name] = np.where(good, values[safe], np.nan)
    frame['dc_age_s'] = np.where(good, t-event_time[safe], np.nan)
    span = np.full(len(e), np.nan)
    span[K-1:] = event_time[K-1:] - event_time[:len(e)-K+1]
    frame['dc_20_span_s'] = np.where(good & np.isfinite(w[safe]), span[safe], np.nan)
    trade = sec['trade_notional'].to_numpy()
    trade_count = sec['trade_count'].to_numpy()
    for lag in LAGS:
        ret, rv = trailing_price_features(mid, fresh, q, lag)
        frame[f'return_{lag}s_bps'] = ret
        frame[f'rv_{lag}s_bps'] = rv
        frame[f'trade_notional_{lag}s'] = trailing_sum(trade, q, lag)
        frame[f'trade_count_{lag}s'] = trailing_sum(trade_count, q, lag)
        frame[f'dc_count_{lag}s'] = trailing_sum(event, q, lag)
    # Current 5s volume versus disjoint preceding 20m average, available at t.
    hist_trade = trailing_sum(trade, q-5, 1200) / 240
    frame['trade_ratio_5s_vs_prior20m'] = frame.trade_notional_5s / np.maximum(hist_trade, 1.)

    global_burst = np.zeros(len(mid)//5, bool)
    cascade = global_burst.copy()
    # Counts retain source message semantics; known duplicates/anomalies marked below.
    frames, _ = lb.load_liquidations(symbol, sec)
    for exchange in lb.EXCHANGES:
        for side in lb.SIDES:
            prefix = f'{exchange}_{side}'
            raw = frames[exchange].loc[frames[exchange].side.eq(side)
                & ~frames[exchange].duplicate_extra_copy & ~frames[exchange].suspect_price]
            s = raw.effective_timestamp.to_numpy()//1_000_000 - START
            use = (s >= 0) & (s < len(mid))
            s = s[use].astype(int)
            c = np.bincount(s, minlength=len(mid))
            v = np.bincount(s, weights=raw.notional.to_numpy()[use], minlength=len(mid))
            for lag in LAGS:
                frame[f'{prefix}_liq_count_{lag}s'] = trailing_sum(c, q, lag)
                frame[f'{prefix}_liq_notional_{lag}s'] = trailing_sum(v, q, lag)
            hist_c = trailing_sum(c, q-5, 1200)/240
            hist_v = trailing_sum(v, q-5, 1200)/240
            frame[f'{prefix}_count_ratio_5s'] = frame[f'{prefix}_liq_count_5s'] / np.maximum(hist_c,1.)
            frame[f'{prefix}_notional_ratio_5s'] = frame[f'{prefix}_liq_notional_5s'] / np.maximum(hist_v,1000.)
            f = pl.read_parquet(lb.OUT / f'features_{symbol}_{exchange}_{side}.parquet',
                                columns=['burst','potential_cascade'])
            global_burst |= f['burst'].to_numpy()
            cascade |= f['potential_cascade'].to_numpy()
            del c,v,f
    current_bin = q//5-1
    safe_bin = np.maximum(current_bin, 0)
    frame['burst_known_now'] = (current_bin >= 0) & global_burst[safe_bin]
    frame['cascade_known_now'] = (current_bin >= 0) & cascade[safe_bin]
    # Aggregate coincident/overlapping stream bursts into causal symbol episodes.
    starts = np.flatnonzero(global_burst & ~np.r_[False, global_burst[:-1]])
    stops = np.flatnonzero(global_burst & ~np.r_[global_burst[1:], False])
    detections = START+(starts+1)*5
    global_catalog = pd.DataFrame({'symbol':symbol,
        'onset_timestamp':(START+starts*5)*1_000_000,
        'first_detection_timestamp':detections*1_000_000,
        'end_timestamp':(START+(stops+1)*5)*1_000_000,
        'split':partition(detections-5),
        'has_potential_cascade':[bool(cascade[a:b+1].any()) for a,b in zip(starts,stops)]})
    global_catalog.to_parquet(OUT / f'forecast_episodes_{symbol}.parquet', index=False)
    targets, next_time = future_labels(t, detections-5, ends=partition_end(t)-5)
    for name, val in targets.items(): frame[name] = val
    safe_next = np.where(next_time < END+1, next_time, -1)
    frame['next_burst_onset_timestamp'] = np.where(safe_next >= 0, safe_next*1_000_000, -1)
    frame['history_ready'] = (t >= READY) & (state[q-1] >= 0) & frame.rv_60s_bps.notna() & frame.W_20.notna()
    frame['forecast_eligible'] = frame.history_ready & ~frame.burst_known_now
    snapshot_meta = snapshot_meta.merge(frame, on=['timestamp','symbol'], how='left', validate='many_to_one')
    snapshot_meta.to_parquet(OUT / f'episode_market_states_{symbol}.parquet', index=False)
    snapshot_meta.to_csv(OUT / f'episode_market_states_{symbol}.csv', index=False)
    frame = frame.copy()
    numeric = frame.select_dtypes(include=['float64']).columns
    frame[numeric] = frame[numeric].astype('float32')
    frame.to_parquet(OUT / f'market_states_{symbol}.parquet', index=False)
    print(symbol, 'DC events', len(e), 'market states', len(frame), 'snapshots', len(snapshot_meta), flush=True)
    return {'symbol':symbol,'seconds':len(mid),'dc_events':len(e),'market_rows':len(frame),
            'snapshot_rows':len(snapshot_meta),'fit_transitions':int(fit.sum()), **calibration}


def feature_columns(frame, include_liquidity=True):
    exact = {'symbol_eth','bbo_age_upper_s','dc_age_s','dc_20_span_s','trade_ratio_5s_vs_prior20m',
             'W_20','liquidity_L','L_empirical','last_transition_probability'}
    starts = ('rv_', 'return_', 'trade_notional_', 'trade_count_', 'dc_up_',
              'dc_reversal_', 'dc_overshoot_', 'dc_count_', 'binance_', 'bybit_')
    columns = [c for c in frame if c in exact or c.startswith(starts)]
    if not include_liquidity:
        columns = [c for c in columns if c not in {'W_20','liquidity_L','L_empirical','last_transition_probability'}]
    return columns


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def metrics(y, p, threshold):
    alarm = p >= threshold
    tp = int((alarm & (y == 1)).sum())
    return {'n':len(y), 'positives':int(y.sum()), 'prevalence':float(y.mean()),
            'average_precision':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
            'log_loss':float(log_loss(y,p,labels=[0,1])), 'brier':float(brier_score_loss(y,p)),
            'threshold':threshold,'alerts':int(alarm.sum()),
            'precision':tp/max(1,int(alarm.sum())), 'recall':tp/max(1,int(y.sum()))}


def event_metrics(frame, probabilities, threshold, episodes, horizon):
    """Episode coverage including no-DC-event episodes; strict onset leads separate."""
    rows=[]
    for symbol in lb.SYMBOLS:
        f=frame[frame.symbol.eq(symbol)]
        p=probabilities[frame.symbol.eq(symbol).to_numpy()]
        eligible_times=f.timestamp.to_numpy(dtype=np.int64)//1_000_000
        alerts=eligible_times[p >= threshold]
        for e in episodes[episodes.symbol.eq(symbol)].itertuples():
            onset=e.onset_timestamp//1_000_000; detect=e.first_detection_timestamp//1_000_000
            idx=np.searchsorted(alerts,onset-horizon,side='left')
            a=alerts[idx] if idx<len(alerts) and alerts[idx]<onset else None
            ix=np.searchsorted(eligible_times,onset-horizon,side='left')
            possible=ix<len(eligible_times) and eligible_times[ix]<onset
            later=np.searchsorted(alerts,detect-horizon,side='left')
            detected=later<len(alerts) and alerts[later]<detect
            rows.append({'symbol':symbol,'first_detection_timestamp':e.first_detection_timestamp,
                         'onset_timestamp':e.onset_timestamp,'has_potential_cascade':e.has_potential_cascade,
                         'opportunity':bool(possible),'warned_before_detection':bool(detected),
                         'warned_before_onset':a is not None and a<onset,
                         'lead_to_detection_s': detect-a if a is not None else np.nan,
                         'lead_to_onset_s': onset-a if a is not None else np.nan})
    return pd.DataFrame(rows)


def train_models():
    print('Loading event-time training population',flush=True)
    data=pd.concat([pd.read_parquet(OUT/f'market_states_{s}.parquet') for s in lb.SYMBOLS],ignore_index=True)
    data=data[data.is_dc_event & data.forecast_eligible].sort_values(['timestamp','symbol']).reset_index(drop=True)
    catalog=pd.concat([pd.read_parquet(OUT/f'forecast_episodes_{s}.parquet') for s in lb.SYMBOLS],ignore_index=True)
    cols=feature_columns(data)
    (OUT/'feature_columns.json').write_text(json.dumps(cols,indent=2))
    summary=[]; frozen=[]; importance=[]; daily=[]; predictions=[]; event_rows=[]
    params=dict(objective='binary',n_estimators=400,num_leaves=15,max_depth=5,
                learning_rate=.05,min_child_samples=100,reg_lambda=5.,
                max_bin=127,verbosity=-1,n_jobs=4,random_state=SEED,
                deterministic=True,force_col_wise=True)
    for h in HORIZONS:
        label=f'target_burst_{h}s'
        d=data[data[label]>=0]
        train=d.split.eq('train').to_numpy(); val=d.split.eq('validation').to_numpy()
        test=d.split.eq('test').to_numpy()
        y=d[label].to_numpy(dtype=int)
        prior=float(y[train].mean())
        for variant in ('without_liquidity_score','full'):
            use=feature_columns(d,include_liquidity=variant=='full')
            x=d[use]
            model=lgb.LGBMClassifier(**params)
            model.fit(x[train],y[train],eval_set=[(x[val],y[val])],eval_metric='binary_logloss',
                      callbacks=[lgb.early_stopping(35,verbose=False)])
            pv=model.predict_proba(x[val])[:,1]
            threshold=choose_threshold(y[val],pv)
            model.booster_.save_model(str(OUT/f'model_{variant}_{h}s.txt'))
            # Persist tuning decisions before reading any test predictions/metrics.
            frozen.append({'horizon_s':h,'variant':variant,'best_iteration':model.best_iteration_,
                           'threshold':threshold,'features':use,'train_prevalence':prior})
            (OUT/'frozen_models.json').write_text(json.dumps({'parameters':params,'models':frozen},indent=2))
            for phase,mask in [('validation',val),('test',test)]:
                p=pv if phase=='validation' else model.predict_proba(x[mask])[:,1]
                m=metrics(y[mask],p,threshold)
                summary.append({'split':phase,'horizon_s':h,'model':variant,**m})
                pred=d.loc[mask,['timestamp','symbol','split',label]].copy().rename(columns={label:'target'})
                pred['probability']=p;pred['horizon_s']=h;pred['model']=variant
                pred['alarm']=p>=threshold
                predictions.append(pred)
                for day,g in pred.groupby(pd.to_datetime(pred.timestamp,unit='us',utc=True).dt.strftime('%Y-%m-%d')):
                    daily.append({'day':day,'split':phase,'horizon_s':h,'model':variant,
                                  **metrics(g.target.to_numpy(),g.probability.to_numpy(),threshold)})
                ev=catalog[catalog.split.eq(phase)]
                em=event_metrics(d.loc[mask],p,threshold,ev,h)
                em['split']=phase;em['horizon_s']=h;em['model']=variant
                event_rows.append(em)
            for c,gain in zip(use,model.booster_.feature_importance(importance_type='gain')):
                importance.append({'horizon_s':h,'model':variant,'feature':c,'gain':float(gain)})
            print('Fit',h,variant,'train',train.sum(),'val',val.sum(),'test',test.sum(),
                  'trees',model.best_iteration_,flush=True)
            del model,x;gc.collect()
        for phase,mask in [('validation',val),('test',test)]:
            summary.append({'split':phase,'horizon_s':h,'model':'constant_train_prevalence',
                            **metrics(y[mask],np.full(mask.sum(),prior),.5)})
    pd.DataFrame(summary).to_csv(OUT/'model_metrics.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(event_rows,ignore_index=True).to_csv(OUT/'event_warning_evaluation.csv',index=False)
    data.groupby(['split','symbol']).agg(observations=('timestamp','size'),start=('timestamp','min'),end=('timestamp','max')).to_csv(OUT/'training_population.csv')


def record_specification():
    OUT.mkdir(exist_ok=True)
    days=pd.DataFrame({'day':pd.date_range('2025-12-01','2026-02-28',tz='UTC')})
    # Use timestamp() instead of depending on datetime internal resolution.
    times=np.array([int(x.timestamp()) for x in days.day])
    days['split']=partition(times)
    days['role']=np.where(times<FIT_END,'transition_fit',np.where(times<READY,'phi_and_detector_calibration',days.split))
    days.to_csv(OUT/'calendar_split.csv',index=False)
    spec={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
          'lags_seconds':LAGS,'forecast_horizons_seconds':HORIZONS,'theta_bps':THETA_BPS,'k_transitions':K,
          'price_sampling':'last BBO strictly before each second boundary; one-second sampled DC approximation',
          'target':'new symbol-wide burst onset in (t,t+h]; onset is a retrospective label (detection - 5s), never a feature; current known burst excluded causally',
          'strict_early_warning':'alert in [onset-h,onset), strictly before onset; detection-time coverage separately reported',
          'split':'60/15/15 complete UTC dates; first 14 train days are nuisance calibration, not supervised training',
          'test_caveat':'2026-02-14..28 was viewed in prior EDA; historical reuse, not virgin holdout',
          'phi':'Gaussian CDF standardized by held-out calibration W_20; empirical survival as sensitivity feature',
          'markov':'first-order approximation; not an established property of observed price states',
          'open_interest':'absent from all 8 input files; omitted with user agreement, no volume substitution',
          'liquidation_quality':'predictor counts exclude exact duplicate extras and causal suspect-price flags; target detector unchanged, hence labels remain provisional',
          'lightgbm':'binary classifier; binary log loss; no class reweighting/downsampling; early stopping and threshold on validation only',
          'external_sources':[
              'https://arxiv.org/html/1402.2198',
              'https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html',
              'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html']}
    (OUT/'experiment_specification.json').write_text(json.dumps(spec,indent=2))
    env={'python':platform.python_version(),'lightgbm':lgb.__version__,'sklearn':sklearn.__version__,
         'numpy':np.__version__,'numba':numba.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))


def write_manifest():
    sources=[lb.OUT/'run_manifest.json',*lb.OUT.glob('seconds_*.parquet'),
             *lb.OUT.glob('features_*.parquet'),lb.OUT/'episodes.parquet',
             *(ROOT/'src').glob('liquidation_early_warning*.py')]
    outputs=[p for p in OUT.iterdir() if p.is_file() and p.name not in
             {'run_manifest.json','pytest_results.xml'}]
    manifest={'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},
              'outputs':{p.name:digest(p) for p in outputs}}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


def verify_saved():
    m=json.loads((OUT/'run_manifest.json').read_text())
    for p,h in m['sources'].items():
        if digest(resolve_legacy(p))!=h:raise ValueError('Changed source '+p)
    for p,h in m['outputs'].items():
        if digest(OUT/p)!=h:raise ValueError('Changed output '+p)
    return m


def main(stage='all'):
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    if stage in ('all','features'):
        record_specification()
        lb.verify_saved_run()
        quality=[]
        for symbol in lb.SYMBOLS:
            quality.append(build_features(symbol));gc.collect()
        pd.DataFrame(quality).to_csv(OUT/'feature_build_summary.csv',index=False)
    if stage in ('all','train'):train_models()
    if stage in ('all','train','manifest'):write_manifest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','features','train','manifest'])
    args=parser.parse_args();main(args.stage)
