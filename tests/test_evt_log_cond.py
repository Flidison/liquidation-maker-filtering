"""Boundary, causality, source reconciliation and sealed-test checks."""
import json
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import brier_score_loss

import evt_log_cond_modeling as m


def quote_batch(ts, bid):
    ts = np.asarray(ts, dtype=np.int64)
    bid = np.asarray(bid, dtype=float)
    return {'timestamp':ts,'bid_price':bid,'ask_price':bid+1,
            'bid_amount':bid/10,'ask_amount':bid/20}


def test_asof_strict_equal_times_and_split_timestamp_groups():
    batches = [quote_batch([10,20,20],[100,101,102]),quote_batch([20,30],[103,104])]
    actual = m.asof_quotes(batches, [5,10,15,20,21,30,31])
    np.testing.assert_array_equal(actual.quote_timestamp,[-1,-1,10,10,20,20,30])
    np.testing.assert_allclose(actual.bid_price,[np.nan,np.nan,100,100,103,103,104],equal_nan=True)


def test_asof_future_changes_leave_history_unchanged():
    old = m.asof_quotes([quote_batch([10,20,30],[100,101,102])],[15,20,25])
    new = m.asof_quotes([quote_batch([10,20,30],[100,101,999999])],[15,20,25])
    pd.testing.assert_frame_equal(old,new)


def test_asof_unsorted_feed_is_rejected():
    with pytest.raises(ValueError):m.asof_quotes([quote_batch([20,10],[100,101])],[25])


@pytest.mark.parametrize('window',[5,10,20])
def test_intensity_half_open_and_empty(window):
    q = 50_000_000
    ts = np.array([q-window*10**6-1,q-window*10**6,q-1,q,q+1])
    assert m.intensity(ts,[q],window)[0] == 2/window
    assert m.intensity([], [q], window)[0] == 0


def test_bybit_delay_changes_availability():
    ts = np.array([9_900_000]) + 200_000
    np.testing.assert_allclose(m.intensity(ts,[10_000_000,10_100_000,10_100_001],5),[0,0,.2])


def test_queue_difference_normalization_and_zero_denominator():
    diff, imbalance = m.queue_features([3,0,0,-1,np.nan],[1,4,0,2,1])
    np.testing.assert_allclose(diff,[2,-4,0,np.nan,np.nan],equal_nan=True)
    np.testing.assert_allclose(imbalance,[.5,-1,np.nan,np.nan,np.nan],equal_nan=True)


def test_nonoverlapping_return_lags_and_causality():
    mid = np.exp(np.arange(100)**2/10000)
    age = np.ones(100)
    result = m.lagged_returns(mid,age,np.array([60]))
    for j in range(10):
        expected = (59-5*j)**2-(54-5*j)**2
        assert result[f'log_return_5s_lag{j}_bps'][0] == pytest.approx(expected)
    changed = mid.copy();changed[60:] = 9999
    for name, value in m.lagged_returns(changed,age,np.array([60])).items():
        np.testing.assert_array_equal(value,result[name])
    age[9] = 3
    assert np.isnan(m.lagged_returns(mid,age,np.array([60]))['log_return_5s_lag9_bps'][0])


def test_metrics_are_probability_mse_and_binary_macro_f1():
    y=np.array([0,0,0,1]);p=np.array([.1,.2,.3,.4])
    result=m.evaluate(y,p,.35)
    assert result['mse']==pytest.approx(np.mean((y-p)**2))
    assert result['mse']==pytest.approx(brier_score_loss(y,p))
    assert result['f1']==1 and result['f1_macro']==1


def test_test_build_is_sealed_until_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    with pytest.raises(RuntimeError,match='sealed'):m.build_features('test')


def test_retuning_is_rejected_after_test(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    (tmp_path/'test_evaluation_receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='retune'):m.tune()


def test_repeated_test_call_only_returns_saved_results(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    (tmp_path/'test_evaluation_receipt.json').write_text('{}')
    pd.DataFrame({'mse':[.12]}).to_csv(tmp_path/'test_metrics.csv',index=False)
    monkeypatch.setattr(m,'build_features',lambda *args:pytest.fail('Repeated test feature access'))
    assert m.evaluate_test_once().mse[0]==.12


@pytest.mark.parametrize('lags',range(5,11))
def test_feature_allowlist(lags):
    features=m.feature_names(lags)
    assert len([c for c in features if 'lag' in c])==lags
    assert not any('target' in c or 'timestamp' in c or 'burst' in c for c in features)


@pytest.mark.integration
def test_saved_development_against_raw_liquidations_and_direct_prices():
    path=m.OUT/'features_development.parquet'
    if not path.exists():pytest.skip('Feature build not finished')
    f=pd.read_parquet(path)
    assert not f.split.eq('test').any()
    assert not f.duplicated(['timestamp','symbol']).any()
    for symbol in m.lb.SYMBOLS:
        sample=f[f.symbol.eq(symbol)].sample(75,random_state=19)
        sec=pl.read_parquet(m.lb.OUT/f'seconds_{symbol}.parquet',columns=['mid_close','quote_age_upper_s'])
        mid=sec['mid_close'].to_numpy();age=sec['quote_age_upper_s'].to_numpy()
        for ex in m.lb.EXCHANGES:
            raw=pl.read_parquet(m.lb.source_path(symbol,f'{ex}_liquidations')).to_pandas()
            ts=raw.timestamp.to_numpy()+(200_000 if ex=='bybit' else 0)
            for side in m.lb.SIDES:
                for w in m.WINDOWS:
                    expected=[int(((raw.side.to_numpy()==side)&(ts>=t-w*10**6)&(ts<t)).sum())/w for t in sample.timestamp]
                    np.testing.assert_allclose(sample[f'{ex}_{side}_intensity_{w}s'],expected,rtol=1e-6)
        for row in sample.itertuples():
            q=row.timestamp//10**6-m.START
            for w in m.WINDOWS:
                prices=mid[q-w-1:q]
                valid=np.isfinite(prices).all() and (age[q-w-1:q]<=2).all()
                actual=getattr(row,f'rv_{w}s_bps')
                if valid:assert actual==pytest.approx(np.sqrt(np.sum(np.diff(np.log(prices))**2))*10000,rel=2e-5,abs=1e-5)
                else:assert np.isnan(actual)
            if row.usable:
                assert row.timestamp+ m.PURGE_SECONDS*10**6 < m.ew.partition_end(np.array([row.timestamp//10**6]))[0]*10**6
                assert row.quote_timestamp<row.timestamp
    q=f.queue_bid_base+f.queue_ask_base
    np.testing.assert_allclose(f.queue_difference_base,f.queue_bid_base-f.queue_ask_base,rtol=1e-5,atol=1e-4,equal_nan=True)
    np.testing.assert_allclose(f.queue_imbalance,(f.queue_bid_base-f.queue_ask_base)/q,rtol=1e-4,atol=1e-6,equal_nan=True)


@pytest.mark.integration
def test_real_quote_snapshots_against_direct_raw_windows():
    path=m.OUT/'features_development.parquet'
    if not path.exists():pytest.skip('Feature build not finished')
    f=pd.read_parquet(path)
    for symbol in m.lb.SYMBOLS:
        sample=f[f.symbol.eq(symbol)&f.usable].sample(5,random_state=217)
        for row in sample.itertuples():
            raw=pd.read_parquet(m.lb.source_path(symbol,'binance_booktickers'),
                filters=[('timestamp','>=',row.timestamp-2_000_000),('timestamp','<',row.timestamp)])
            last=raw.iloc[-1]
            assert row.quote_timestamp==int(last.timestamp)
            assert row.queue_bid_base==pytest.approx(last.bid_amount,rel=1e-6)
            assert row.queue_ask_base==pytest.approx(last.ask_amount,rel=1e-6)


@pytest.mark.integration
def test_scaler_was_fitted_only_on_training_rows():
    path=m.OUT/'train_only_scaler.joblib'
    if not path.exists():pytest.skip('Training not finished')
    scaler=joblib_load(path)
    f=pd.read_parquet(m.OUT/'features_development.parquet')
    train=f[f.usable&f.split.eq('train')]
    assert scaler.n_samples_seen_==len(train)
    np.testing.assert_allclose(scaler.mean_,train[m.feature_names(10)].to_numpy(dtype=float).mean(axis=0),rtol=1e-12,atol=1e-12)


@pytest.mark.integration
def test_all_targets_against_independent_episode_catalog():
    for period in ('development','test'):
        path=m.OUT/f'features_{period}.parquet'
        if not path.exists():continue
        f=pd.read_parquet(path)
        for symbol in m.lb.SYMBOLS:
            g=f[f.symbol.eq(symbol)]
            episodes=pd.read_parquet(m.ew.OUT/f'forecast_episodes_{symbol}.parquet')
            onset=np.sort(episodes.onset_timestamp.to_numpy()//10**6)
            t=g.timestamp.to_numpy()//10**6
            for h in m.HORIZONS:
                n=np.searchsorted(onset,t+h,side='right')-np.searchsorted(onset,t,side='right')
                expected=np.where(t+h+5<m.ew.partition_end(t),(n>0).astype(int),-1)
                np.testing.assert_array_equal(g[f'target_burst_{h}s'].to_numpy(),expected)


@pytest.mark.integration
def test_saved_model_selections_and_final_predictions():
    if not (m.OUT/'test_evaluation_receipt.json').exists():pytest.skip('Final test not evaluated')
    m.verify_results()
    trials=pd.read_csv(m.OUT/'validation_search.csv')
    frozen=json.loads((m.OUT/'frozen_selection.json').read_text())
    assert len(trials)==5*(6*5+6*3*3*10)
    for choice in frozen['models']:
        candidates=trials[trials.horizon_s.eq(choice['horizon_s'])&trials.model.eq(choice['model'])&trials.converged]
        assert choice['validation_mse']==pytest.approx(candidates.validation_mse.min())
        assert choice['n_lags'] in m.LAG_COUNTS
        if choice['model']=='lightgbm':
            model=joblib_load(m.OUT/choice['model_file'])
            assert model.n_estimators_==choice['n_estimators']
    p=pd.read_parquet(m.OUT/'predictions_test.parquet')
    metrics=pd.read_csv(m.OUT/'test_metrics.csv')
    assert p.probability.between(0,1).all()
    for (h,model),f in p.groupby(['horizon_s','model']):
        row=metrics[metrics.horizon_s.eq(h)&metrics.model.eq(model)].iloc[0]
        expected=m.evaluate(f.target.to_numpy(),f.probability.to_numpy(),row.threshold)
        for metric in ['mse','precision','recall','f1','f1_macro']:
            assert row[metric]==pytest.approx(expected[metric])


def joblib_load(path):
    import joblib
    return joblib.load(path)
