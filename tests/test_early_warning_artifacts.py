"""Checks on all new stored rows; no claim that market hypotheses must be true."""
import json
import numpy as np
import pandas as pd
import polars as pl
import pytest
import liquidation_early_warning as ew

pytestmark=pytest.mark.integration


@pytest.fixture(scope='module',params=['btcusdt','ethusdt'])
def stored(request):
    s=request.param
    return s,pd.read_parquet(ew.OUT/f'market_states_{s}.parquet')


def test_all_sources_and_code_unchanged():
    ew.verify_saved()


def test_all_snapshots_and_causal_known_class(stored):
    s,_=stored
    f=pd.read_parquet(ew.OUT/f'episode_market_states_{s}.parquet')
    original=pd.read_parquet(ew.lb.OUT/'episodes.parquet')
    original=original[original.symbol.eq(s)]
    assert f.episode_id.nunique()==len(original)
    assert len(f)==len(original)+original.first_cascade_detection_timestamp.notna().sum()
    assert not f.duplicated(['episode_id','anchor']).any()
    assert f.timestamp.notna().all()
    assert f.query("anchor=='first_detection'").class_known_at_anchor.eq('Liquidation burst').all()
    assert f.query("anchor=='cascade_detection'").class_known_at_anchor.eq('Potential liquidation cascade').all()


def test_feature_causality_availability_and_no_oi(stored):
    s,f=stored
    cols=ew.feature_columns(f)
    assert not any(c.startswith('open_interest') for c in f)
    assert not any('target' in c or 'timestamp' in c or 'classification' in c for c in cols)
    assert f.timestamp.is_unique
    assert f.timestamp.is_monotonic_increasing
    t=f.timestamp.to_numpy()//1_000_000
    assert f.loc[t<ew.READY,['W_20','liquidity_L','last_transition_probability']].isna().all().all()
    assert not np.isinf(f[cols].to_numpy()).any()
    assert f.loc[f.forecast_eligible,'rv_60s_bps'].notna().all()
    assert f.loc[f.forecast_eligible,'W_20'].notna().all()
    assert not (f.forecast_eligible & f.burst_known_now).any()
    np.testing.assert_array_equal(f.split.to_numpy(),ew.partition(t))


def test_every_target_against_search_left_interval_counts(stored):
    s,f=stored
    e=pd.read_parquet(ew.OUT/f'forecast_episodes_{s}.parquet')
    onsets=np.sort(e.onset_timestamp.to_numpy()//1_000_000)
    t=f.timestamp.to_numpy()//1_000_000
    start=np.searchsorted(onsets,t,side='right')
    for h in ew.HORIZONS:
        end=np.searchsorted(onsets,t+h,side='right')
        observed=t+h+5<ew.partition_end(t)
        expected=np.where(observed,(end>start).astype(int),-1)
        np.testing.assert_array_equal(f[f'target_burst_{h}s'].to_numpy(),expected)


def test_random_historical_features_from_direct_second_slices(stored):
    s,f=stored
    sec=pl.read_parquet(ew.lb.OUT/f'seconds_{s}.parquet')
    mid=sec['mid_close'].to_numpy();age=sec['quote_age_upper_s'].to_numpy()
    volume=sec['trade_notional'].to_numpy()
    sample=f.sample(n=150,random_state=77)
    for row in sample.itertuples():
        q=row.timestamp//1_000_000-ew.START
        for h in ew.LAGS:
            if q<=h:continue
            p=mid[q-h-1:q]
            r=np.diff(np.log(p))*10000
            valid=np.isfinite(p).all() and (age[q-h-1:q]<=2).all()
            value=getattr(row,f'rv_{h}s_bps')
            if valid:
                assert value==pytest.approx(np.sqrt(np.sum(r*r)),rel=2e-5,abs=1e-5)
                assert getattr(row,f'return_{h}s_bps')==pytest.approx(r.sum(),rel=2e-5,abs=1e-5)
            else:assert np.isnan(value)
            assert getattr(row,f'trade_notional_{h}s')==pytest.approx(volume[q-h:q].sum(),rel=2e-5,abs=.05)


def test_all_markov_surprises_from_saved_transition_table(stored):
    s,f=stored
    d=pd.read_parquet(ew.OUT/f'dc_events_{s}.parquet')
    table=pd.read_csv(ew.OUT/f'transition_matrix_{s}.csv')
    matrix=table.pivot(index='previous_state',columns='next_state',values='probability').to_numpy()
    np.testing.assert_allclose(matrix.sum(axis=1),1.,atol=1e-12)
    p=matrix[d.previous_state.to_numpy(),d.state.to_numpy()]
    gamma=-np.log(p)
    # Independent direct shifted sum of last k transition surprises.
    expected=np.full(len(d),np.nan)
    total=np.zeros(len(d)-ew.K+1)
    for k in range(ew.K):total+=gamma[k:k+len(total)]
    expected[ew.K-1:]=total/ew.K
    good=np.zeros(len(d),bool)
    good[ew.K-1:]=d.segment.to_numpy()[ew.K-1:]==d.segment.to_numpy()[:len(d)-ew.K+1]
    good &= d.timestamp.to_numpy()>=ew.READY*1_000_000
    expected[~good]=np.nan
    np.testing.assert_allclose(d.W_20,expected,rtol=1e-10,atol=1e-10,equal_nan=True)


def test_fixed_models_and_prediction_denominators():
    config=json.loads((ew.OUT/'frozen_models.json').read_text())
    p=pd.read_parquet(ew.OUT/'predictions.parquet')
    m=pd.read_csv(ew.OUT/'model_metrics.csv')
    assert len(config['models'])==10
    assert len(m)==30
    assert p.probability.between(0,1).all()
    assert not p.duplicated(['timestamp','symbol','horizon_s','model']).any()
    for row in config['models']:
        assert 1<=row['best_iteration']<=config['parameters']['n_estimators']
        assert not any('target' in c or 'timestamp' in c for c in row['features'])
        f=p[(p.horizon_s==row['horizon_s']) & p.model.eq(row['variant'])]
        np.testing.assert_array_equal(f.alarm.to_numpy(),f.probability.to_numpy()>=row['threshold'])
        for split in ('validation','test'):
            g=f[f.split.eq(split)]
            rec=m[m.split.eq(split)&m.model.eq(row['variant'])&m.horizon_s.eq(row['horizon_s'])].iloc[0]
            assert len(g)==rec.n
            assert g.target.sum()==rec.positives
            assert g.alarm.sum()==rec.alerts
