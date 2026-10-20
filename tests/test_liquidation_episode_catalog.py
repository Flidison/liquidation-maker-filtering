"""Reconcile every marked detector bin to exactly one saved burst episode."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import liquidation_bursts as lb


@pytest.mark.integration
@pytest.mark.parametrize('symbol',lb.SYMBOLS)
@pytest.mark.parametrize('exchange',lb.EXCHANGES)
@pytest.mark.parametrize('side',lb.SIDES)
def test_complete_episode_partition_and_first_cascade_time(symbol,exchange,side):
    f=pl.read_parquet(lb.OUT/f'features_{symbol}_{exchange}_{side}.parquet',
        columns=['timestamp','phase','burst','potential_cascade','liquidation_count','liquidation_notional']).to_pandas()
    e=pd.read_parquet(lb.OUT/'episodes.parquet')
    e=e.loc[e.symbol.eq(symbol)&e.exchange.eq(exchange)&e.side.eq(side)].sort_values('first_detection_timestamp')
    marked=f.loc[f.burst].set_index('timestamp')
    covered=[]
    for r in e.itertuples():
        ticks=np.arange(r.first_detection_timestamp,r.end_timestamp+1,5_000_000,dtype=np.int64)
        windows=marked.loc[ticks]
        assert windows.phase.eq(r.phase).all()
        assert len(windows)==r.flagged_windows
        assert windows.liquidation_count.sum()==r.liquidation_count
        assert windows.liquidation_notional.sum()==pytest.approx(r.liquidation_notional,abs=1e-6)
        confirms=windows.index[windows.potential_cascade]
        if len(confirms):
            assert r.classification=='Potential liquidation cascade'
            assert int(confirms.min())==r.first_cascade_detection_timestamp
        else:
            assert r.classification=='Liquidation burst'
            assert pd.isna(r.first_cascade_detection_timestamp)
        assert r.onset_timestamp==r.first_detection_timestamp-5_000_000
        covered.extend(ticks)
    np.testing.assert_array_equal(np.asarray(covered),marked.index.to_numpy())
    # Adjacent marked bins in the same phase must not be split into two episodes.
    split=(e.first_detection_timestamp.to_numpy()[1:]-e.end_timestamp.to_numpy()[:-1])==5_000_000
    same=e.phase.to_numpy()[1:]==e.phase.to_numpy()[:-1]
    assert not np.any(split&same)
