import numpy as np
import pandas as pd
import pytest

import evt_log_cond_calibration_5s as calibration


def test_quantile_bins_do_not_split_ties_and_cover_zero_one():
    p = np.r_[np.zeros(60), np.full(30, .2), np.ones(10)]
    edges = calibration.bin_boundaries(p)
    y = np.zeros(100, dtype=int)
    y[-4:] = 1
    table = calibration.grouped_calibration(y, p, edges)
    assert table.n.sum() == 100
    assert table.positives.sum() == 4
    assert len(table) < 10
    assert table.loc[table.predicted.eq(0), 'n'].item() == 60
    assert np.sum(table.n * table.predicted) / 100 == pytest.approx(p.mean())


def test_saved_calibration_reconciles_test_population_and_weighting():
    bins = pd.read_csv(calibration.OUT / 'calibration_bins.csv')
    metrics = pd.read_csv(calibration.OUT / 'calibration_metrics.csv')
    for row in metrics.itertuples():
        f = bins.loc[bins.model.eq(row.model) & bins['split'].eq(row.split)]
        assert f.n.sum() == row.n
        assert f.positives.sum() == row.positives
        assert np.average(f.predicted, weights=f.n) == pytest.approx(row.mean_probability, abs=1e-14)
        assert np.average(f.observed, weights=f.n) == pytest.approx(row.observed_frequency, abs=1e-14)
        assert np.average(np.abs(f.predicted - f.observed), weights=f.n) == pytest.approx(row.ece, abs=1e-14)
