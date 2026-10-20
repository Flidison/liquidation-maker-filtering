"""Validate the single test pass without making any more test predictions."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import evt_log_cond_test_reduced_5s as assessment
import evt_log_cond_reduced_5s as reduced


@pytest.mark.integration
def test_completed_test_run_never_loads_estimator_again(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError('A completed test run must not load a model for new inference')
    monkeypatch.setattr(joblib, 'load', forbid)
    metrics = assessment.evaluate_once()
    assert set(metrics['split']) == {'test'}
    receipt = assessment.verify_results()
    assert receipt['test_prediction_passes_new_model'] == 1
    assert receipt['models_fitted'] == 0


def test_interrupted_pass_cannot_silently_predict_again(tmp_path, monkeypatch):
    freeze = tmp_path / 'frozen_evaluation.json'
    freeze.write_text('{}')
    monkeypatch.setattr(assessment, 'FREEZE', freeze)
    monkeypatch.setattr(assessment, 'RECEIPT', tmp_path / 'receipt.json')
    with pytest.raises(RuntimeError, match='already started'):
        assessment.evaluate_once()


@pytest.mark.integration
def test_saved_test_metrics_match_probabilities_and_frozen_threshold():
    receipt = assessment.verify_results()
    predictions = pd.read_parquet(assessment.OUT / 'predictions_test.parquet')
    metrics = pd.read_csv(assessment.OUT / 'test_metrics.csv')
    settings = json.loads((reduced.OUT / 'parameters.json').read_text())
    assert len(predictions) == receipt['rows'] == 160901
    y = predictions.target.to_numpy(dtype=int)
    assert y.sum() == receipt['positives'] == 631
    for row in metrics.itertuples():
        p = predictions[row.model].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        alarm = p >= row.threshold
        tp = int(np.sum(alarm & (y == 1)))
        fp = int(np.sum(alarm & (y == 0)))
        fn = int(np.sum(~alarm & (y == 1)))
        assert row.mse == pytest.approx(np.mean((p-y)**2), abs=1e-15)
        assert row.precision == pytest.approx(tp/(tp+fp) if tp+fp else 0)
        assert row.recall == pytest.approx(tp/(tp+fn))
        assert row.f1 == pytest.approx(2*tp/(2*tp+fp+fn))
        if row.model == reduced.MODEL_NAME:
            assert row.threshold == pytest.approx(settings['threshold'], abs=1e-15)
    intervals = pd.read_csv(assessment.OUT / 'paired_day_mse_intervals.csv')
    for row in intervals.itertuples():
        expected = np.mean((predictions[reduced.MODEL_NAME]-y)**2
                           - (predictions[row.comparator]-y)**2)
        assert row.delta_mse_new_minus_comparator == pytest.approx(expected, abs=1e-15)
        assert row.days == 15
