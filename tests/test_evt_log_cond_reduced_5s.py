"""Check development isolation and saved smaller-model predictions."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import evt_log_cond_modeling as original
import evt_log_cond_reduced_5s as reduced


def example_frame():
    return pd.DataFrame({
        'timestamp': [original.READY * 10**6, original.VAL_START * 10**6],
        'split': ['train', 'validation'], 'usable': [True, True]})


def test_test_partition_rejected_even_if_marked_unusable():
    frame = example_frame()
    frame.loc[2] = [original.TEST_START * 10**6, 'test', False]
    with pytest.raises(ValueError, match='Only train/validation'):
        reduced.development_only(frame)


def test_test_timestamp_cannot_be_disguised_as_validation():
    frame = example_frame()
    frame.loc[1, 'timestamp'] = original.TEST_START * 10**6
    with pytest.raises(ValueError, match='UTC split boundaries'):
        reduced.development_only(frame)


def test_future_target_cannot_cross_split_boundary():
    frame = example_frame()
    frame.loc[0, 'timestamp'] = (original.VAL_START - original.PURGE_SECONDS) * 10**6
    with pytest.raises(ValueError, match='horizon crosses'):
        reduced.development_only(frame)


@pytest.mark.integration
def test_saved_probabilities_parameters_and_metrics():
    manifest = reduced.verify_results()
    assert not manifest['test_features_read']
    assert not manifest['test_predictions_computed']
    settings = json.loads((reduced.OUT / 'parameters.json').read_text())
    model = joblib.load(reduced.OUT / 'model_lightgbm_reduced_5s.joblib')
    assert model.booster_.num_trees() == settings['actual_trees'] == 5
    assert model.get_params()['max_depth'] == 8
    assert settings['n_lags'] == 5
    assert model.feature_name_ == original.feature_names(5)
    assert settings['actual_max_depth'] <= 8
    assert not settings['hyperparameter_search']
    assert model.classes_.tolist() == [0, 1]

    data = reduced.development_only(pd.read_parquet(original.OUT / 'features_development.parquet'))
    predictions = pd.read_parquet(reduced.OUT / 'predictions_development.parquet')
    pd.testing.assert_frame_equal(data[['timestamp', 'symbol', 'split']],
                                  predictions[['timestamp', 'symbol', 'split']])
    assert set(predictions['split']) == {'train', 'validation'}
    # Recompute a development sample, not any test observations.
    sample = np.linspace(0, len(data) - 1, 1000, dtype=int)
    actual = model.predict_proba(data.iloc[sample][model.feature_name_].astype('float64'))[:, 1]
    np.testing.assert_allclose(actual, predictions.iloc[sample][reduced.MODEL_NAME], atol=1e-15)
    metrics = pd.read_csv(reduced.OUT / 'metrics.csv')
    thresholds = metrics.groupby('model').threshold.first().to_dict()
    for row in metrics.itertuples():
        part = predictions.loc[predictions['split'].eq(row.split)]
        p = part[row.model].to_numpy()
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        expected = original.evaluate(part.target, p, thresholds[row.model])
        for metric in ['mse', 'precision', 'recall', 'f1', 'f1_macro']:
            assert getattr(row, metric) == pytest.approx(expected[metric], abs=1e-14)
    v = predictions.loc[predictions['split'].eq('validation')]
    expected_threshold = original.ew.choose_threshold(v.target, v[reduced.MODEL_NAME])
    assert settings['threshold'] == pytest.approx(expected_threshold)


@pytest.mark.integration
def test_frozen_original_experiment_remains_valid():
    assert original.verify_results()['test_evaluation_passes'] == 1


@pytest.mark.integration
def test_reopening_completed_revision_does_not_fit_or_predict(monkeypatch):
    # Any joblib model load would mean we entered the training/prediction path.
    def forbid(*args, **kwargs):
        raise AssertionError('Completed revision must return cached results')
    monkeypatch.setattr(joblib, 'load', forbid)
    result = reduced.run()
    assert set(result['split']) == {'train', 'validation'}
