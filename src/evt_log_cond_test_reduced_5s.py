"""One user-requested historical-test pass of the frozen reduced 5s model.

Never fits or tunes. Old-model probabilities are read from their saved test run.
A completion receipt makes reruns read-only; an interrupted pass cannot silently
predict a second time. The test period was previously inspected, so this is a
retrospective comparison, not a new independent holdout experiment.
"""
from research_paths import resolve_legacy
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import evt_log_cond_modeling as original
import evt_log_cond_reduced_5s as reduced

OUT = reduced.OUT / 'test_evaluation'
RECEIPT = OUT / 'receipt.json'
FREEZE = OUT / 'frozen_evaluation.json'


def paired_day_intervals(predictions, n_bootstrap=2000):
    """Resample paired whole UTC days, retaining both symbols in each block."""
    frame = predictions.copy()
    frame['day'] = pd.to_datetime(frame.timestamp, unit='us', utc=True).dt.strftime('%Y-%m-%d')
    new_error = (frame[reduced.MODEL_NAME] - frame.target)**2
    rows = []
    for comparator in ['lightgbm_original', 'logistic_original', 'constant_train_prevalence']:
        frame['delta'] = new_error - (frame[comparator] - frame.target)**2
        daily = frame.groupby('day').delta.agg(['sum', 'size'])
        rng = np.random.default_rng(original.SEED + 5)
        draws = rng.integers(0, len(daily), size=(n_bootstrap, len(daily)))
        numerator, denominator = daily['sum'].to_numpy(), daily['size'].to_numpy()
        samples = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
        rows.append({'comparator': comparator, 'days': len(daily),
                     'delta_mse_new_minus_comparator': float(numerator.sum() / denominator.sum()),
                     'ci_low': float(np.quantile(samples, .025)),
                     'ci_high': float(np.quantile(samples, .975)),
                     'bootstrap_replicates': n_bootstrap})
    return pd.DataFrame(rows)


def verify_results():
    receipt = json.loads(RECEIPT.read_text())
    assert receipt['test_prediction_passes_new_model'] == 1
    assert receipt['models_fitted'] == 0
    assert receipt['frozen_evaluation_sha256'] == original.digest(FREEZE)
    frozen = json.loads(FREEZE.read_text())
    assert frozen['code_sha256'] == original.digest(Path(__file__))
    for name, expected in frozen['inputs'].items():
        assert original.digest(original.resolve_legacy(name)) == expected, name
    for name, expected in receipt['outputs'].items():
        assert original.digest(OUT / name) == expected, name
    return receipt


def evaluate_once():
    if RECEIPT.exists():
        verify_results()
        return pd.read_csv(OUT / 'test_metrics.csv')
    if FREEZE.exists():
        raise RuntimeError('A test pass was already started; inspect saved outputs before recovery. '
                           'A second predict_proba call is intentionally blocked.')
    original.verify_results()
    reduced.verify_results()
    settings = json.loads((reduced.OUT / 'parameters.json').read_text())
    features = settings['features']
    assert features == original.feature_names(5)
    assert settings['parameters']['n_estimators'] == 5
    assert settings['parameters']['max_depth'] == 8
    inputs = [reduced.OUT / name for name in ['manifest.json', 'parameters.json',
              'model_lightgbm_reduced_5s.joblib', 'metrics.csv']]
    inputs += [original.OUT / name for name in ['features_test.parquet', 'predictions_test.parquet',
               'test_metrics.csv', 'test_evaluation_receipt.json']]
    inputs += [Path(original.__file__), Path(reduced.__file__)]
    input_hashes = {str(p.relative_to(original.ROOT)): original.digest(p) for p in inputs}

    data = pd.read_parquet(original.OUT / 'features_test.parquet')
    assert data['split'].eq('test').all()
    data = data.loc[data.usable].reset_index(drop=True)
    assert (data.timestamp >= original.TEST_START * 10**6).all()
    assert (data.timestamp / 10**6 + original.PURGE_SECONDS < original.END).all()
    assert not data.duplicated(['timestamp', 'symbol']).any()
    predictions = data[['timestamp', 'symbol']].copy()
    predictions['target'] = data['target_burst_5s'].astype(int)
    thresholds = {reduced.MODEL_NAME: settings['threshold'], 'constant_train_prevalence': .5}
    old = pd.read_parquet(original.OUT / 'predictions_test.parquet', filters=[('horizon_s', '=', 5)])
    for name in ['lightgbm', 'logistic']:
        cached = old.loc[old.model.eq(name)].reset_index(drop=True)
        pd.testing.assert_frame_equal(predictions[['timestamp', 'symbol']], cached[['timestamp', 'symbol']])
        np.testing.assert_array_equal(predictions.target, cached.target)
        predictions[name + '_original'] = cached.probability
        thresholds[name + '_original'] = float(cached.threshold.unique().item())
    predictions['constant_train_prevalence'] = settings['train_prevalence']

    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive creation claims the single evaluation before any model inference.
    frozen = {'frozen_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'code_sha256': original.digest(Path(__file__)), 'inputs': input_hashes,
              'parameters': settings['parameters'], 'features': features,
              'threshold': settings['threshold'], 'rows': len(data),
              'authorization': 'User requested one test evaluation of the fixed reduced model',
              'historical_test_previously_seen': True}
    with FREEZE.open('x') as stream:
        json.dump(frozen, stream, indent=2, ensure_ascii=False)
    model = joblib.load(reduced.OUT / 'model_lightgbm_reduced_5s.joblib')
    assert model.classes_.tolist() == [0, 1]
    assert model.feature_name_ == features
    print(f'One test predict_proba call: {len(data):,} observations; no fitting/tuning.', flush=True)
    predictions[reduced.MODEL_NAME] = model.predict_proba(data[features].astype('float64'))[:, 1]
    predictions.to_parquet(OUT / 'predictions_test.parquet', index=False)
    prior = settings['train_prevalence']
    baseline_mse = float(np.mean((predictions.target - prior)**2))
    rows = []
    for name in [reduced.MODEL_NAME, 'lightgbm_original', 'logistic_original', 'constant_train_prevalence']:
        scores = original.evaluate(predictions.target, predictions[name], thresholds[name])
        rows.append({'model': name, 'split': 'test', 'horizon_s': 5, **scores,
                     'baseline_mse': baseline_mse,
                     'brier_skill_vs_train_prior': 1 - scores['mse'] / baseline_mse})
    metrics = pd.DataFrame(rows)
    old_metrics = pd.read_csv(original.OUT / 'test_metrics.csv')
    for name in ['lightgbm', 'logistic', 'constant_train_prevalence']:
        renamed = name + '_original' if name != 'constant_train_prevalence' else name
        a = metrics.loc[metrics.model.eq(renamed)].iloc[0]
        b = old_metrics.loc[old_metrics.model.eq(name) & old_metrics.horizon_s.eq(5)].iloc[0]
        for metric in ['mse', 'precision', 'recall', 'f1', 'f1_macro']:
            assert np.isclose(a[metric], b[metric], rtol=1e-12, atol=1e-15)
    metrics.to_csv(OUT / 'test_metrics.csv', index=False)
    development = pd.read_csv(reduced.OUT / 'metrics.csv')
    pd.concat([development, metrics], ignore_index=True).to_csv(OUT / 'comparison_by_split.csv', index=False)
    paired_day_intervals(predictions).to_csv(OUT / 'paired_day_mse_intervals.csv', index=False)
    files = ['predictions_test.parquet', 'test_metrics.csv', 'comparison_by_split.csv',
             'paired_day_mse_intervals.csv']
    original.dump(RECEIPT, {
        'completed_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'frozen_evaluation_sha256': original.digest(FREEZE),
        'test_prediction_passes_new_model': 1, 'models_fitted': 0,
        'old_model_predictions_source': 'cached original test run, no new inference',
        'test_period_utc': '[2026-02-14, 2026-03-01)',
        'rows': len(data), 'positives': int(predictions.target.sum()),
        'outputs': {name: original.digest(OUT / name) for name in files},
        'historical_test_previously_seen': True,
        'bootstrap_note': '2000 paired UTC-day resamples of saved squared errors; '
                          'no additional prediction passes. Pointwise 95% intervals, 15 days.'})
    verify_results()
    return metrics


if __name__ == '__main__':
    print(evaluate_once()[['model', 'n', 'positives', 'mse', 'precision', 'recall', 'f1',
                          'brier_skill_vs_train_prior']].to_string(index=False))
    print(pd.read_csv(OUT / 'paired_day_mse_intervals.csv').to_string(index=False))
