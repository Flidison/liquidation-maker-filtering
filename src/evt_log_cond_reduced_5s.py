"""Fixed smaller 5s LightGBM experiment, evaluated on development data only.

The historical test has already been inspected. This revision does not read its
features or make predictions on it, and does not overwrite the frozen experiment.
"""
from research_paths import resolve_legacy
from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import evt_log_cond_modeling as original

OUT = original.ROOT / 'artifacts/results/evt' / 'reduced_5s'
N_TREES = 5
N_LAGS = 5
MAX_DEPTH = 8
MODEL_NAME = 'lightgbm_reduced_5s'


def reduced_model(old_model):
    """A fresh estimator; retain all other parameters of the original model."""
    params = old_model.get_params(deep=False)
    params.update(n_estimators=N_TREES, max_depth=MAX_DEPTH)
    return lgb.LGBMClassifier(**params)


def development_only(frame):
    """Reject test rows before even applying the usable mask."""
    allowed = frame['split'].isin(['train', 'validation'])
    if not allowed.all():
        raise ValueError('Only train/validation may enter the reduced experiment')
    t = frame['timestamp'].to_numpy()
    tr = frame['split'].eq('train').to_numpy()
    valid_dates = ((t >= original.READY * 10**6)
                   & (t < original.TEST_START * 10**6)
                   & (tr == (t < original.VAL_START * 10**6)))
    if not valid_dates.all():
        raise ValueError('Rows do not respect the original UTC split boundaries')
    result = frame.loc[frame.usable].reset_index(drop=True)
    if set(result['split']) != {'train', 'validation'}:
        raise ValueError('Both development partitions are required')
    # Same population as the original comparison, including its 305s purge.
    for split, end in [('train', original.VAL_START), ('validation', original.TEST_START)]:
        times = result.loc[result['split'].eq(split), 'timestamp']
        if not ((times / 10**6 + original.PURGE_SECONDS) < end).all():
            raise ValueError('Target horizon crosses a partition boundary')
    return result


def tree_depth(node):
    if 'leaf_index' in node:
        return 0
    return 1 + max(tree_depth(node['left_child']), tree_depth(node['right_child']))


def verify_results():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['code_sha256'] != original.digest(Path(__file__)):
        raise RuntimeError('Reduced experiment code changed: saved results are stale')
    for path, expected in manifest['inputs'].items():
        if original.digest(original.resolve_legacy(path)) != expected:
            raise RuntimeError(f'Changed input: {path}')
    for path, expected in manifest['outputs'].items():
        if original.digest(OUT / path) != expected:
            raise RuntimeError(f'Changed output: {path}')
    return manifest


def run():
    if (OUT / 'manifest.json').exists():
        verify_results()
        return pd.read_csv(OUT / 'metrics.csv')
    source = original.OUT
    inputs = [source / 'features_development.parquet', source / 'frozen_selection.json',
              source / 'train_only_scaler.joblib', source / 'model_lightgbm_5s.joblib',
              source / 'model_logistic_5s.joblib', source / 'validation_metrics.csv',
              Path(original.__file__)]
    hashes = {str(p.relative_to(original.ROOT)): original.digest(p) for p in inputs}
    frozen = json.loads((source / 'frozen_selection.json').read_text())
    choices = {r['model']: r for r in frozen['models'] if r['horizon_s'] == 5}
    assert frozen['development_sha256'] == original.digest(inputs[0])
    assert frozen['code_sha256'] == original.digest(Path(original.__file__))
    assert frozen['scaler_sha256'] == original.digest(source / 'train_only_scaler.joblib')
    for choice in choices.values():
        assert choice['model_sha256'] == original.digest(source / choice['model_file'])

    data = development_only(pd.read_parquet(source / 'features_development.parquet'))
    train = data['split'].eq('train')
    validation = data['split'].eq('validation')
    y = data['target_burst_5s'].to_numpy(dtype=int)
    assert set(y[train]) == {0, 1} and set(y[validation]) == {0, 1}
    old_gb = joblib.load(source / choices['lightgbm']['model_file'])
    model = reduced_model(old_gb)
    features = original.feature_names(N_LAGS)
    x = data[features].astype('float64')
    print(f'Fit fixed 5s revision: {N_TREES} trees, depth {MAX_DEPTH}, {N_LAGS} lags;',
          f'train={train.sum()}, validation={validation.sum()}', flush=True)
    model.fit(x.loc[train], y[train])
    reduced_p = model.predict_proba(x)[:, 1]
    threshold = original.ew.choose_threshold(y[validation], reduced_p[validation])
    prediction_map = {MODEL_NAME: (reduced_p, threshold)}

    # Old estimators are only used for development diagnostics, never refitted.
    old_gb_p = old_gb.predict_proba(data[choices['lightgbm']['features']].astype('float64'))[:, 1]
    prediction_map['lightgbm_original'] = (old_gb_p, choices['lightgbm']['threshold'])
    scaler = joblib.load(source / 'train_only_scaler.joblib')
    lr = joblib.load(source / choices['logistic']['model_file'])
    scaled = scaler.transform(data[original.feature_names(10)].astype('float64'))
    lr_p = lr.predict_proba(scaled[:, :len(choices['logistic']['features'])])[:, 1]
    prediction_map['logistic_original'] = (lr_p, choices['logistic']['threshold'])
    prior = float(y[train].mean())
    prediction_map['constant_train_prevalence'] = (np.full(len(y), prior), .5)

    metric_rows = []
    predictions = data[['timestamp', 'symbol', 'split']].copy()
    predictions['target'] = y
    for name, (p, cutoff) in prediction_map.items():
        predictions[name] = p
        for split in ['train', 'validation']:
            mask = data['split'].eq(split)
            baseline_mse = float(np.mean((prior - y[mask])**2))
            scores = original.evaluate(y[mask], p[mask], cutoff)
            metric_rows.append({'model': name, 'split': split, 'horizon_s': 5,
                                **scores, 'baseline_mse': baseline_mse,
                                'brier_skill_vs_train_prior': 1 - scores['mse'] / baseline_mse})
    metrics = pd.DataFrame(metric_rows)
    saved = pd.read_csv(source / 'validation_metrics.csv')
    for name in ['lightgbm', 'logistic']:
        actual = metrics.loc[metrics.model.eq(name + '_original') & metrics.split.eq('validation'), 'mse'].item()
        expected = saved.loc[saved.model.eq(name) & saved.horizon_s.eq(5), 'mse'].item()
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-15)

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / 'model_lightgbm_reduced_5s.joblib')
    model.booster_.save_model(str(OUT / 'lightgbm_reduced_5s.txt'))
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    predictions.to_parquet(OUT / 'predictions_development.parquet', index=False)
    old_trees = old_gb.booster_.dump_model()['tree_info']
    new_trees = model.booster_.dump_model()['tree_info']
    settings = {
        'horizon_s': 5, 'n_lags': N_LAGS, 'features': features,
        'parameters': model.get_params(deep=False), 'threshold': float(threshold),
        'threshold_rule': 'maximum F1 on validation only; does not affect probability MSE',
        'hyperparameter_search': False, 'train_prevalence': prior,
        'actual_trees': model.booster_.num_trees(),
        'actual_max_depth': max(tree_depth(t['tree_structure']) for t in new_trees),
        'original': {'n_lags': choices['lightgbm']['n_lags'],
                     'n_estimators': old_gb.get_params()['n_estimators'],
                     'max_depth': old_gb.get_params()['max_depth'],
                     'actual_max_depth': max(tree_depth(t['tree_structure']) for t in old_trees)},
        'population': {split: int((data['split'] == split).sum()) for split in ['train', 'validation']},
        'interpretation': 'Development revision requested after inspection of historical test. '
                          'Validation reused; not an independent generalization estimate. '
                          'A new future period is required for independent testing.'}
    original.dump(OUT / 'parameters.json', settings)
    files = ['model_lightgbm_reduced_5s.joblib', 'lightgbm_reduced_5s.txt',
             'metrics.csv', 'predictions_development.parquet', 'parameters.json']
    original.dump(OUT / 'manifest.json', {
        'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'code_sha256': original.digest(Path(__file__)), 'inputs': hashes,
        'outputs': {name: original.digest(OUT / name) for name in files},
        'test_features_read': False, 'test_predictions_computed': False,
        'selection_after_historical_test_inspection': True})
    verify_results()
    return metrics


if __name__ == '__main__':
    print(run()[['model', 'split', 'mse', 'brier_skill_vs_train_prior',
                 'precision', 'recall', 'f1']].to_string(index=False))
