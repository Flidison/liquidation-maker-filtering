"""Isolated, immutable E9 experiment using the frozen E3 training sample.

Run: python -m maker_hybrid.catboost_experiment [--output PATH]
Never calls the old train/predict/evaluate entry points or changes their markers.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import ROOT, OUT, HORIZONS, SYMBOLS, SEED, VALIDATION, END, digest, save_json
from .features import feature_columns, trade_features
from .regression import POLICY_START

E9_OUT = ROOT / 'artifacts/results/catboost' / 'v1'
KEYS = ['second', 'taker_sign', 'alignment']
# Fixed before reading any E9 validation outcome. No eval_set or early stopping.
PARAMETERS = dict(loss_function='RMSE', iterations=32, depth=4, learning_rate=0.05,
                  l2_leaf_reg=3.0, random_seed=SEED, thread_count=4, task_type='CPU',
                  boosting_type='Plain', grow_policy='SymmetricTree', bootstrap_type='No',
                  random_strength=0.0, border_count=128, leaf_estimation_method='Newton',
                  leaf_estimation_iterations=1, boost_from_average=True,
                  use_best_model=False, allow_writing_files=False, verbose=False)


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


def immutable_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f'Artifact mismatch: {path}; choose a new --output version')
    else:
        save_json(path, obj)


def checked_receipt(path, expected):
    """Missing receipt means incomplete, never success because an old marker exists."""
    path = Path(path)
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    for k, v in expected.items():
        if r.get(k) != v:
            raise RuntimeError(f'Stale E9 cache ({k}): {path}; use a new output version')
    artifact = path.parent / r['artifact']
    if not artifact.exists() or digest(artifact) != r['sha256']:
        raise RuntimeError(f'Corrupt or missing E9 artifact: {artifact}')
    return r


def specification():
    old = json.loads((OUT / 'regressors_complete.json').read_text())
    sample_hash = digest(OUT / 'regression_training_sample.parquet')
    if sample_hash != old['training_sample_sha256']:
        raise RuntimeError('Frozen E3 training sample changed')
    cols = feature_columns('E3')
    e3 = [m for m in old['models'] if m['kind'] == 'E3']
    assert len(e3) == 6 and all(m['features'] == cols for m in e3)
    for m in e3:
        if digest(OUT / 'models' / m['model_file']) != m['sha256']:
            raise RuntimeError('Frozen E3 model changed')
    return dict(experiment='E9', estimator='CatBoostRegressor', parameters=PARAMETERS,
                catboost_version=importlib.metadata.version('catboost'),
                features=cols, training_sample_sha256=sample_hash,
                e3_manifest_sha256=digest(OUT / 'regressors_complete.json'),
                training_code_sha256=digest(Path(__file__)),
                feature_code_sha256=digest(Path(__file__).with_name('features.py')),
                source_models=e3, official_train=['2025-12-01', '2026-01-31'],
                effective_sample=['2025-12-22', '2026-01-31'],
                validation=['2026-02-01', '2026-02-28'],
                purge='second + 301 < boundary (identical to E3 for every horizon)',
                target='-taker_sign * (mid_asof(t+h)-price)/price * 10000 + 0.5',
                group_target='sum(w*y)/sum(w), w=min(price*amount,100000)',
                group_weight='sum(w)*16, normalized to mean 1 inside each fit',
                sample='Existing E3 deterministic 1/16 grouped sample; never resampled',
                validation_labels_used_for_fit=False, hyperparameter_search=False,
                primary_comparison='E9 minus E3; combined 250m/day; 30/120/300s; Bonferroni 3',
                evaluation='Full grouped trade population, exact boundary trade resolution')


def training_inputs(data, cols, boundary, h):
    rows = data.second + 301 < boundary
    x = data.loc[rows, cols].astype('float32')
    y = data.loc[rows, f'target_{h}'].to_numpy()
    w = data.loc[rows, f'weight_{h}'].to_numpy()
    w = w / w.mean()
    if not (np.isfinite(x).all().all() and np.isfinite(y).all() and
            np.isfinite(w).all() and (w > 0).all()):
        raise ValueError('Nonfinite E3 sample or nonpositive weights')
    return rows, x, y, w


def train(output=E9_OUT):
    from catboost import CatBoostRegressor
    output = Path(output)
    spec = specification()
    immutable_json(output / 'experiment_specification.json', spec)
    fingerprint = object_hash(spec)
    data = pd.read_parquet(OUT / 'regression_training_sample.parquet')
    models, importance = [], []
    for tag, boundary in [('holdout', POLICY_START), ('final', VALIDATION)]:
        for h in HORIZONS:
            rows, x, y, w = training_inputs(data, spec['features'], boundary, h)
            original = next(m for m in spec['source_models'] if m['tag'] == tag and m['tau'] == h)
            assert int(rows.sum()) == original['training_rows']
            assert int(data.loc[rows, 'second'].max()) == original['last_training_second']
            name = f'E9_{h}_{tag}.cbm'
            path = output / 'models' / name
            receipt_path = path.with_suffix('.json')
            expected = dict(experiment_fingerprint=fingerprint, tag=tag, tau=h,
                            rows=int(rows.sum()), boundary=boundary,
                            last_training_second=int(data.loc[rows, 'second'].max()),
                            last_label_available_before=int(data.loc[rows, 'second'].max()) + 301,
                            x_sha256=array_hash(x.to_numpy()), y_sha256=array_hash(y),
                            normalized_weight_sha256=array_hash(w))
            receipt = checked_receipt(receipt_path, expected)
            model = CatBoostRegressor()
            if receipt is None:
                if path.exists():
                    raise RuntimeError(f'Uncommitted model exists: {path}; choose a new version')
                model = CatBoostRegressor(**PARAMETERS)
                model.fit(x, y, sample_weight=w)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp.cbm')
                model.save_model(str(tmp))
                reloaded = CatBoostRegressor().load_model(str(tmp))
                probe = x.iloc[::max(1, len(x) // 1000)]
                np.testing.assert_array_equal(model.predict(probe, thread_count=4), reloaded.predict(probe, thread_count=4))
                tmp.rename(path)
                receipt = dict(**expected, artifact=name, sha256=digest(path),
                               resolved_parameters=model.get_all_params(),
                               requested_parameters=PARAMETERS, feature_names=model.feature_names_,
                               tree_count=model.tree_count_, reload_predictions_identical=True)
                immutable_json(receipt_path, receipt)
            else:
                model.load_model(str(path))
            models.append(receipt)
            if tag == 'final':
                importance += [dict(model='E9', tau=h, feature=c, prediction_values_change=float(v))
                               for c, v in zip(spec['features'], model.feature_importances_)]
            print(f'E9 {tag} {h}s: {rows.sum():,} groups, {len(x.columns)} features; saved', flush=True)
    # Only newly generated E9 metadata is written here.
    if not (output / 'feature_importance.csv').exists():
        pd.DataFrame(importance).to_csv(output / 'feature_importance.csv', index=False)
    immutable_json(output / 'training_complete.json', dict(experiment_fingerprint=fingerprint, models=models))
    return spec


def predict(output=E9_OUT):
    from catboost import CatBoostRegressor
    output = Path(output)
    spec = specification()
    immutable_json(output / 'experiment_specification.json', spec)
    fingerprint = object_hash(spec)
    completed = []
    for tag, start, end in [('holdout', POLICY_START, VALIDATION), ('final', VALIDATION, END)]:
        models = {}
        for h in HORIZONS:
            path = output / 'models' / f'E9_{h}_{tag}.cbm'
            if checked_receipt(path.with_suffix('.json'), {'experiment_fingerprint': fingerprint}) is None:
                raise RuntimeError('Train E9 first')
            models[h] = CatBoostRegressor().load_model(str(path))
        for symbol in SYMBOLS:
            for day in range(start, end, 86400):
                name = pd.Timestamp(day, unit='s', tz='UTC').strftime('%Y-%m-%d') + '.parquet'
                old_path = OUT / 'predictions' / tag / symbol / name
                group_path = OUT / 'groups' / symbol / name
                state_path = OUT / 'states' / symbol / name
                path = output / 'predictions' / tag / symbol / name
                receipt_path = path.with_suffix('.json')
                expected = dict(experiment_fingerprint=fingerprint, e3_prediction_sha256=digest(old_path),
                                groups_sha256=digest(group_path), states_sha256=digest(state_path),
                                model_hashes={str(h):digest(output/'models'/f'E9_{h}_{tag}.cbm') for h in HORIZONS})
                r = checked_receipt(receipt_path, expected)
                if r is None:
                    if path.exists():
                        raise RuntimeError(f'Uncommitted prediction cache: {path}')
                    g = pd.read_parquet(group_path, columns=KEYS)
                    old = pd.read_parquet(old_path, columns=KEYS)
                    pd.testing.assert_frame_equal(g, old)
                    x = trade_features(g, pd.read_parquet(state_path))[spec['features']].astype('float32')
                    p = g.copy()
                    for h in HORIZONS:
                        p[f'E9_{h}'] = models[h].predict(x, thread_count=4).astype('float32')
                    assert np.isfinite(p[[f'E9_{h}' for h in HORIZONS]]).all().all()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.tmp.parquet')
                    p.to_parquet(tmp, index=False)
                    tmp.rename(path)
                    r = dict(**expected, artifact=name, sha256=digest(path), rows=len(p))
                    immutable_json(receipt_path, r)
                completed.append(dict(path=str(path.relative_to(output)), sha256=r['sha256'], rows=r['rows']))
                print(f'E9 predictions {tag} {symbol} {name}', flush=True)
    immutable_json(output/'predictions_complete.json', dict(experiment_fingerprint=fingerprint, files=completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=E9_OUT)
    args = parser.parse_args()
    train(args.output)
    predict(args.output)


if __name__ == '__main__':
    main()
