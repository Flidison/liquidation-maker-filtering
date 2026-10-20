"""Evaluate E9 on the E3 population without changing any legacy result."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import OUT, HORIZONS, SYMBOLS, SEED, VALIDATION, END, digest
from .catboost_experiment import E9_OUT, KEYS, immutable_json, checked_receipt, object_hash
from .regression import POLICY_START
from .evaluate import BUDGETS, daily_stats, summarize_daily, load_period
from .policy import calibrate, ranked_groups, full_group_keep


def load_e9_period(output, tag, h):
    f = load_period(tag, h)
    parts = []
    expected_days = 10 if tag == 'holdout' else 28
    for symbol in SYMBOLS:
        paths = sorted((Path(output) / 'predictions' / tag / symbol).glob('*.parquet'))
        if len(paths) != expected_days:
            raise RuntimeError(f'Incomplete E9 {tag} predictions for {symbol}')
        for path in paths:
            p = pd.read_parquet(path, columns=KEYS + [f'E9_{h}'])
            p['symbol'] = symbol
            parts.append(p)
    p = pd.concat(parts, ignore_index=True)
    if tag == 'holdout':
        p = p[p.second + 301 < VALIDATION].reset_index(drop=True)
    pd.testing.assert_frame_equal(f[KEYS + ['symbol']], p[KEYS + ['symbol']])
    f[f'E9_{h}'] = p[f'E9_{h}']
    return f


def paired_e9_e3(daily, draws_count=5000):
    """Paired circular day-block bootstrap; all BTC/ETH shocks kept together.

    Primary family was fixed in the specification: three horizons, combined
    250m/day. Other cells are descriptive, not a second model-selection search.
    """
    rows = []
    d = daily[daily.calibration_period.eq('validation')]
    for (scope, h, budget), f in d.groupby(['scope', 'tau', 'budget_per_day']):
        a = f[f.model.eq('E9')].sort_values('day')
        b = f[f.model.eq('E3')].sort_values('day')
        if len(a) == 0 or len(b) == 0:
            continue
        assert a.day.tolist() == b.day.tolist()
        np.testing.assert_allclose(a.all_w, b.all_w, rtol=1e-12)
        np.testing.assert_allclose(a.all_wy, b.all_wy, rtol=1e-12, atol=1e-3)
        n = len(a)
        observed = a.kept_wy.sum()/a.kept_w.sum() - b.kept_wy.sum()/b.kept_w.sum()
        primary = scope == 'combined' and budget == 250000000
        alpha = .05/3 if primary else .05
        for block in [1, 5]:
            rng = np.random.default_rng(SEED)
            starts = rng.integers(0, n, size=(draws_count, int(np.ceil(n/block))))
            ix = ((starts[..., None] + np.arange(block)) % n).reshape(draws_count, -1)[:, :n]
            def weighted_mean(z):
                w = z.kept_w.to_numpy()[ix].sum(axis=1)
                wy = z.kept_wy.to_numpy()[ix].sum(axis=1)
                return np.divide(wy, w, out=np.full(draws_count, np.nan), where=w>0)
            delta = weighted_mean(a) - weighted_mean(b)
            finite = delta[np.isfinite(delta)]
            q = np.quantile(finite, [alpha/2, 1-alpha/2]) if len(finite) else [np.nan, np.nan]
            complete = len(finite) == draws_count
            rows.append(dict(model='E9', comparator='E3', scope=scope, tau=h, budget_per_day=budget,
                             delta_score=observed, block_days=block, ci_low=q[0] if complete else np.nan,
                             ci_high=q[1] if complete else np.nan, conditional_ci_low=q[0], conditional_ci_high=q[1],
                             confidence_level=1-alpha, primary_family_bonferroni=primary,
                             finite_bootstrap_fraction=len(finite)/draws_count,
                             interpretation='reused validation; descriptive uncertainty, not independent confirmation'))
    return pd.DataFrame(rows)


def run(output=E9_OUT):
    output = Path(output)
    spec = json.loads((output/'experiment_specification.json').read_text())
    pred_manifest = json.loads((output/'predictions_complete.json').read_text())
    for entry in pred_manifest['files']:
        if digest(output/entry['path']) != entry['sha256']:
            raise RuntimeError('E9 prediction cache changed')
    fingerprint = object_hash(dict(predictions=digest(output/'predictions_complete.json'),
                                  old_daily=digest(OUT/'policy_daily.csv'),
                                  evaluation_code=digest(Path(__file__))))
    marker = output/'evaluation_complete.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior['evaluation_fingerprint'] != fingerprint:
            raise RuntimeError('Evaluation inputs changed; choose a new E9 version')
        for path, sha in prior['outputs'].items():
            if digest(output/path) != sha:
                raise RuntimeError(f'E9 report changed: {path}')
        return
    policies, all_daily, errors, transfer = [], [], [], []
    for tag, start, end in [('holdout', POLICY_START, VALIDATION), ('final', VALIDATION, END)]:
        period = 'train_holdout' if tag == 'holdout' else 'validation'
        days = (end-start)//86400
        for h in HORIZONS:
            chunk = output/'evaluation_chunks'/f'{tag}_{h}.json'
            cached = checked_receipt(chunk, {'evaluation_fingerprint':fingerprint})
            if cached is not None:
                payload = json.loads((chunk.parent/cached['artifact']).read_text())
            else:
                f_all = load_e9_period(output, tag, h)
                cp, cd, ce, ct = [], [], [], []
                for scope in [*SYMBOLS, 'combined']:
                    f = (f_all if scope == 'combined' else f_all[f_all.symbol.eq(scope)]).reset_index(drop=True)
                    rank = ranked_groups(f, 'E9', h)
                    for budget in BUDGETS:
                        policy, whole, partial = calibrate(f, 'E9', h, budget, days, rank)
                        policy.update(scope=scope, calibration_period=period)
                        cp.append(policy)
                        if policy['feasible']:
                            cd += daily_stats(f, h, whole, partial, start, days, policy).to_dict('records')
                        if tag == 'final' and scope == 'combined' and budget == 250000000:
                            # January threshold applied unchanged to February final-model scores.
                            frozen = next(p for p in policies if p['scope']==scope and p['tau']==h and
                                          p['budget_per_day']==budget and p['calibration_period']=='train_holdout')
                            kept, boundary = full_group_keep(f[f'E9_{h}'], f.tie_key, f.alignment, frozen)
                            # Unique uint64 group identities include time: January boundary cannot recur.
                            assert not boundary.any()
                            cfg = dict(frozen, calibration_period='january_threshold_on_february')
                            ct += daily_stats(f, h, np.flatnonzero(kept), None, start, days, cfg).to_dict('records')
                    for model in ['E9', 'E3']:
                        p = f[f'{model}_{h}'].to_numpy(float)
                        numerator = (f[f'wy2_{h}']-2*p*f[f'wy_{h}']+p*p*f[f'w_{h}']).sum()
                        ce.append(dict(period=period, model=model, symbol=scope, tau=h,
                                       weighted_trade_mse_bps2=float(numerator/f[f'w_{h}'].sum())))
                    print(f'E9 evaluated {period} {scope} {h}s', flush=True)
                payload = dict(policies=cp, daily=cd, errors=ce, transfer=ct)
                path = chunk.with_name(chunk.stem+'_data.json')
                immutable_json(path, payload)
                immutable_json(chunk, dict(evaluation_fingerprint=fingerprint, artifact=path.name, sha256=digest(path)))
                del f_all, f
                gc.collect()
            policies += payload['policies']
            all_daily += payload['daily']
            errors += payload['errors']
            transfer += payload['transfer']
    daily = pd.DataFrame(all_daily)
    report = summarize_daily(daily)
    old = pd.read_csv(OUT/'policy_daily.csv')
    matched_daily = pd.concat([old[old.model.eq('E3')], daily], ignore_index=True)
    paired = paired_e9_e3(matched_daily)
    original = pd.read_csv(OUT/'all_validation_results.csv')
    combined_report = pd.concat([original, report], ignore_index=True)
    files = dict(policy_daily=daily, validation_report=report, comparison_report=combined_report,
                 paired_e9_e3=paired, regression_error_diagnostics=pd.DataFrame(errors),
                 threshold_transfer_daily=pd.DataFrame(transfer),
                 threshold_transfer_report=summarize_daily(pd.DataFrame(transfer)))
    for name, table in files.items():
        path = output/f'{name}.csv'
        if path.exists():
            raise RuntimeError(f'Uncommitted report exists: {path}; choose a new output version')
        table.to_csv(path, index=False)
    immutable_json(output/'budget_policies.json', policies)
    config = dict(experiment='E9', horizons=list(HORIZONS), features=spec['features'],
                  experiment_fingerprint=object_hash(spec),
                  threshold_role='February scores/weights calibrated, no February markout tuning; fixed thereafter',
                  policies={str(h):{'combined':next(p for p in policies if p['tau']==h and p['scope']=='combined'
                            and p['calibration_period']=='validation' and p['budget_per_day']==250000000)} for h in HORIZONS},
                  models={str(h):dict(file=f'models/E9_{h}_final.cbm', sha256=digest(output/'models'/f'E9_{h}_final.cbm')) for h in HORIZONS})
    immutable_json(output/'catboost_config.json', config)
    outputs = [f'{name}.csv' for name in files] + ['catboost_config.json', 'budget_policies.json']
    immutable_json(marker, dict(evaluation_fingerprint=fingerprint,
                               outputs={p:digest(output/p) for p in outputs},
                               scores_use_all_trades=True, hidden_test_available=False,
                               model_fitting_uses_february=False,
                               policy_calibration='scores and weights only; no markout values',
                               grouped_weighted_loss='sum_i w_i(y_i-p)^2 = W(ybar-p)^2 + constant',
                               benchmark_originals_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=E9_OUT)
    run(parser.parse_args().output)
