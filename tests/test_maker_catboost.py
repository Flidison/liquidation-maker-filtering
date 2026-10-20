"""E9 invariants: identical E3 inputs, causality, scoring, caches and preservation."""
import json

import numpy as np
import pandas as pd
import pytest

from maker_hybrid.core import ROOT, OUT, US, HORIZONS, VALIDATION, digest, save_json
from maker_hybrid.features import feature_columns, trade_features
from maker_hybrid.catboost_experiment import (E9_OUT, PARAMETERS, training_inputs, array_hash,
                                             checked_receipt, immutable_json)


def test_e9_weighted_grouped_squared_loss_has_only_constant_residual():
    rng = np.random.default_rng(91)
    group = rng.integers(0, 8, 200)
    w = np.minimum(rng.lognormal(9, 3, 200), 100000)
    y = rng.normal(size=200)*30 + .5
    W = np.bincount(group, weights=w)
    WY = np.bincount(group, weights=w*y)
    mean = WY/W
    residual = np.sum(w*(y-mean[group])**2)
    for pred in [rng.normal(size=8), np.zeros(8), np.full(8, 50.)]:
        raw = np.sum(w*(y-pred[group])**2)
        grouped = np.sum(W*(mean-pred)**2)
        assert raw-grouped == pytest.approx(residual, rel=1e-12)
        np.testing.assert_allclose(np.bincount(group, weights=2*w*(pred[group]-y)), 2*W*(pred-mean))


def test_e9_cache_requires_own_receipt_and_rejects_stale_or_corrupt(tmp_path):
    save_json(tmp_path/'regressors_complete.json', {'old':'E3 is complete'})
    assert checked_receipt(tmp_path/'E9.json', {'fingerprint':'new'}) is None
    model = tmp_path/'model.cbm'
    model.write_bytes(b'model bytes')
    receipt = tmp_path/'E9.json'
    save_json(receipt, {'artifact':model.name, 'sha256':digest(model), 'fingerprint':'new'})
    assert checked_receipt(receipt, {'fingerprint':'new'})
    with pytest.raises(RuntimeError, match='Stale'):
        checked_receipt(receipt, {'fingerprint':'different'})
    model.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Corrupt'):
        checked_receipt(receipt, {'fingerprint':'new'})


def test_versioned_specification_cannot_silently_overwrite(tmp_path):
    p = tmp_path/'spec.json'
    immutable_json(p, {'iterations':32})
    immutable_json(p, {'iterations':32})
    with pytest.raises(RuntimeError, match='choose a new'):
        immutable_json(p, {'iterations':64})
    assert json.loads(p.read_text()) == {'iterations':32}


@pytest.mark.integration
def test_e9_exact_e3_features_samples_weights_and_purge():
    manifest = json.loads((E9_OUT/'training_complete.json').read_text())
    old = json.loads((OUT/'regressors_complete.json').read_text())
    spec = json.loads((E9_OUT/'experiment_specification.json').read_text())
    assert spec['training_sample_sha256'] == old['training_sample_sha256'] == digest(OUT/'regression_training_sample.parquet')
    assert spec['features'] == feature_columns('E3')
    assert not spec['validation_labels_used_for_fit'] and not spec['hyperparameter_search']
    data = pd.read_parquet(OUT/'regression_training_sample.parquet')
    for m in manifest['models']:
        source = next(x for x in old['models'] if x['kind']=='E3' and x['tau']==m['tau'] and x['tag']==m['tag'])
        rows, x, y, w = training_inputs(data, spec['features'], m['boundary'], m['tau'])
        assert rows.sum() == m['rows'] == source['training_rows']
        assert array_hash(x.to_numpy()) == m['x_sha256']
        assert array_hash(y) == m['y_sha256']
        assert array_hash(w) == m['normalized_weight_sha256']
        assert m['last_label_available_before'] < m['boundary'] <= VALIDATION
        assert w.mean() == pytest.approx(1.)
        assert m['feature_names'] == spec['features']
        assert m['tree_count'] == PARAMETERS['iterations'] == 32
        assert m['requested_parameters'] == PARAMETERS
        assert m['reload_predictions_identical']
        assert digest(E9_OUT/'models'/m['artifact']) == m['sha256']


def test_e9_training_mask_does_not_consume_boundary_or_future_labels():
    f = pd.DataFrame({'second':[100, 101, 400, 1000], 'x':[1.,2.,3.,4.],
                      'target_30':[1.,99999.,99999.,99999.], 'weight_30':[1.,1.,1.,1.]})
    rows, x, y, w = training_inputs(f, ['x'], 402, 30)
    assert rows.tolist() == [True, False, False, False]
    assert y.tolist() == [1.] and x.x.tolist() == [1.] and w.tolist() == [1.]


def _synthetic():
    s = np.arange(1000, 1140)
    tr = pd.DataFrame({'timestamp':s*US+1000, 'ticker':'perp:btcusdt',
                       'side':np.where(s%2, 'buy', 'sell'), 'price':100+np.arange(len(s))*.01, 'amount':1.})
    b = pd.DataFrame({'timestamp':s*US-1, 'ticker':'perp:btcusdt', 'bid_price':tr.price-.01,
                      'ask_price':tr.price+.01, 'bid_amount':2., 'ask_amount':1.})
    l = pd.DataFrame({'timestamp':[1030*US,1110*US], 'side':['buy','sell'], 'price':[100.,100.], 'amount':[1.,1e9]})
    return tr, b, l


@pytest.mark.integration
def test_e9_native_models_predictions_are_causal_and_reload_exactly(tmp_path):
    from catboost import CatBoostRegressor
    from maker_hybrid.submission import market_batches, _empty_liq
    tr, b, l = _synthetic()
    a = list(market_batches(tr, b, l, _empty_liq(), 'btcusdt'))[0]
    z = list(market_batches(tr.iloc[:100], b.iloc[:100], l.iloc[:1], _empty_liq(), 'btcusdt'))[0]
    x = trade_features(a[2], a[4])[feature_columns('E3')].astype('float32')
    prefix = trade_features(z[2], z[4])[feature_columns('E3')].astype('float32')
    pd.testing.assert_frame_equal(x.iloc[:len(prefix)], prefix)
    for h in HORIZONS:
        m = CatBoostRegressor().load_model(str(E9_OUT/'models'/f'E9_{h}_final.cbm'))
        full = m.predict(x, thread_count=4)
        np.testing.assert_array_equal(full[:len(prefix)], m.predict(prefix, thread_count=4))
        m.save_model(str(tmp_path/f'{h}.cbm'))
        restored = CatBoostRegressor().load_model(str(tmp_path/f'{h}.cbm'))
        np.testing.assert_array_equal(full, restored.predict(x, thread_count=4))


@pytest.mark.integration
def test_e9_public_interface_prefix_order_missing_inputs_and_ties():
    from maker_hybrid.catboost_submission import catboost_signal
    from maker_hybrid.submission import _empty_liq
    tr, b, l = _synthetic()
    full = catboost_signal(tr, b, l, _empty_liq())
    short = catboost_signal(tr.iloc[:100], b.iloc[:100], l.iloc[:1], _empty_liq())
    reordered = catboost_signal(tr.iloc[::-1], b, l, _empty_liq())
    for h in HORIZONS:
        np.testing.assert_array_equal(full[h][:100], short[h])
        np.testing.assert_array_equal(full[h][::-1], reordered[h])
        assert full[h].dtype == np.uint8 and set(full[h]) <= {0,1}
    assert all(len(v)==0 for v in catboost_signal(tr.iloc[:0], b, l, l).values())
    assert all(np.all(v==0) for v in catboost_signal(tr, None, l, l).values())
    unknown = tr.copy(); unknown['ticker'] = 'unknown'
    assert all(np.all(v==0) for v in catboost_signal(unknown, b, l, l).values())
    # Empty liquidations are valid zero-flow features, not a missing-model error.
    assert all(len(v)==len(tr) for v in catboost_signal(tr, b, None, None).values())
    cfg = json.loads((E9_OUT/'catboost_config.json').read_text())
    for h in HORIZONS:cfg['models'][str(h)]['file'] = 'absent.cbm'
    with pytest.warns(RuntimeWarning, match='model unavailable'):
        fallback = catboost_signal(tr, b, l, l, config=cfg)
    assert all(np.all(v==0) for v in fallback.values())


def test_e9_evaluation_same_population_exact_mse_and_binary_budgets():
    policies = json.loads((E9_OUT/'budget_policies.json').read_text())
    assert len(policies) == 2*3*3*4
    assert all(-.001<=p['extra_turnover_over_budget']<=100000.001 for p in policies if p['feasible'])
    e = pd.read_csv(E9_OUT/'regression_error_diagnostics.csv').query("model == 'E3'")
    old = pd.read_csv(OUT/'regression_error_diagnostics.csv').query("model == 'E3'")
    keys = ['period','symbol','tau']
    np.testing.assert_allclose(e.sort_values(keys).weighted_trade_mse_bps2,
                               old.sort_values(keys).weighted_trade_mse_bps2, rtol=1e-12)
    paired = pd.read_csv(E9_OUT/'paired_e9_e3.csv')
    assert len(paired.query('primary_family_bonferroni and block_days == 5')) == 3
    np.testing.assert_allclose(paired.query('primary_family_bonferroni').confidence_level, 1-.05/3)
    # Native uint64 ties remain exact above float's integer precision.
    from maker_hybrid.policy import full_group_keep
    key = 2**63+101
    p = {'model':'E9','score_threshold':2.,'group_tie_cut':key,'trade_tie_cut':.5}
    keep, edge = full_group_keep([2,2,2,3], np.array([key-1,key,key+1,key+2],np.uint64), [0]*4, p)
    np.testing.assert_array_equal(keep,[1,0,0,1]); np.testing.assert_array_equal(edge,[0,1,0,0])


