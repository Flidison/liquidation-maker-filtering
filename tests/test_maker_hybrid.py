import numpy as np
import pandas as pd
import pytest
from maker_hybrid.core import *
from maker_hybrid.features import state_features


def liq(t=(),side=(),price=(),amount=()):
    return pd.DataFrame(dict(timestamp=np.asarray(t,dtype=np.int64),side=side,price=price,amount=amount))


def test_maker_side_clipping_and_rebate():
    pnl,age=maker_labels(np.array([0,0]),np.array([100.,100.]),np.array([1,-1]),
        np.array([0,30*US]),np.array([100.,101.]),30)
    np.testing.assert_allclose(pnl,[-99.5,100.5])
    w=np.minimum([1000000.,50000.],100000)
    r=official_score(pnl,w,[1,0],1)
    assert r['pnl_kept']==100.5
    assert r['pnl_filtered']==-99.5
    assert r['kept_turnover_per_day']==50000
    assert r['score']==pytest.approx(400/3)


def test_future_midpoint_exact_backward_asof_and_end_boundary():
    pnl,age=maker_labels(np.array([0,1,2])*US,np.full(3,100.),np.ones(3),
        np.array([0,30,31])*US,np.array([100.,101.,102.]),30)
    np.testing.assert_allclose(pnl[:2],[-99.5,-199.5])
    assert np.isnan(pnl[2])
    assert age[0]==0


def test_bybit_latency_equal_timestamp_and_expiring_window():
    a=liquidation_arrays(liq(),liq([US],['buy'],[100.],[1.]))
    t=np.array([1200000,1200001,6200000,6200001])
    np.testing.assert_array_equal(alignment(a,t,np.ones(4)),[0,1,1,0])


def test_full_trade_score_equals_grouped_sufficient_statistics():
    rng=np.random.default_rng(123)
    g=rng.integers(0,7,size=100)
    y=rng.normal(size=100)
    w=np.minimum(rng.lognormal(10,3,size=100),100000)
    keep_group=np.array([True,False,True,False,False,True,True])
    a=official_score(y,w,~keep_group[g],3)
    W=np.bincount(g,weights=w,minlength=7)
    WY=np.bincount(g,weights=w*y,minlength=7)
    b=WY[keep_group].sum()/W[keep_group].sum()-WY.sum()/W.sum()
    assert a['score']==pytest.approx(b)


def test_market_features_prefix_invariance_and_same_second_future_trade():
    seconds=np.arange(1000,1120)
    q=pd.DataFrame({'second':seconds,'quote_timestamp':seconds*US-1,
        'bid_price':100+np.arange(120)*.01,'ask_price':100.02+np.arange(120)*.01,
        'bid_amount':1.,'ask_amount':2.})
    flow=pd.DataFrame({'second':seconds,'taker_sign':1,'trade_count':2,'notional':100.})
    a=liquidation_arrays(liq([1020*US,1100*US],['buy','sell'],[100.,100.],[1.,10000.]),liq())
    full=state_features(seconds,q,flow,a,'btcusdt')
    short_a=liquidation_arrays(liq([1020*US],['buy'],[100.],[1.]),liq())
    prefix=state_features(seconds[:80],q.iloc[:80],flow.iloc[:80],short_a,'btcusdt')
    pd.testing.assert_frame_equal(full.iloc[:80].reset_index(drop=True),prefix)
    changed=flow.copy();changed.loc[79,'notional']=1e12
    revised=state_features(seconds,q,changed,a,'btcusdt')
    pd.testing.assert_frame_equal(full.iloc[:80],revised.iloc[:80])


def test_as_inventory_sign_and_units():
    r,b,a=as_quotes(100,2,.1,4,5,1.5)
    assert r==96
    assert (a+b)/2==r
    assert a>b
    assert as_quotes(100,-2,.1,4,5,1.5)[0]==104


def test_boundary_only_day_uses_float_accumulators():
    from maker_hybrid.evaluate import daily_stats
    f=pd.DataFrame({'second':[START],'w_30':[100.5],'wy_30':[201.], 'n_30':[1],
                    'fresh_w_30':[100.5],'fresh_wy_30':[201.]})
    p={k:v for k,v in zip(['model','scope','tau','budget_per_day','calibration_period'],['E7','combined',30,1,'test'])}
    d=daily_stats(f,30,np.array([],int),(0,f),START,1,p)
    assert d.kept_w.iloc[0]==100.5


def test_trade_hash_keeps_identical_occurrences_and_future_invariance():
    from maker_hybrid.policy import trade_hash
    f=pd.DataFrame({'timestamp':[1,1,2,3],'side':['buy']*4,'price':[100.]*4,'amount':[1.]*4})
    h=trade_hash(f,'btcusdt')
    assert h[0]!=h[1]
    np.testing.assert_array_equal(h[:3],trade_hash(f.iloc[:3],'btcusdt'))


def test_native_submission_order_sides_missing_liquidations():
    from maker_hybrid.submission import signal_with_config
    tr=pd.DataFrame({'timestamp':[2*US,2*US,3*US,3*US],
        'ticker':['perp:ethusdt','perp:btcusdt','perp:ethusdt','perp:btcusdt'],
        'side':['sell','buy','buy','sell'],'price':100.,'amount':1.})
    l=liq([US,US],['buy','sell'],[100.,100.],[1.,1.])
    l['ticker']=['perp:btcusdt','ETHUSDT']
    cfg={'policies':{str(h):{'combined':{'model':'E1_60','native':True}} for h in HORIZONS}}
    r=signal_with_config(tr,pd.DataFrame(),l,liq(),cfg)
    for a in r.values():
        np.testing.assert_array_equal(a,[0,0,1,1]);assert a.dtype==np.uint8
    z=signal_with_config(tr,pd.DataFrame(),liq(),liq(),cfg)
    assert all(np.all(a==1) for a in z.values())
    empty=signal_with_config(tr.iloc[:0],pd.DataFrame(),liq(),liq(),cfg)
    assert all(len(a)==0 for a in empty.values())


@pytest.mark.integration
def test_market_batch_future_invariance_actual_predictions():
    from maker_hybrid.submission import market_batches,_model
    from maker_hybrid.features import trade_features,feature_columns
    seconds=np.arange(1000,1140)
    tr=pd.DataFrame({'timestamp':seconds*US+1000,'side':np.where(seconds%2,'buy','sell'),
                     'price':100+np.arange(len(seconds))*.01,'amount':1.,'ticker':'perp:btcusdt'})
    b=pd.DataFrame({'timestamp':seconds*US-1,'bid_price':tr.price-.01,'ask_price':tr.price+.01,
                    'bid_amount':2.,'ask_amount':1.,'ticker':'perp:btcusdt'})
    l=liq([1030*US,1110*US],['buy','sell'],[100.,100.],[1.,10000.])
    full=list(market_batches(tr,b,l,liq(),'btcusdt'))[0]
    short=list(market_batches(tr.iloc[:100],b.iloc[:100],l.iloc[:1],liq(),'btcusdt'))[0]
    a,b=full[4],short[4]
    pd.testing.assert_frame_equal(a.iloc[:len(b)],b)
    cols=feature_columns('LR')
    for side in ['buy','sell']:
        m=_model(f'lr_{side}_final.joblib')
        np.testing.assert_array_equal(m.predict_proba(a[cols])[:len(b)],m.predict_proba(b[cols]))


@pytest.mark.integration
def test_chronological_artifact_provenance_and_schema():
    from maker_hybrid.features import feature_columns
    from maker_hybrid.logistic import future_burst
    np.testing.assert_array_equal(future_burst(np.array([10,11,15]),np.array([15])),[1,1,0])
    for p in (OUT/'models').glob('lr_*.json'):
        cfg=json.loads(p.read_text())
        assert cfg['last_training_second']+10 < cfg['boundary']
    m=json.loads((OUT/'regressors_complete.json').read_text())
    for cfg in m['models']:
        assert cfg['last_label_available_before']<cfg['boundary']<=VALIDATION
        assert digest(OUT/'models'/cfg['model_file'])==cfg['sha256']
    for kind in ['LR','E2','E3','E4','E5','E6','E7']:
        assert not any('target' in c or 'future' in c or 'pnl' in c for c in feature_columns(kind))
    assert not any(c.startswith('liq_') or c=='alignment' for c in feature_columns('E4'))


def test_missing_model_falls_back_without_fitting(monkeypatch):
    import maker_hybrid.submission as sub
    tr=pd.DataFrame({'timestamp':[2*US],'ticker':['perp:btcusdt'],'side':['buy'],'price':[100.],'amount':[1.]})
    b=pd.DataFrame({'timestamp':[US],'ticker':['perp:btcusdt'],'bid_price':[99.],'ask_price':[101.],
                    'bid_amount':[1.],'ask_amount':[1.]})
    cfg={'policies':{str(h):{'combined':{'model':'E7'}} for h in HORIZONS},'features':{}}
    def absent(name):raise FileNotFoundError(name)
    monkeypatch.setattr(sub,'_model',absent)
    with pytest.warns(RuntimeWarning):r=sub.signal_with_config(tr,b,liq(),liq(),cfg)
    assert all(np.array_equal(x,[0]) for x in r.values())


@pytest.mark.integration
def test_all_raw_trade_rows_are_preserved():
    totals={s:sum(json.loads(p.read_text())['raw_trade_rows'] for p in (OUT/'groups'/s).glob('*.json')) for s in SYMBOLS}
    assert totals=={'btcusdt':401902513,'ethusdt':705880385}


@pytest.mark.integration
def test_scaler_means_use_only_purged_training_rows():
    import joblib
    from maker_hybrid.features import feature_columns
    cols=feature_columns('LR')
    f=pd.read_parquet(OUT/'lr_training_sample.parquet',columns=['second',*cols])
    for tag in ['2025-12-22','final']:
        meta_path=OUT/'models'/f'lr_{tag}.json'
        if not meta_path.exists():
            tag='fold2025-12-22';meta_path=OUT/'models'/f'lr_{tag}.json'
        meta=json.loads(meta_path.read_text())
        past=f.loc[f.second+10<meta['boundary'],cols]
        model=joblib.load(OUT/'models'/f'lr_buy_{tag}.joblib')
        assert model['standardscaler'].n_samples_seen_==len(past)==meta['rows']
        np.testing.assert_allclose(model['standardscaler'].mean_,past.to_numpy(dtype=float).mean(axis=0),rtol=1e-12,atol=1e-12)


def test_matched_budgets_are_binary_with_one_trade_tolerance():
    policies=json.loads((OUT/'budget_policies.json').read_text())
    assert all(-.001<=p['extra_turnover_over_budget']<=100000.001 for p in policies if p['feasible'])
    from maker_hybrid.policy import full_group_keep
    p={'model':'E7','score_threshold':2.,'group_tie_cut':5,'trade_tie_cut':.1}
    keep,boundary=full_group_keep([3,2,2,1],[9,4,5,0],[1,0,-1,1],p)
    np.testing.assert_array_equal(keep,[1,1,0,0]);np.testing.assert_array_equal(boundary,[0,0,1,0])


def test_policy_keys_have_disjoint_symbol_and_time_namespaces():
    seconds=np.repeat(np.arange(10000,13000),6)
    sign=np.tile([-1,-1,-1,1,1,1],3000)
    align=np.tile([-1,0,1,-1,0,1],3000)
    values=np.r_[policy_hash(seconds,'btcusdt',sign,align),policy_hash(seconds,'ethusdt',sign,align)]
    assert values.dtype==np.uint64
    assert len(np.unique(values))==len(values)
    # Regression case: these distinct groups collided under the old additive salt.
    assert policy_hash([12000],'btcusdt',[-1],[-1])[0]!=policy_hash([11228],'ethusdt',[1],[0])[0]


def test_price_strength_rule_is_causal_and_deterministic():
    from maker_hybrid.submission import signal_with_config
    tr=pd.DataFrame({'timestamp':[2*US,2*US+1,70*US],'ticker':'perp:btcusdt',
                     'side':['buy','sell','buy'],'price':100.,'amount':1.})
    p={'model':'E1_60_strength','score_threshold':50.,'group_tie_cut':0,'trade_tie_cut':0}
    cfg={'policies':{str(h):{'combined':p} for h in HORIZONS}}
    past=liq([US],['buy'],[100.],[1.]);future=liq([US,100*US],['buy','sell'],[100.,100.],[1.,1e9])
    a=signal_with_config(tr,pd.DataFrame(),past,liq(),cfg)
    b=signal_with_config(tr,pd.DataFrame(),future,liq(),cfg)
    for h in HORIZONS:
        np.testing.assert_array_equal(a[h],[0,1,1]);np.testing.assert_array_equal(a[h],b[h])
