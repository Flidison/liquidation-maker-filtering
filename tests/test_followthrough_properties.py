"""Independent synthetic oracles and algebraic properties, not snapshot tests."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import liquidation_followthrough as lf


def path(mid, ages=None, start=0):
    return lf.PricePath(pl.DataFrame({"second": np.arange(start, start+len(mid)),
        "mid_close": mid, "quote_age_upper_s": np.ones(len(mid), int) if ages is None else ages}))


@pytest.mark.parametrize("seed", [11, 19, 101])
def test_random_intervals_against_direct_second_returns(seed):
    rng=np.random.default_rng(seed)
    mid=100*np.exp(np.cumsum(rng.normal(0,.0003,300)))
    ages=np.ones(300,int)
    ages[[41,42,170]]=3
    mid[[22,90]]=np.nan
    p=path(mid,ages,start=2000)
    left=rng.integers(-2,303,200)
    right=left+rng.integers(0,30,200)
    actual=p.interval(left+2000,right+2000)
    for i,(a,b) in enumerate(zip(left,right)):
        valid=(1<=a<b<=300 and np.isfinite(mid[a-1:b]).all() and (ages[a-1:b]<=2).all())
        assert actual['valid'][i]==valid
        if not valid:
            assert np.isnan(actual['return_bps'][i]) and np.isnan(actual['rv_bps'][i])
        else:
            increments=np.log(mid[a:b]/mid[a-1:b-1])
            assert actual['return_bps'][i]==pytest.approx(np.log(mid[b-1]/mid[a-1])*1e4,abs=2e-8)
            assert actual['rv_bps'][i]==pytest.approx(np.linalg.norm(increments)*1e4,abs=2e-8)


@pytest.mark.parametrize("h", [1,5,20])
def test_constant_price_has_zero_return_and_rv(h):
    p=path(np.full(40,100.))
    v=p.interval([2],[2+h])
    assert v['valid'][0]
    assert v['return_bps'][0]==0 and v['rv_bps'][0]==0 and v['rms_bps'][0]==0


@pytest.mark.parametrize("h", [5,20])
def test_completed_second_boundary_excludes_the_next_second(h):
    mid=np.full(50,100.)
    mid[h+1:]=110.  # change becomes available at boundary h+2
    p=path(mid)
    assert p.interval([1],[h+1])['return_bps'][0]==0
    assert p.interval([1],[h+2])['return_bps'][0]==pytest.approx(np.log(1.1)*1e4)


def test_returns_add_and_variances_add_over_disjoint_intervals():
    mid=100*np.exp(np.cumsum(np.random.default_rng(71).normal(0,.001,100)))
    p=path(mid)
    first,second,whole=[p.interval([a],[b]) for a,b in [(3,17),(17,52),(3,52)]]
    assert whole['return_bps'][0]==pytest.approx(first['return_bps'][0]+second['return_bps'][0],abs=1e-9)
    assert whole['rv_bps'][0]**2==pytest.approx(first['rv_bps'][0]**2+second['rv_bps'][0]**2,abs=1e-8)
    assert abs(whole['return_bps'][0])<=np.sqrt(49)*whole['rv_bps'][0]+1e-8


def test_price_scale_invariance_and_reciprocal_sign():
    mid=100*np.exp(np.cumsum(np.random.default_rng(72).normal(0,.001,100)))
    a,b,c=[p.interval([5],[25]) for p in [path(mid),path(mid*1000),path(1/mid)]]
    np.testing.assert_allclose(a['rv_bps'],b['rv_bps'],atol=1e-8)
    np.testing.assert_allclose(a['rv_bps'],c['rv_bps'],atol=1e-8)
    np.testing.assert_allclose(a['return_bps'],-c['return_bps'],atol=1e-8)


def test_mutating_quotes_after_target_cannot_change_outcome():
    mid=100*np.exp(np.cumsum(np.random.default_rng(73).normal(0,.001,100)))
    changed=mid.copy();changed[25:]*=100
    a,b=[p.interval([5],[25]) for p in [path(mid),path(changed)]]
    for name in ('return_bps','rv_bps','rms_bps','valid'):
        np.testing.assert_array_equal(a[name],b[name])


@pytest.mark.parametrize("stale_at", [4,5,12,24])
def test_any_stale_boundary_invalidates_whole_measurement(stale_at):
    mid=np.full(40,100.)
    ages=np.ones(40,int);ages[stale_at]=3
    v=path(mid,ages).interval([5],[25])
    assert not v['valid'][0] and np.isnan(v['rv_bps'][0])


@pytest.mark.parametrize("phase,boundary", [('train',lf.lb.VALIDATION),('validation',lf.lb.END)])
@pytest.mark.parametrize("side,sign", [('buy',1),('sell',-1)])
def test_horizon_exactly_at_split_is_valid_one_second_later_is_not(phase,boundary,side,sign):
    start=boundary-60
    p=path(100*np.exp(np.arange(100)*.0001),start=start)
    c=pd.DataFrame({'episode_id':['e'], 'symbol':['btcusdt'],'exchange':['bybit'],'side':[side],
        'phase':[phase],'classification':[lf.CLASSES[2]],'onset_timestamp':[(boundary-30)*1_000_000],
        'first_detection_timestamp':[(boundary-25)*1_000_000],
        'first_cascade_detection_timestamp':pd.array([(boundary-20)*1_000_000],dtype='Int64'),
        'end_timestamp':[(boundary-20)*1_000_000], 'end_recognition_timestamp':[(boundary-15)*1_000_000]})
    c=lf.measure_catalog(c,p)
    o=lf.measure_outcomes(c,p,np.zeros(20,bool))
    e=o.query("anchor == 'episode_end' and horizon_s == 20").iloc[0]
    assert e.target_within_phase and e.post_signed_return_bps==pytest.approx(sign*20)
    assert set(o.query("anchor == 'onset'").class_known_at_anchor)=={'Not yet detected'}
    assert set(o.query("anchor == 'first_detection'").class_known_at_anchor)=={lf.CLASSES[1]}
    assert set(o.query("anchor == 'cascade_detection'").class_known_at_anchor)=={lf.CLASSES[2]}
    c.end_timestamp+=1_000_000
    later=lf.measure_outcomes(c,p,np.zeros(20,bool)).query("anchor == 'episode_end' and horizon_s == 20").iloc[0]
    assert not later.target_within_phase and np.isnan(later.post_return_bps)


def test_bootstrap_counts_days_and_preserves_episode_weight():
    # Day1 has9 events at0, day2 has1event at10: event mean1, unweighted day mean5.
    days=np.repeat(np.arange(5),[9,1,1,1,1])
    values=np.r_[np.zeros(9),10.,20.,30.,40.]
    frame=pd.DataFrame({'anchor_timestamp':days*86_400_000_000,'value':values})
    got=lf.cluster_mean_ci(frame,'value',seed=7,draws=1000)
    rng=np.random.default_rng(7);draws=rng.integers(0,5,(1000,5))
    expected=np.array([np.concatenate([values[days==d] for d in row]).mean() for row in draws])
    np.testing.assert_allclose(got[:2],np.quantile(expected,[.025,.975]))
    assert got[2]==5
    assert np.isnan(lf.cluster_mean_ci(frame[frame.anchor_timestamp<4*86_400_000_000],'value')[0])
