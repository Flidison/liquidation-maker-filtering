"""Tests of inference code; no test requires the market hypothesis to be true."""
import itertools
import numpy as np
import pandas as pd
import pytest
import liquidation_statistical_validation as sv


def test_holm_reference_values_and_order_invariance():
    np.testing.assert_allclose(sv.holm([.01,.04,.03]),[.03,.06,.06])
    p=np.array([.03,.001,.8,.02,1.,.0001])
    order=np.array([3,5,1,0,4,2])
    np.testing.assert_allclose(sv.holm(p)[order],sv.holm(p[order]))
    assert np.all(sv.holm(p)>=p) and np.all(sv.holm(p)<=1)


@pytest.mark.parametrize('invalid',[[np.nan,.05],[-.01,.1],[1.1,.1]])
def test_holm_rejects_missing_or_invalid_pvalues(invalid):
    with pytest.raises(ValueError):sv.holm(invalid)


@pytest.mark.parametrize('sums',[[1,2,3],[0,0],[1,-1,0],[3,-5,8,-2],[1,1,1,1]])
def test_signflip_matches_independent_exhaustive_enumeration(sums):
    a=np.array(sums,float)
    stats=np.array([abs(np.dot(sign,a)) for sign in itertools.product([-1,1],repeat=len(a))])
    expected=np.mean(stats>=abs(a.sum())-1e-10)
    actual,method,_,_=sv.signflip_p(a)
    assert actual==pytest.approx(expected)
    assert method=='exact'
    assert sv.signflip_p(-a)[0]==actual


def test_primary_family_is_fixed_complete_and_unique():
    f=pd.DataFrame(sv.fixed_family())
    assert len(f)==42 and not f.duplicated().any()
    assert f.groupby('metric').size().sort_values().tolist()==[6,12,12,12]


def test_cluster_inference_uses_episode_weight_and_omits_missing(monkeypatch):
    monkeypatch.setattr(sv,'BOOTSTRAP_DRAWS',1000)
    days=np.repeat(np.arange(5),[8,1,1,1,1])
    values=np.r_[np.zeros(8),1.,2.,3.,4.]
    result,agg=sv.cluster_inference(np.r_[values,np.nan],np.r_[days,2],np.arange(7),seed=10)
    assert result['effect_mean']==pytest.approx(values.mean())
    assert result['n_valid_episodes']==12 and result['n_valid_utc_days']==5
    assert result['n_calendar_days']==7 and agg.loc[6,'count']==0
    assert result['effect_mean']!=pytest.approx(2)  # unweighted mean of active-day means
    assert result['mean_ci95_low']<=result['effect_mean']<=result['mean_ci95_high']


@pytest.mark.integration
def test_saved_statistical_effects_and_adjustment_reconcile():
    saved=pd.read_csv(sv.OUT/'statistical_tests_validation.csv')
    outcomes=pd.read_parquet(sv.SOURCE/'all_outcomes.parquet').query("phase == 'validation' and anchor == 'episode_end'")
    catalog=pd.read_parquet(sv.SOURCE/'episode_catalog.parquet').query("phase == 'validation'")
    assert len(saved)==42
    np.testing.assert_allclose(saved.p_holm_42,sv.holm(saved.p_signflip_two_sided),atol=1e-12)
    for r in saved.itertuples():
        g=(catalog if r.horizon_s==0 else outcomes)
        g=g.loc[g.symbol.eq(r.symbol)&g.classification.eq(r.classification)]
        if r.horizon_s:g=g.loc[g.horizon_s.eq(r.horizon_s)]
        if r.metric==sv.METRICS[0]:v=g.post_signed_return_bps
        elif r.metric==sv.METRICS[1]:v=g.post_rv_bps-g.pre_anchor_rv_bps
        elif r.metric==sv.METRICS[2]:v=g.post_rv_bps-g.pre_onset_rv_bps
        else:v=g.during_rms_bps-g.before_same_duration_rms_bps
        assert r.n_valid_episodes==v.notna().sum()
        assert r.effect_mean==pytest.approx(v.mean(),abs=1e-10)
