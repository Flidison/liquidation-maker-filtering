"""Independent invariants for event-time features, labels and split isolation."""
import numpy as np
import pandas as pd
import pytest
import liquidation_early_warning as ew


def dc(p, valid=None, threshold=(.01,.02,.04)):
    p=np.asarray(p,float)
    return ew.directional_changes(p, np.ones(len(p),bool) if valid is None else valid,
                                  np.array(threshold))


def test_running_extreme_not_previous_tick():
    state,segment,event,progress,overshoot=dc([0,.05,.06,.055,.049,.039,.019])
    np.testing.assert_array_equal(state,[-1,7,7,7,6,4,0])
    np.testing.assert_array_equal(np.flatnonzero(event),[4,5,6])
    assert progress[3,0]==pytest.approx(.5)
    assert overshoot[2,0]==pytest.approx(1.)


def test_atomic_simultaneous_threshold_crossing():
    state,_,event,_,_=dc([0,.05,-.01])
    assert list(state)==[-1,7,0]
    assert list(event)==[False,False,True]


def test_gap_resets_extremes_and_transition_chain():
    state,segment,event,_,_=dc([0,.05,.01,99,1.,1.05,1.],
                              np.array([1,1,1,0,1,1,1],bool))
    assert state[3]==state[4]==-1
    assert not event[5]
    assert event[6]
    assert segment[6]>segment[2]


@pytest.mark.parametrize('seed',[5,17,91])
def test_future_prices_do_not_change_existing_features(seed):
    rng=np.random.default_rng(seed)
    p=rng.normal(0,.007,500).cumsum()
    first=dc(p[:250]); full=dc(p)
    altered=p.copy();altered[250:]+=50
    changed=dc(altered)
    for a,b,c in zip(first,full,changed):
        np.testing.assert_allclose(a,b[:250],equal_nan=True)
        np.testing.assert_allclose(a,c[:250],equal_nan=True)


def test_constant_price_emits_no_initialized_state_or_event():
    state,_,event,_,_=dc(np.ones(100))
    assert (state==-1).all() and not event.any()


def test_monotone_rise_has_no_fabricated_reversal():
    state,_,event,_,_=dc(np.arange(100)*.003)
    assert state[-1]==7 and not event.any()


def test_transition_probabilities_normalized_and_nonself():
    prev=np.tile([0,1,3,7],20);cur=np.tile([1,3,7,0],20)
    _,matrix,counts=ew.fit_transition_matrix(prev,cur)
    np.testing.assert_allclose(matrix.sum(axis=1),1.)
    assert np.all(np.diag(matrix)==0)
    assert np.all(matrix[~np.eye(8,dtype=bool)]>0)
    assert counts.sum()==80
    assert matrix[0,1]>matrix[0,2]


def test_surprise_mean_log_probability_and_gap():
    p=np.array([.5,.25,.125,.5,.25])
    w=ew.rolling_surprise(p,np.array([0,0,0,1,1]),k=3)
    assert np.isnan(w[:2]).all() and np.isnan(w[3:]).all()
    assert w[2]==pytest.approx(-np.log(p[:3]).mean())


def test_low_liquidity_score_means_rare_not_likely_path():
    w=np.array([-2.,0.,2.,np.nan])
    gaussian,empirical,_=ew.liquidity_scores(w,np.linspace(-3,3,200))
    assert np.all(np.diff(gaussian[:3])<0)
    assert np.all(np.diff(empirical[:3])<0)
    assert np.isnan(gaussian[-1]) and np.isnan(empirical[-1])


def test_trailing_sum_excludes_unfinished_second():
    v=np.array([1.,2.,4.,8.,16.])
    np.testing.assert_allclose(ew.trailing_sum(v,np.array([1,2,4,5]),2),[np.nan,3,12,24],equal_nan=True)


@pytest.mark.parametrize('lag',ew.LAGS)
def test_return_rv_against_direct_loops(lag):
    rng=np.random.default_rng(19)
    mid=np.exp(rng.normal(0,.001,300).cumsum())*100
    fresh=np.ones(300,bool);fresh[170]=False
    q=np.array([60,90,175,250])
    ret,rv=ew.trailing_price_features(mid,fresh,q,lag)
    for k,t in enumerate(q):
        if t<=lag or not fresh[t-lag-1:t].all():
            assert np.isnan(ret[k]) and np.isnan(rv[k])
        else:
            r=np.diff(np.log(mid[t-lag-1:t]))*10000
            assert ret[k]==pytest.approx(r.sum(),abs=1e-9)
            assert rv[k]==pytest.approx(np.sqrt(np.sum(r*r)),abs=1e-9)


def test_future_labels_strict_now_inclusive_horizon_and_unavailable():
    result,nxt=ew.future_labels(np.array([95,100,105,116]),np.array([100,120]),
                                horizons=(5,20),ends=np.array([200,200,200,125]))
    np.testing.assert_array_equal(nxt,[100,120,120,120])
    np.testing.assert_array_equal(result['target_burst_5s'],[1,0,0,1])
    np.testing.assert_array_equal(result['target_burst_20s'],[1,1,1,-1])


def test_boundary_purge_includes_detection_delay():
    end=ew.TEST_START
    t=np.array([end-26,end-25,end-20])
    result,_=ew.future_labels(t,np.array([end-10]),horizons=(20,),ends=np.full(3,end-5))
    np.testing.assert_array_equal(result['target_burst_20s'],[1,-1,-1])


def test_whole_utc_days_sixty_fifteen_fifteen():
    days=np.arange(ew.START,ew.END,86400)
    split=ew.partition(days)
    assert len(days)==90
    assert [(split==x).sum() for x in ['train','validation','test']]==[60,15,15]
    np.testing.assert_array_equal(ew.partition(days+86399),split)


def test_feature_allowlist_prevents_label_and_episode_hindsight():
    f=pd.DataFrame(columns=['timestamp','next_burst_onset_timestamp','classification',
        'target_burst_5s','first_detection_timestamp','end_timestamp','open_interest_now',
        'forecast_eligible','history_ready','burst_known_now','symbol_eth','W_20',
        'liquidity_L','rv_5s_bps','binance_buy_liq_count_5s','dc_up_2p5bps'])
    assert set(ew.feature_columns(f))=={'symbol_eth','W_20','liquidity_L','rv_5s_bps',
                                      'binance_buy_liq_count_5s','dc_up_2p5bps'}
    assert 'W_20' not in ew.feature_columns(f,False)


def test_episode_coverage_counts_no_opportunity_and_requires_strict_lead():
    f=pd.DataFrame({'timestamp':np.array([94,99,101])*1_000_000,'symbol':['btcusdt']*3})
    episodes=pd.DataFrame({'symbol':['btcusdt']*2,'onset_timestamp':np.array([100,200])*1_000_000,
                          'first_detection_timestamp':np.array([105,205])*1_000_000,
                          'has_potential_cascade':[True,False]})
    e=ew.event_metrics(f,np.array([.9,.1,.9]),.5,episodes,10)
    assert e.warned_before_onset.tolist()==[True,False]
    assert e.opportunity.tolist()==[True,False]
    assert e.lead_to_onset_s.iloc[0]==6


@pytest.mark.parametrize('seed',[1,9,21])
def test_bootstrap_ap_with_ties_matches_independent_replication(seed):
    from liquidation_early_warning_evaluation import ap_cache,weighted_ap
    from sklearn.metrics import average_precision_score
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,100);p=rng.choice([.05,.2,.8],100)
    days=rng.integers(0,5,100);weights=np.array([2,0,1,1,3])
    idx=np.repeat(np.arange(100),weights[days])
    expected=average_precision_score(y[idx],p[idx])
    assert weighted_ap(ap_cache(y,p,days),weights)==pytest.approx(expected)
