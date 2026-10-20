"""Integration audit using direct interval loops and independent raw BBO scans."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
import pytest
import liquidation_followthrough as lf

REPORT=lf.ROOT/'artifacts/results/statistics'
pytestmark=pytest.mark.integration


@pytest.fixture(scope='module')
def outcomes():
    return pd.read_parquet(lf.OUT/'all_outcomes.parquet')


@pytest.fixture(scope='module')
def catalog():
    return pd.read_parquet(lf.OUT/'episode_catalog.parquet')


def test_source_and_artifact_lineage():
    lf.verify_saved()


def test_episode_completeness_and_one_to_many_contract(catalog,outcomes):
    original=pd.read_parquet(lf.lb.OUT/'episodes.parquet')
    flagged=catalog.query("classification != 'Normal activity'")
    assert catalog.episode_id.is_unique
    assert set(flagged.episode_id)==set(original.episode_id)
    assert not outcomes.duplicated(['episode_id','anchor','horizon_s']).any()
    counts=outcomes.groupby('episode_id').size().reindex(catalog.episode_id).to_numpy()
    np.testing.assert_array_equal(counts,np.where(catalog.classification.eq(lf.CLASSES[2]),8,6))
    assert set(outcomes.episode_id)==set(catalog.episode_id)
    assert (catalog.first_detection_timestamp==catalog.onset_timestamp+5_000_000).all()
    assert (catalog.end_timestamp>=catalog.first_detection_timestamp).all()
    assert (catalog.duration_s*1e6==catalog.end_timestamp-catalog.onset_timestamp).all()
    assert outcomes.target_timestamp.sub(outcomes.anchor_timestamp).eq(outcomes.horizon_s*1_000_000).all()


def test_all_signed_returns_directions_and_future_phase_masks(outcomes):
    v=outcomes
    np.testing.assert_allclose(v.post_signed_return_bps,v.post_return_bps*np.where(v.side.eq('buy'),1,-1),equal_nan=True)
    assert v.loc[v.post_return_bps>1e-10,'post_price_direction'].eq('up').all()
    assert v.loc[v.post_return_bps< -1e-10,'post_price_direction'].eq('down').all()
    assert v.loc[v.post_return_bps.isna(),'post_price_direction'].eq('missing').all()
    end=np.where(v.phase.eq('train'),lf.lb.VALIDATION,lf.lb.END)*1_000_000
    assert v.target_within_phase.eq(v.target_timestamp<=end).all()
    assert v.loc[~v.target_within_phase,'post_return_bps'].isna().all()
    assert v.query("anchor == 'onset'").class_known_at_anchor.eq('Not yet detected').all()
    assert v.query("anchor == 'cascade_detection'").classification.eq(lf.CLASSES[2]).all()
    valid=v.post_return_bps.notna()
    assert (v.loc[valid,'post_return_bps'].abs()<=np.sqrt(v.loc[valid,'horizon_s'])*v.loc[valid,'post_rv_bps']+1e-6).all()


def direct_intervals(seconds,starts,stops):
    """Direct sum of every 1s squared return; no PricePath and no prefix sums."""
    base=int(seconds['second'][0])
    mid=seconds['mid_close'].to_numpy()
    age=seconds['quote_age_upper_s'].to_numpy()
    ret=np.full(len(starts),np.nan);rv=ret.copy()
    starts,stops=np.asarray(starts,dtype=np.int64),np.asarray(stops,dtype=np.int64)
    lengths=stops-starts
    for h in np.unique(lengths):
        ix=np.flatnonzero((lengths==h)&(starts>base)&(stops<=base+len(mid)))
        if h<=0 or not len(ix): continue
        a=starts[ix]-base-1
        fresh=np.isfinite(mid[a])&(mid[a]>0)&(age[a]<=2)
        squared=np.zeros(len(ix))
        for offset in range(1,int(h)+1):
            j=a+offset
            fresh &= np.isfinite(mid[j])&(mid[j]>0)&(age[j]<=2)
            with np.errstate(invalid='ignore',divide='ignore'):
                squared+=np.log(mid[j]/mid[j-1])**2
        ret[ix]=np.where(fresh,np.log(mid[a+h]/mid[a])*1e4,np.nan)
        rv[ix]=np.where(fresh,np.sqrt(squared)*1e4,np.nan)
    return ret,rv


@pytest.mark.parametrize('symbol',['btcusdt','ethusdt'])
def test_every_saved_outcome_against_direct_second_calculation(symbol,outcomes,catalog):
    seconds=pl.read_parquet(lf.lb.OUT/f'seconds_{symbol}.parquet')
    v=outcomes.loc[outcomes.symbol.eq(symbol)]
    a=v.anchor_timestamp.to_numpy()//1_000_000
    o=v.onset_timestamp.to_numpy()//1_000_000
    h=v.horizon_s.to_numpy()
    evidence=[]
    for prefix,start,stop in [('post',a,a+h),('pre_anchor',a-h,a),('pre_onset',o-h,o),('from_onset',o,a+h)]:
        ret,rv=direct_intervals(seconds,start,stop)
        if prefix in ('post','from_onset'):
            ret[~v.target_within_phase]=np.nan;rv[~v.target_within_phase]=np.nan
        for name,expected in [('return_bps',ret),('rv_bps',rv)]:
            actual=v[f'{prefix}_{name}'].to_numpy()
            np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-9,equal_nan=True)
            evidence.append({'symbol':symbol,'metric':f'{prefix}_{name}','rows':len(v),
                'max_abs_error_bps':np.nanmax(np.abs(actual-expected))})
    c=catalog.loc[catalog.symbol.eq(symbol)]
    onset=c.onset_timestamp.to_numpy()//1_000_000;end=c.end_timestamp.to_numpy()//1_000_000
    ret,rv=direct_intervals(seconds,onset,end)
    np.testing.assert_allclose(c.during_return_bps,ret,atol=1e-5,equal_nan=True)
    np.testing.assert_allclose(c.during_rv_bps,rv,atol=1e-5,equal_nan=True)
    REPORT.mkdir(exist_ok=True)
    pd.DataFrame(evidence).to_csv(REPORT/f'direct_recalculation_{symbol}.csv',index=False)


def test_price_and_rv_summaries_have_correct_denominators(outcomes):
    price=pd.read_csv(lf.OUT/'price_summary.csv')
    vol=pd.read_csv(lf.OUT/'volatility_summary.csv')
    keys=['phase','symbol','classification','anchor','horizon_s']
    groups=dict(tuple(outcomes.groupby(keys)))
    for row in price.itertuples():
        g=groups[tuple(getattr(row,k) for k in keys)]
        if row.side!='ALL':g=g[g.side.eq(row.side)]
        x=g.post_return_bps.dropna();signed=g.loc[x.index,'post_signed_return_bps']
        assert row.episodes==len(g) and row.valid==len(x) and row.missing==len(g)-len(x)
        assert row.mean_post_return_bps==pytest.approx(x.mean(),abs=1e-10)
        assert row.mean_post_signed_return_bps==pytest.approx(signed.mean(),abs=1e-10)
        assert row.median_post_return_bps==pytest.approx(x.median(),abs=1e-10)
        assert row.up_share==pytest.approx((x>1e-10).sum()/len(x))
        assert row.down_share==pytest.approx((x< -1e-10).sum()/len(x))
        assert row.up_share+row.down_share+row.unchanged_share==pytest.approx(1)
    for row in vol.itertuples():
        g=groups[tuple(getattr(row,k) for k in keys)]
        if row.side!='ALL':g=g[g.side.eq(row.side)]
        pair=g[['post_rv_bps',f'{row.baseline}_rv_bps']].dropna()
        delta=pair.iloc[:,0]-pair.iloc[:,1]
        assert row.valid_pairs==len(pair)
        assert row.mean_delta_rv_bps==pytest.approx(delta.mean(),abs=1e-10)
        assert row.increased_share==pytest.approx((delta>1e-10).mean())
        assert row.decreased_share==pytest.approx((delta< -1e-10).mean())
        assert row.increased_share+row.decreased_share+row.unchanged_share==pytest.approx(1)


@pytest.mark.parametrize('symbol',['btcusdt','ethusdt'])
def test_normal_episodes_exactly_cover_eligible_active_nonburst_bins(symbol,catalog):
    fs={(ex,side):pl.read_parquet(lf.lb.OUT/f'features_{symbol}_{ex}_{side}.parquet',
        columns=['timestamp','eligible','liquidation_count','burst']).to_pandas()
        for ex in lf.lb.EXCHANGES for side in lf.lb.SIDES}
    any_burst=np.logical_or.reduce([f.burst.to_numpy() for f in fs.values()])
    for (ex,side),f in fs.items():
        c=catalog.loc[catalog.symbol.eq(symbol)&catalog.exchange.eq(ex)&catalog.side.eq(side)&catalog.classification.eq(lf.CLASSES[0])]
        expected=f.loc[f.eligible&(f.liquidation_count>0)&~any_burst,'timestamp'].to_numpy()
        reconstructed=np.concatenate([np.arange(a,b+1,5_000_000,dtype=np.int64)
            for a,b in zip(c.first_detection_timestamp,c.end_timestamp)])
        np.testing.assert_array_equal(np.sort(reconstructed),expected)
        assert (c.flagged_windows==0).all()


def test_stratified_samples_against_raw_bbo(outcomes):
    samples=outcomes.loc[outcomes.post_return_bps.notna() & outcomes.anchor.isin(['first_detection','episode_end'])]
    keys=['phase','symbol','classification','anchor','horizon_s']
    chosen=samples.groupby(keys,group_keys=False).sample(n=1,random_state=1011)
    records=[]
    for r in chosen.itertuples():
        t=r.anchor_timestamp//1_000_000;h=r.horizon_s
        boundaries=np.arange(t-h,t+h+1)*1_000_000
        source=lf.ROOT/'liquidation_task'/'data'/'binance_booktickers'/f'perp_{r.symbol}.parquet'
        raw=pl.scan_parquet(source).filter((pl.col('timestamp')>=int(boundaries[0]-5_000_000))&
            (pl.col('timestamp')<int(boundaries[-1]))).select('timestamp','bid_price','ask_price').collect().sort('timestamp',maintain_order=True)
        ts=raw['timestamp'].to_numpy();idx=np.searchsorted(ts,boundaries,side='left')-1
        assert (idx>=0).all()
        mid=((raw['bid_price'].to_numpy()+raw['ask_price'].to_numpy())/2)[idx]
        increments=np.log(mid[1:]/mid[:-1])
        ret=np.log(mid[-1]/mid[h])*1e4
        rv=np.linalg.norm(increments[h:])*1e4
        previous_rv=np.linalg.norm(increments[:h])*1e4
        assert ret==pytest.approx(r.post_return_bps,abs=1e-5)
        assert rv==pytest.approx(r.post_rv_bps,abs=1e-5)
        assert previous_rv==pytest.approx(r.pre_anchor_rv_bps,abs=1e-5)
        records.append({'episode_id':r.episode_id,**{k:getattr(r,k) for k in keys},
            'return_error_bps':ret-r.post_return_bps,'rv_error_bps':rv-r.post_rv_bps,
            'pre_rv_error_bps':previous_rv-r.pre_anchor_rv_bps})
    REPORT.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(REPORT/'raw_bbo_samples.csv',index=False)
