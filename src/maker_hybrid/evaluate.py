"""Full-frame official scores, matched turnover, and paired day-block uncertainty."""
import gc
from .core import *
from .baselines import totals_to_metrics
from .regression import KINDS, POLICY_START
from .policy import calibrate, ranked_groups

BUDGETS=(500000.,750000.,10000000.,250000000.)
SUMS=['all_w','all_wy','all_n','kept_w','kept_wy','kept_n',
      'all_fresh_w','all_fresh_wy','kept_fresh_w','kept_fresh_wy']


def load_period(tag,h):
    frames=[]
    for symbol in SYMBOLS:
        for path in sorted((OUT/'predictions'/tag/symbol).glob('*.parquet')):
            fields=['second','taker_sign','alignment',f'w_{h}',f'wy_{h}',f'wy2_{h}',f'n_{h}',
                    f'fresh_w_{h}',f'fresh_wy_{h}']
            g=pd.read_parquet(OUT/'groups'/symbol/path.name,columns=fields)
            p=pd.read_parquet(path,columns=['second','taker_sign','alignment','tie_key',
                'feature_quote_age_upper_s','volatility_regime',*[f'{k}_{h}' for k in KINDS]])
            pd.testing.assert_frame_equal(g[['second','taker_sign','alignment']],p[['second','taker_sign','alignment']])
            for c in p.columns.difference(g.columns):g[c]=p[c]
            g['tie_key']=policy_hash(g.second,symbol,g.taker_sign,g.alignment)
            g['symbol']=symbol
            frames.append(g)
    f=pd.concat(frames,ignore_index=True)
    if tag=='holdout':
        # Do not consume labels crossing the Jan/Feb boundary for policy selection.
        f=f[f.second+301<VALIDATION].reset_index(drop=True)
    return f


def daily_stats(f,h,whole,partial,start,days,policy):
    d=((f.second.to_numpy()-start)//86400).astype(int)
    keep=np.zeros(len(f),bool);keep[whole]=True
    pairs={'w':f[f'w_{h}'].to_numpy(),'wy':f[f'wy_{h}'].to_numpy(),'n':f[f'n_{h}'].to_numpy(),
           'fresh_w':f[f'fresh_w_{h}'].to_numpy(),'fresh_wy':f[f'fresh_wy_{h}'].to_numpy()}
    table=pd.DataFrame({'day':pd.date_range(pd.Timestamp(start,unit='s',tz='UTC'),periods=days).strftime('%Y-%m-%d')})
    for name,v in pairs.items():
        table['all_'+name]=np.bincount(d,weights=v,minlength=days)
        table['kept_'+name]=np.bincount(d[keep],weights=v[keep],minlength=days).astype(float)
    if partial is not None:
        boundary,part=partial
        day=d[boundary]
        for name in pairs:table.loc[day,'kept_'+name]+=part[f'{name}_{h}'].sum()
    for k in ['model','scope','tau','budget_per_day','calibration_period']:
        table[k]=policy[k]
    return table


def summarize_daily(daily,bootstrap=5000):
    rows=[]
    keys=['model','scope','tau','budget_per_day','calibration_period']
    for key,f in daily.groupby(keys,dropna=False):
        f=f.sort_values('day');a=f[SUMS].sum().to_dict();days=len(f)
        metrics=totals_to_metrics(a,days)
        rng=np.random.default_rng(SEED)
        draws=rng.integers(0,days,size=(bootstrap,days))
        kw=f.kept_w.to_numpy()[draws].sum(axis=1)
        ky=f.kept_wy.to_numpy()[draws].sum(axis=1)
        aw=f.all_w.to_numpy()[draws].sum(axis=1)
        ay=f.all_wy.to_numpy()[draws].sum(axis=1)
        score=np.divide(ky,kw,out=np.full(bootstrap,np.nan),where=kw>0)-ay/aw
        finite=score[np.isfinite(score)]
        best=(f.kept_wy-f.all_wy/f.all_w*f.kept_w).idxmax()
        minus=f.drop(best)[SUMS].sum().to_dict()
        rows.append({**dict(zip(keys,key)),**metrics,
            'symbol':key[1],'days':days,'ci_low':float(np.quantile(finite,.025)) if len(finite)==bootstrap else np.nan,
            'ci_high':float(np.quantile(finite,.975)) if len(finite)==bootstrap else np.nan,
            'conditional_ci_low':float(np.quantile(finite,.025)) if len(finite) else np.nan,
            'conditional_ci_high':float(np.quantile(finite,.975)) if len(finite) else np.nan,
            'ci_status':'all bootstrap draws defined' if len(finite)==bootstrap else 'some draws retain no trades; conditional quantiles are not a nominal CI',
            'bootstrap_finite_fraction':len(finite)/bootstrap,
            'minimum_daily_kept_turnover':f.kept_w.min(),
            'fraction_days_below_500k':float((f.kept_w<500000).mean()),
            'days_with_kept_trades':int((f.kept_n>0).sum()),
            'retained_trade_count':int(a['kept_n']),
            'largest_day_share_of_kept_turnover':f.kept_w.max()/a['kept_w'] if a['kept_w'] else np.nan,
            'score_without_best_day':totals_to_metrics(minus,days-1)['score'],
            'fresh_future_quote_score':a['kept_fresh_wy']/a['kept_fresh_w']-a['all_fresh_wy']/a['all_fresh_w'] if a['kept_fresh_w'] else np.nan,
            'kept_stale_weight_fraction':1-a['kept_fresh_w']/a['kept_w'] if a['kept_w'] else np.nan,
            'matching_extra_turnover_usd':a['kept_w']-key[3]*days})
    return pd.DataFrame(rows)


def paired_comparisons(daily):
    """Primary family: E5/E7 vs E3 and E7 vs E1, combined $250m/day, 3 horizons.

    Nine comparisons use Bonferroni simultaneous coverage. Others are descriptive.
    Reused February and policy/model selection still preclude independent alpha.
    """
    rows=[]
    for scope in [*SYMBOLS,'combined']:
        for h in HORIZONS:
            for budget in BUDGETS:
                d=daily[(daily.scope==scope)&(daily.tau==h)&(daily.budget_per_day==budget)&
                        (daily.calibration_period=='validation')]
                days=sorted(d.day.unique())
                rng=np.random.default_rng(SEED)
                draws=rng.integers(0,len(days),size=(5000,len(days)))
                distributions={};observed={}
                for model,f in d.groupby('model'):
                    f=f.set_index('day').reindex(days)
                    w=f.kept_w.to_numpy();y=f.kept_wy.to_numpy()
                    sums=w[draws].sum(axis=1)
                    distributions[model]=np.divide(y[draws].sum(axis=1),sums,out=np.full(5000,np.nan),where=sums>0)
                    observed[model]=y.sum()/w.sum() if w.sum() else np.nan
                for a,b in [('E3','E1'),('E5','E3'),('E6','E3'),('E7','E5'),('E7','E3'),('E7','E1')]:
                    if a not in distributions or b not in distributions:continue
                    delta=distributions[a]-distributions[b]
                    finite=delta[np.isfinite(delta)]
                    primary=(scope=='combined' and budget==250000000 and (a,b) in [('E5','E3'),('E7','E3'),('E7','E1')])
                    alpha=.05/9 if primary else .05
                    rows.append({'scope':scope,'tau':h,'budget_per_day':budget,'model':a,'comparator':b,
                        'delta_score':observed[a]-observed[b],'ci_low':np.quantile(finite,alpha/2) if len(finite)==5000 else np.nan,
                        'ci_high':np.quantile(finite,1-alpha/2) if len(finite)==5000 else np.nan,
                        'confidence_level':1-alpha,'primary_family_bonferroni':primary,
                        'finite_bootstrap_fraction':len(finite)/5000})
    return pd.DataFrame(rows)


def run():
    policy_file=OUT/'budget_policies.json'
    if (OUT/'validation_report.csv').exists():return
    policies=[];daily=[];regimes=[];regression_errors=[]
    for tag,start,end in [('holdout',POLICY_START,VALIDATION),('final',VALIDATION,END)]:
        period='train_holdout' if tag=='holdout' else 'validation'
        days=(end-start)//86400
        for h in HORIZONS:
            allf=load_period(tag,h)
            expected=20 if tag=='holdout' else 56
            if len(list((OUT/'predictions'/tag).glob('*/*.parquet')))!=expected:
                raise RuntimeError('Incomplete prediction period')
            for scope in [*SYMBOLS,'combined']:
                f=(allf if scope=='combined' else allf[allf.symbol.eq(scope)]).reset_index(drop=True)
                for model in ['E0_random','E1',*KINDS]:
                    rank=ranked_groups(f,model,h)
                    for budget in BUDGETS:
                        policy,whole,partial=calibrate(f,model,h,budget,days,rank)
                        policy.update(scope=scope,calibration_period=period)
                        policies.append(policy)
                        if not policy['feasible']:continue
                        d=daily_stats(f,h,whole,partial,start,days,policy)
                        daily.append(d)
                        # Regimes use the same binary decisions and official scorer within each population.
                        if tag=='final' and budget==250000000 and scope=='combined':
                            keep=np.zeros(len(f),bool);keep[whole]=True
                            for family,values in [('liquidation',f.alignment.to_numpy()),
                                                  ('volatility',f.volatility_regime.to_numpy()),
                                                  ('feature_staleness',(f.feature_quote_age_upper_s>2).astype(int).to_numpy())]:
                                for regime in np.unique(values):
                                    mask=values==regime;kk=keep&mask
                                    a={f'all_{n}':float(f.loc[mask,f'{n}_{h}'].sum()) for n in ['w','wy','n']}
                                    a.update({f'kept_{n}':float(f.loc[kk,f'{n}_{h}'].sum()) for n in ['w','wy','n']})
                                    if partial is not None and mask[partial[0]]:
                                        for n in ['w','wy','n']:a['kept_'+n]+=partial[1][f'{n}_{h}'].sum()
                                    regimes.append({'model':model,'tau':h,'family':family,'regime':int(regime),**totals_to_metrics(a,days)})
                    if model in KINDS:
                        p=f[f'{model}_{h}'].to_numpy(float)
                        mse=(f[f'wy2_{h}']-2*p*f[f'wy_{h}']+p*p*f[f'w_{h}']).sum()/f[f'w_{h}'].sum()
                        regression_errors.append({'period':period,'model':model,'symbol':scope,'tau':h,'weighted_trade_mse_bps2':mse})
                    print('Evaluated',period,scope,h,model,flush=True)
            del allf;gc.collect()
            save_json(policy_file,policies)
            pd.concat(daily,ignore_index=True).to_csv(OUT/'policy_daily.csv',index=False)
    d=pd.concat(daily,ignore_index=True)
    report=summarize_daily(d)
    pd.DataFrame(regimes).to_csv(OUT/'regime_scores.csv',index=False)
    pd.DataFrame(regression_errors).to_csv(OUT/'regression_error_diagnostics.csv',index=False)
    paired_comparisons(d).to_csv(OUT/'paired_comparisons.csv',index=False)
    report.to_csv(OUT/'validation_report.csv',index=False)
    save_json(OUT/'evaluation_complete.json',{'scores_use_all_trades':True,
        'matching_precision':'budget rounded upward by <= one clipped trade, <=100000 USD per full period',
        'validation_role':'fixed-threshold calibration using scores/weights only, then retrospective Score comparison',
        'bootstrap':'5000 paired whole UTC days, both symbols kept together in combined scope',
        'primary_family':'9 Bonferroni-adjusted E5/E7/E1/E3 comparisons, combined 250m/day across 3 horizons',
        'hidden_test_available':False})


if __name__=='__main__':run()
