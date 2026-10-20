"""Reproducible report tables and plots from frozen model predictions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .evaluate import summarize_daily,SUMS
from .baselines import totals_to_metrics
from .features import feature_columns
from .policy import full_group_keep

NAMES={'E0_random':'Random','E1':'Keep-with 5s','E1_60':'Keep-with 60s','E1_60_strength':'Keep-with ranked by flow',
       'E2':'Ridge (new)','E3':'LightGBM','E4':'No liquidations','E5':'+ LR','E6':'+ AS','E7':'+ LR + AS'}
COLORS={'E0_random':'#92989e','E1':'#a5b9c0','E1_60':'#b57c22','E1_60_strength':'#ac5628','E2':'#1f77b4','E3':'#55535b','E4':'#a74432','E5':'#22886e','E7':'#7a4c9b'}
FIG=OUT/'figures'


def save(fig,name):
    FIG.mkdir(exist_ok=True);fig.savefig(FIG/name,dpi=160,bbox_inches='tight');plt.close(fig)


def block_comparisons(d):
    rows=[]
    for block in [1,5]:
        for scope in ['btcusdt','ethusdt','combined']:
            for h in HORIZONS:
                x=d[(d.scope==scope)&(d.tau==h)&(d.budget_per_day==250000000)&(d.calibration_period=='validation')]
                days=sorted(x.day.unique());n=len(days)
                rng=np.random.default_rng(SEED)
                starts=rng.integers(0,n,size=(10000,int(np.ceil(n/block))))
                draws=((starts[:,:,None]+np.arange(block))%n).reshape(10000,-1)[:,:n]
                dist={};point={}
                for k,f in x.groupby('model'):
                    f=f.set_index('day').reindex(days)
                    w=f.kept_w.to_numpy();y=f.kept_wy.to_numpy();ws=w[draws].sum(axis=1)
                    dist[k]=np.divide(y[draws].sum(axis=1),ws,out=np.full(len(ws),np.nan),where=ws>0)
                    point[k]=y.sum()/w.sum()
                comparisons=[('E5','E3'),('E7','E3'),('E7','E5'),('E6','E3'),('E3','E4'),('E7','E1_60'),('E7','E1_60_strength'),('E2','E7')]
                for a,b in comparisons:
                    if a not in dist or b not in dist:continue
                    z=dist[a]-dist[b];z=z[np.isfinite(z)]
                    # All 24 combined horizon/comparison tests in this expanded family.
                    alpha=.05/24 if scope=='combined' else .05
                    rows.append({'block_days':block,'scope':scope,'tau':h,'model':a,'comparator':b,
                        'delta_score':point[a]-point[b],'ci_low':np.quantile(z,alpha/2) if len(z)==10000 else np.nan,
                        'ci_high':np.quantile(z,1-alpha/2) if len(z)==10000 else np.nan,'finite_bootstrap_fraction':len(z)/10000,
                        'confidence_level':1-alpha,'interpretation':'source-informed expanded family; retrospective February'})
    pd.DataFrame(rows).to_csv(OUT/'expanded_paired_comparisons.csv',index=False)


def tables():
    r=pd.concat([pd.read_csv(OUT/'validation_report.csv'),pd.read_csv(OUT/'baseline60_report.csv')],ignore_index=True)
    baseline=pd.read_csv(OUT/'baseline_summary.csv').query("model == 'E0'").copy()
    baseline['scope']=baseline.symbol
    baseline['calibration_period']=baseline['split'].replace({'train':'train_all'})
    baseline['budget_per_day']=-1.
    baseline['ci_low']=0.;baseline['ci_high']=0.
    r=pd.concat([r,baseline],ignore_index=True)
    r['confidence_interval']=[f'[{lo:.6f}, {hi:.6f}]' if np.isfinite(lo) and np.isfinite(hi) else 'undefined: empty-kept bootstrap draws' for lo,hi in zip(r.ci_low,r.ci_high)]
    r['confidence_level']=.95
    r.to_csv(OUT/'all_validation_results.csv',index=False)
    d=pd.concat([pd.read_csv(OUT/'policy_daily.csv'),pd.read_csv(OUT/'baseline60_daily.csv')],ignore_index=True)
    week=pd.to_datetime(d.day).dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
    d['week_start']=week
    rows=[]
    for key,f in d.groupby(['model','scope','tau','budget_per_day','calibration_period','week_start']):
        rows.append(dict(zip(['model','scope','tau','budget_per_day','calibration_period','week_start'],key))|
            {'observed_days':len(f),'period_start':f.day.min(),'period_end':f.day.max()}|totals_to_metrics(f[SUMS].sum().to_dict(),len(f)))
    pd.DataFrame(rows).to_csv(OUT/'weekly_scores.csv',index=False)
    block_comparisons(d)
    delta=pd.read_csv(OUT/'price_cleaning_sensitivity_deltas.csv')
    sensitivity=[]
    native=d[(d.model=='E1_60_native')&(d.calibration_period=='validation')]
    for (scope,h),f in native.groupby(['scope','tau']):
        change=delta[delta.tau.eq(h)]
        if scope!='combined':change=change[change.symbol.eq(scope)]
        a=f[SUMS].sum().to_dict();base=totals_to_metrics(a,len(f))
        a['kept_w']+=change.delta_kept_w.sum();a['kept_wy']+=change.delta_kept_wy.sum()
        after=totals_to_metrics(a,len(f))
        sensitivity.append({'symbol':scope,'tau':h,'raw_price_score':base['score'],'flagged_removed_score':after['score'],
                            'delta_score':after['score']-base['score'],'changed_decisions':int(change.changed_decisions.sum())})
    pd.DataFrame(sensitivity).to_csv(OUT/'price_cleaning_sensitivity_summary.csv',index=False)
    # Include train feature correlations, with labels explicitly excluded.
    train=pd.read_parquet(OUT/'regression_training_sample.parquet',columns=feature_columns('E7'))
    # Fixed systematic 1/10 of the TRAIN group sample; descriptive correlations.
    corr=train.iloc[::10].corr();pairs=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            if abs(corr.loc[a,b])>=.95:pairs.append({'feature_a':a,'feature_b':b,'correlation':corr.loc[a,b]})
    pd.DataFrame(pairs).to_csv(OUT/'high_feature_correlations.csv',index=False)
    imp=pd.read_csv(OUT/'feature_importance.csv')
    imp['family']=np.select([imp.feature.str.startswith('p_'),imp.feature.str.startswith('as_'),imp.feature.str.startswith('liq_')],['LR_probability','AS_risk','liquidation'],default='market')
    imp.groupby(['kind','tau','family'],as_index=False).gain.sum().to_csv(OUT/'feature_family_importance.csv',index=False)
    return r,d


def plots(r,d):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for i,scope in enumerate(['btcusdt','ethusdt','combined']):
        for j,h in enumerate(HORIZONS):
            f=r[(r.scope==scope)&(r.tau==h)&(r.calibration_period=='validation')]
            for k in ['E0_random','E1_60_strength','E2','E3','E4','E7']:
                x=f[f.model.eq(k)].sort_values('budget_per_day')
                ax[i,j].plot(x.kept_turnover_per_day/1e6,x.score,'o-',label=NAMES[k],color=COLORS[k],ms=3)
            ax[i,j].set_xscale('log');ax[i,j].axhline(0,color='black',lw=.6);ax[i,j].set_title(f'{scope.upper()} | {h}s')
            if j==0:ax[i,j].set_ylabel('Official Score, bps')
            if i==2:ax[i,j].set_xlabel('Retained clipped USD million/day')
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,bbox_to_anchor=(.5,-.015))
    fig.suptitle('February: same retained-turnover targets; thresholds use scores and weights only',y=1.01)
    fig.tight_layout();save(fig,'score_turnover.png')
    cal=pd.read_csv(OUT/'lr_calibration_bins.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for j,side in enumerate(['buy','sell']):
        for s,c in [('btcusdt','#287fa3'),('ethusdt','#ba6829')]:
            f=cal[(cal.symbol==s)&(cal.side==side)]
            ax[j].plot(f.predicted*100,f.observed*100,'o-',label=s.upper(),color=c)
        limit=max(ax[j].get_xlim()[1],ax[j].get_ylim()[1]);ax[j].plot([0,limit],[0,limit],'k--',lw=.8)
        ax[j].set(title=f'New {side} burst within 5s',xlabel='Mean predicted probability, %',ylabel='Observed event rate, %');ax[j].legend()
    fig.suptitle('Uncalibrated LR predict_proba; equal-count reliability bins on February')
    fig.tight_layout();save(fig,'lr_calibration.png')
    f=d[(d.scope=='combined')&(d.tau==120)&(d.budget_per_day==250000000)&(d.calibration_period=='validation')]
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for k in ['E1_60_strength','E2','E3','E7']:
        x=f[f.model.eq(k)].sort_values('day');dates=pd.to_datetime(x.day)
        score=x.kept_wy/x.kept_w.replace(0,np.nan)-x.all_wy/x.all_w
        ax[0].plot(dates,score,'.-',label=NAMES[k],color=COLORS[k]);ax[1].plot(dates,x.kept_w/1e6,'.-',color=COLORS[k])
    ax[0].axhline(0,color='black',lw=.6);ax[0].set_ylabel('Daily Score, bps');ax[0].legend(ncol=4)
    ax[1].set_ylabel('Retained USD million/day');ax[1].set_xlabel('UTC date (February 2026)')
    fig.suptitle('120s, combined $250m/day average: daily concentration matters');fig.tight_layout();save(fig,'daily_stability.png')
    wk=pd.read_csv(OUT/'weekly_scores.csv');fig,ax=plt.subplots(1,3,figsize=(14,4))
    for j,h in enumerate(HORIZONS):
        for k in ['E1_60_strength','E2','E3','E7']:
            x=wk[(wk.model==k)&(wk.tau==h)&(wk.scope=='combined')&(wk.budget_per_day==250000000)].sort_values('week_start')
            # Periods remain separate at the Jan/Feb split; partial weeks are labeled.
            ax[j].plot(np.arange(len(x)),x.score,'o-',label=NAMES[k],color=COLORS[k])
            ax[j].set_xticks(np.arange(len(x)),[a[5:]+' to\n'+b[5:]+'\n'+str(int(n))+'d' for a,b,n in zip(x.period_start,x.period_end,x.observed_days)],fontsize=8)
        ax[j].set_title(f'{h}s');ax[j].axhline(0,color='black',lw=.6)
    ax[0].set_ylabel('Weekly Score, bps');ax[0].legend(fontsize=8)
    fig.suptitle('Jan22-31 holdout and February; partial weeks show observed days');fig.tight_layout();save(fig,'weekly_stability.png')
    imp=pd.read_csv(OUT/'feature_family_importance.csv');fig,ax=plt.subplots(1,3,figsize=(12,4))
    for j,h in enumerate(HORIZONS):
        f=imp[(imp.kind=='E7')&(imp.tau==h)];ax[j].bar(f.family,f.gain/f.gain.sum(),color=['#888888','#7a4c9b','#b57c22','#287fa3'][:len(f)])
        ax[j].tick_params(axis='x',labelrotation=25);ax[j].set_title(f'E7, {h}s')
    ax[0].set_ylabel('Fraction of training split gain');fig.suptitle('Gain is descriptive; incremental value comes from controlled ablations');fig.tight_layout();save(fig,'feature_importance.png')


def distributions():
    # A reproducible sample of individual trades; never group-mean histograms.
    policies=json.loads((OUT/'budget_policies.json').read_text())
    edges=np.linspace(-150,150,151);hist={}
    for h in HORIZONS:
        p=next(p for p in policies if p['model']=='E7' and p['tau']==h and p['scope']=='combined' and p['calibration_period']=='validation' and p['budget_per_day']==250000000)
        a=np.zeros(len(edges)-1);b=a.copy();outside=np.zeros(4);n=0;boundary_omitted=0
        for s in SYMBOLS:
            for path in sorted((OUT/'predictions'/'final'/s).glob('*.parquet')):
                sample=pd.read_parquet(OUT/'trade_samples'/s/path.name)
                pred=pd.read_parquet(path,columns=['second','taker_sign','alignment','tie_key',f'E7_{h}'])
                f=sample.merge(pred,on=['second','taker_sign','alignment'],validate='many_to_one')
                keep,bound=full_group_keep(f[f'E7_{h}'],f.tie_key,f.alignment,p)
                # At most the one boundary group; exclude because sample lacks duplicate occurrence keys.
                valid=(f[f'w_{h}']>0)&~bound;boundary_omitted+=int(bound.sum())
                y=np.divide(f[f'wy_{h}'],f[f'w_{h}'],out=np.zeros(len(f)),where=f[f'w_{h}'].to_numpy()>0)
                w=f[f'w_{h}'].to_numpy();n+=int(valid.sum())
                for selected,target,j in [(valid&keep,a,0),(valid&~keep,b,2)]:
                    target+=np.histogram(y[selected],bins=edges,weights=w[selected])[0]
                    outside[j]+=w[selected&(y<edges[0])].sum();outside[j+1]+=w[selected&(y>edges[-1])].sum()
        hist[h]=(a,b,outside,n,boundary_omitted)
    fig,ax=plt.subplots(1,3,figsize=(14,4));records=[]
    for j,h in enumerate(HORIZONS):
        a,b,o,n,omitted=hist[h];cent=(edges[:-1]+edges[1:])/2
        for name,v,tail,col in [('kept',a,o[:2],'#22886e'),('filtered',b,o[2:],'#999999')]:
            denom=v.sum()+tail.sum();ax[j].plot(cent,v/denom,label=name,color=col)
            records.append({'tau':h,'group':name,'sample_rows_total':n,'boundary_sample_rows_omitted':omitted,
                            'left_tail_weight_fraction':tail[0]/denom,'right_tail_weight_fraction':tail[1]/denom})
        ax[j].set(title=f'{h}s',xlabel='Maker markout, bps');ax[j].legend()
    ax[0].set_ylabel('Fraction of sampled clipped turnover per 2bps bin')
    fig.suptitle('E7 $250m/day: individual-trade 1/1000 systematic sample; tails reported separately')
    fig.tight_layout();save(fig,'markout_distributions.png')
    pd.DataFrame(records).to_csv(OUT/'distribution_sample_diagnostics.csv',index=False)


def lr_brier_uncertainty():
    f=pd.read_parquet(OUT/'lr_validation_probabilities.parquet');f['day']=f.second//86400
    prior=json.loads((OUT/'models/lr_final.json').read_text())['train_prior'];rows=[]
    for scope in [*SYMBOLS,'combined']:
        for side in ['buy','sell']:
            x=f if scope=='combined' else f[f.symbol.eq(scope)]
            y=x['y_'+side].to_numpy();p=x[f'p_{side}_burst_5s'].to_numpy()
            table=pd.DataFrame({'day':x.day.to_numpy(),'improvement':(prior[side]-y)**2-(p-y)**2}).groupby('day').improvement.agg(['sum','size'])
            n=len(table)
            for block in [1,5]:
                rng=np.random.default_rng(SEED);starts=rng.integers(0,n,(20000,int(np.ceil(n/block))))
                ix=((starts[:,:,None]+np.arange(block))%n).reshape(20000,-1)[:,:n]
                v=table['sum'].to_numpy()[ix].sum(axis=1)/table['size'].to_numpy()[ix].sum(axis=1)
                low,high=np.quantile(v,[.05/12,1-.05/12])
                rows.append({'symbol':scope,'side':side,'block_days':block,'brier_improvement_over_constant':table['sum'].sum()/table['size'].sum(),
                             'ci_low':low,'ci_high':high,'confidence_level':1-.05/6,'multiplicity':'Bonferroni6 symbol/side comparisons; retrospective'})
    pd.DataFrame(rows).to_csv(OUT/'lr_brier_comparison.csv',index=False)


def upper_probability_bins():
    f=pd.read_parquet(OUT/'lr_validation_probabilities.parquet');rows=[]
    for s in [*SYMBOLS,'combined']:
        for side in ['buy','sell']:
            x=f if s=='combined' else f[f.symbol.eq(s)]
            p=x[f'p_{side}_burst_5s'].to_numpy();y=x[f'y_{side}'].to_numpy()
            cuts=np.quantile(p,[.9,.99,.999,1])
            for j,name in enumerate(['90-99 percentile','99-99.9 percentile','99.9-100 percentile']):
                use=(p>=cuts[j])&((p<cuts[j+1]) if j<2 else (p<=cuts[j+1]))
                rows.append({'symbol':s,'side':side,'bin':name,'rows':int(use.sum()),'positives':int(y[use].sum()),'predicted':float(p[use].mean()),'observed':float(y[use].mean()),'min_probability':cuts[j],'max_probability':cuts[j+1]})
    pd.DataFrame(rows).to_csv(OUT/'lr_upper_probability_bins.csv',index=False)


def ablation_plot():
    f=pd.read_csv(OUT/'expanded_paired_comparisons.csv')
    f=f[(f.scope=='combined')&(f.block_days==5)]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(a,b,title) in zip(axes,[('E5','E3','Add LR probabilities'),('E7','E5','Add AS risk features'),('E7','E1_60_strength','Hybrid minus flow-ranked rule')]):
        x=f[(f.model==a)&(f.comparator==b)].set_index('tau').loc[list(HORIZONS)]
        for j,row in enumerate(x.itertuples()):
            ax.plot([row.ci_low,row.ci_high],[j,j],color='#757a83',lw=2)
            ax.plot(row.delta_score,j,'o',color='#7a4c9b')
        ax.axvline(0,color='black',lw=.8,ls='--')
        ax.set_yticks(range(3),[f'{h}s' for h in HORIZONS]);ax.set_ylim(-.5,2.5)
        ax.set_title(title);ax.set_xlabel('Difference in official Score, bps')
        if np.allclose(x.delta_score,0) and np.allclose(x.ci_low,0):
            ax.set_xlim(-1,1);ax.text(.08,1.4,'Identical predictions',fontsize=10)
    fig.suptitle('Combined $250m/day | 5-day blocks | Bonferroni 24 comparisons (99.79% intervals)')
    fig.tight_layout();save(fig,'paired_ablation.png')


if __name__=='__main__':
    r,d=tables();plots(r,d);distributions();lr_brier_uncertainty();upper_probability_bins();ablation_plot()
