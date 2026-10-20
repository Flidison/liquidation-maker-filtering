"""Small presentation helpers; all reported observations come from frozen tables."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from research_paths import ROOT, artifact_dir

COLORS = {'E0_random':'#687681', 'E1':'#89949d', 'E1_60':'#baa46b',
          'E1_60_strength':'#9b6c13', 'E2':'#69783d', 'E3':'#24659a',
          'E4':'#d18a29', 'E5':'#638ba5', 'E6':'#728b80', 'E7':'#ac6580', 'E9':'#744f8b'}
STRATEGIES = ['E0_random','E1_60','E1_60_strength','E2','E3','E4','E5','E6','E7','E9']


def style():
    plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120, 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.axisbelow':True,
        'axes.grid':True, 'grid.alpha':.16, 'savefig.bbox':'tight'})
    pd.set_option('display.max_columns',14)
    pd.set_option('display.max_rows',22)
    pd.set_option('display.float_format',lambda x:f'{x:,.6f}')


def table(group, name):
    return pd.read_csv(artifact_dir(group)/name)


def primary(models=None):
    r = table('catboost','v1/comparison_report.csv')
    r = r.loc[r.scope.eq('combined') & r.symbol.eq('combined') &
              r.budget_per_day.eq(250e6) & r.calibration_period.eq('validation')].copy()
    if models is not None: r = r[r.model.isin(models)]
    assert not r.duplicated(['model','tau']).any()
    return r.sort_values(['model','tau'])


def score_matrix():
    return primary(STRATEGIES).pivot(index='model',columns='tau',values='score').reindex(STRATEGIES)


def score_figure(models=None):
    models = models or STRATEGIES
    matrix = primary(models).pivot(index='model',columns='tau',values='score').reindex(models)
    fig, axes = plt.subplots(1,3,figsize=(13,5),sharey=True,sharex=True)
    for ax,h in zip(axes,[30,120,300]):
        ax.barh(models,matrix[h],color=[COLORS[x] for x in models])
        ax.axvline(0,color='#333333',lw=.8)
        ax.set(title=f'{h} seconds',xlabel='Score (bps)')
    axes[0].invert_yaxis()
    fig.suptitle('February 2026 · BTC + ETH · matched $250m clipped turnover/day')
    fig.tight_layout()
    return fig


def family_importance_figure():
    f=table('maker','feature_family_importance.csv').query("kind in ['E3','E5','E6','E7']")
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    palette={'liquidation':'#24659a','market':'#d18a29','LR_probability':'#ac6580','AS_risk':'#69783d'}
    for ax,h in zip(axes,[30,120,300]):
        p=f[f.tau.eq(h)].pivot(index='kind',columns='family',values='gain').fillna(0)
        p=p.div(p.sum(axis=1),axis=0)*100
        p=p.reindex(columns=palette,fill_value=0)
        p.plot.barh(stacked=True,color=list(palette.values()),ax=ax,legend=False)
        ax.set(title=f'{h} seconds',xlabel='Share of LightGBM gain (%)',ylabel='')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.05))
    fig.suptitle('Feature-family importance · fixed maker regressors')
    fig.tight_layout()
    return fig


def top_importance_figure(group='maker'):
    f=table(group, 'feature_importance.csv' if group=='maker' else 'v1/feature_importance.csv')
    if group=='maker': f=f[f.kind.eq('E3')];value='gain';xlabel='LightGBM gain'
    else: value='prediction_values_change';xlabel='CatBoost PredictionValuesChange'
    fig,axes=plt.subplots(3,1,figsize=(10,8.5))
    for ax,h in zip(axes,[30,120,300]):
        p=f[f.tau.eq(h)].nlargest(5,value).sort_values(value)
        ax.barh(p.feature,p[value],color=COLORS['E3' if group=='maker' else 'E9'])
        ax.set(title=f'{h} seconds',xlabel=xlabel)
        ax.tick_params(axis='y',labelsize=10)
    fig.suptitle('Fixed model feature importance · magnitudes differ between model families')
    fig.tight_layout()
    return fig


def mse_figure():
    f=table('catboost','v1/regression_error_diagnostics.csv').query("period=='validation' and symbol=='combined'")
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,h in zip(axes,[30,120,300]):
        p=f[f.tau.eq(h)].set_index('model').reindex(['E3','E9'])
        bars=ax.bar(p.index,p.weighted_trade_mse_bps2,color=[COLORS[x] for x in p.index])
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
        ax.set_ylim(0,p.weighted_trade_mse_bps2.max()*1.13)
        ax.set(title=f'{h} seconds',ylabel='Weighted MSE (bps²)')
    fig.suptitle('February · exact full-trade errors · each horizon has its own y scale')
    fig.tight_layout()
    return fig


def paired_figure():
    p=table('catboost','v1/paired_e9_e3.csv').query('primary_family_bonferroni and block_days==5').sort_values('tau')
    fig,ax=plt.subplots(figsize=(9,3.5))
    for i,row in enumerate(p.itertuples()):
        ax.plot([row.ci_low,row.ci_high],[i,i],color=COLORS['E9'],lw=2)
        ax.scatter(row.delta_score,i,color=COLORS['E9'],zorder=3)
    ax.axvline(0,color='#333333',lw=1)
    ax.set(yticks=range(len(p)),yticklabels=[f'{h} seconds' for h in p.tau],
           xlabel='E9 − E3 Score (bps)',title='5-day paired block bootstrap · Bonferroni family of 3')
    fig.text(.5,-.01,'98.33% intervals per comparison; reused February validation',ha='center',fontsize=9)
    fig.tight_layout()
    return fig


def daily_primary():
    frames=[table('maker','policy_daily.csv'),table('maker','baseline60_daily.csv'),table('catboost','v1/policy_daily.csv')]
    f=pd.concat(frames,ignore_index=True)
    return f.loc[f.scope.eq('combined') & f.tau.eq(120) & f.budget_per_day.eq(250e6) &
                 f.calibration_period.eq('validation') & f.model.isin(['E1_60_strength','E3','E4','E9'])].copy()


def daily_figure():
    f=daily_primary()
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for model,g in f.groupby('model'):
        g=g.sort_values('day');days=pd.to_datetime(g.day)
        score=g.kept_wy/g.kept_w.replace(0,np.nan)-g.all_wy/g.all_w
        axes[0].plot(days,score,label=model,color=COLORS[model],marker='.',lw=1)
        axes[1].plot(days,g.kept_w/1e6,label=model,color=COLORS[model],marker='.',lw=1)
    axes[0].set(ylabel='Daily Score (bps)',title='February daily stability · 120 seconds · combined budget calibration')
    axes[1].set(ylabel='Kept clipped turnover ($m)',xlabel='UTC day')
    axes[0].axhline(0,color='#333333',lw=.7);axes[0].legend(ncol=4)
    axes[1].axhline(.5,color='#333333',lw=.8,ls='--',label='Official average-daily floor')
    fig.autofmt_xdate();fig.tight_layout()
    return fig


def turnover_figure():
    r=table('catboost','v1/comparison_report.csv')
    r=r.loc[r.scope.eq('combined') & r.symbol.eq('combined') & r.calibration_period.eq('validation') & r.model.isin(['E3','E4','E9'])]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,h in zip(axes,[30,120,300]):
        for model,g in r[r.tau.eq(h)].groupby('model'):
            g=g.sort_values('budget_per_day')
            ax.plot(g.kept_turnover_per_day/1e6,g.score,marker='o',label=model,color=COLORS[model])
        ax.set(xscale='log',xlabel='Kept turnover ($m/day, log)',ylabel='Score (bps)',title=f'{h} seconds')
        ax.axhline(0,color='#333333',lw=.7)
    axes[-1].legend();fig.suptitle('February Score versus matched turnover · BTC + ETH')
    fig.tight_layout();return fig


def liquidation_distribution_figure():
    f=table('quality','distribution_histograms.csv')
    f=f.loc[f.source_id.str.contains('liquidations') & f.metric.eq('notional')]
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,symbol in zip(axes,['btcusdt','ethusdt']):
        for exchange,color in [('binance','#24659a'),('bybit','#d18a29')]:
            q=f.loc[f.source_id.str.contains(exchange) & f.ticker.str.endswith(symbol)]
            # Exact precomputed histogram; log1p transform matches the original audit.
            ax.stairs(q['count'].to_numpy()/q['count'].sum(),np.r_[q.left.to_numpy(),q.right.iloc[-1]],
                      color=color,label=exchange,lw=1.5)
        ax.set(title=symbol.upper(),xlabel='log10(1 + liquidation notional in USDT)')
    axes[0].set_ylabel('Fraction of records per original bin')
    axes[-1].legend();fig.suptitle('Liquidation notional distributions · full-row audit · Dec–Feb')
    fig.tight_layout();return fig
