"""Notebook-native figures for frozen EVT_log_cond model comparisons."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import evt_log_cond_modeling as m

COLORS = {'logistic':'#2867A3','lightgbm':'#C77D26','constant_train_prevalence':'#777777'}
LABELS = {'logistic':'Logistic regression','lightgbm':'LightGBM',
          'constant_train_prevalence':'Train-prevalence baseline'}


def style():
    plt.rcParams.update({'font.size':11,'figure.dpi':130,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
                         'axes.axisbelow':True})


def save(fig, name):
    folder=m.OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/name,dpi=150,bbox_inches='tight')
    return fig


def search_figure(horizon=20):
    style()
    f=pd.read_csv(m.OUT/'validation_search.csv')
    f=f[f.horizon_s.eq(horizon)&f.converged]
    fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for name,mark in [('logistic','o'),('lightgbm','s')]:
        curve=f[f.model.eq(name)].groupby('n_lags').validation_mse.min()
        ax[0].plot(curve.index,curve.values,label=LABELS[name],color=COLORS[name],marker=mark)
    ax[0].set(xlabel='Number of consecutive 5-second return lags',ylabel='Validation probability MSE',
              title='Best candidate for each lag count',xticks=list(m.LAG_COUNTS))
    ax[0].legend()
    curve=f[f.model.eq('lightgbm')].groupby('n_estimators').validation_mse.min()
    ax[1].plot(curve.index,curve.values,color=COLORS['lightgbm'],marker='s')
    ax[1].set(xlabel='Boosting iterations = number of trees',ylabel='Validation probability MSE',
              title='Best LightGBM candidate for each tree count',xticks=list(m.TREES))
    for a in ax:a.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle(f'Validation-only selection · future burst within {horizon}s\n30 Jan–13 Feb 2026 UTC · focused vertical scales; lower is better')
    return save(fig,'validation_search.png')


def mse_figure():
    style()
    f=pd.read_csv(m.OUT/'test_metrics.csv')
    ci=pd.read_csv(m.OUT/'paired_day_mse_intervals.csv').query("split=='test'").set_index('horizon_s').reindex(m.HORIZONS)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(len(m.HORIZONS))
    for i,name in enumerate(['logistic','lightgbm','constant_train_prevalence']):
        v=f[f.model.eq(name)].set_index('horizon_s').reindex(m.HORIZONS).mse
        ax[0].bar(x+(i-1)*.24,v,width=.23,color=COLORS[name],label=LABELS[name])
    ax[0].set(xticks=x,xticklabels=m.HORIZONS,xlabel='Forecast horizon, seconds',
              ylabel='Probability MSE (Brier score)',ylim=(0,None),title='Historical test · lower is better')
    ax[0].legend(loc='upper left',fontsize=9)
    d=ci.mse_lightgbm_minus_logistic.to_numpy()
    # Draw interval endpoints directly; a bootstrap percentile interval need not bracket the point.
    ax[1].hlines(x,ci.ci_low,ci.ci_high,color=COLORS['lightgbm'],lw=2)
    ax[1].scatter(d,x,color=COLORS['lightgbm'],marker='s',zorder=3)
    ax[1].axvline(0,color='#555555',ls='--',lw=1)
    ax[1].set(yticks=x,yticklabels=[f'{h}s' for h in m.HORIZONS],
              xlabel='MSE(LightGBM) − MSE(logistic)',title='Paired 95% intervals over UTC days')
    ax[1].invert_yaxis();ax[1].ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    fig.suptitle('Frozen models · 14–28 Feb 2026 UTC\nNegative difference favors LightGBM; 1,000 whole-day bootstrap draws, 15 days')
    return save(fig,'test_mse.png')


def calibration_figure():
    style()
    f=pd.read_csv(m.OUT/'probability_calibration.csv').query("split=='test'")
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,h in zip(axes.flat,m.HORIZONS):
        data=f[f.horizon_s.eq(h)]
        maximum=max(data.predicted.max(),data.observed.max())*1.12
        maximum=max(maximum,.01)
        ax.plot([0,maximum],[0,maximum],color='#777777',ls='--',lw=1)
        for name,mark in [('logistic','o'),('lightgbm','s')]:
            a=data[data.model.eq(name)].sort_values('predicted')
            ax.plot(a.predicted,a.observed,marker=mark,color=COLORS[name],label=LABELS[name])
        ax.set(title=f'Within {h}s',xlabel='Mean predicted probability',ylabel='Observed frequency',
               xlim=(0,maximum),ylim=(0,maximum))
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
    fig.suptitle('Probability calibration · historical test, frozen predictions\nUp to 10 probability-quantile bins per model; panel scales differ; no test recalibration')
    return save(fig,'probability_calibration.png')


if __name__=='__main__':
    for fn in (search_figure,mse_figure,calibration_figure):
        fig=fn();plt.close(fig)
