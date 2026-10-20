"""Inspectable figures for the event-time warning notebook."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import polars as pl
import liquidation_early_warning as ew

FIG=ew.OUT/'figures'
BLUE='#2E5EAA';ORANGE='#CF782E';GRAY='#606974';OLIVE='#697A35'


def style():
    plt.rcParams.update({'figure.figsize':(12,5),'font.size':11,'axes.titlesize':13,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.2,'savefig.dpi':150,'figure.facecolor':'white'})
    FIG.mkdir(exist_ok=True)


def save(fig,name):
    fig.savefig(FIG/name,bbox_inches='tight')
    return fig


def transition_figure():
    style();fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for ax,s in zip(axes,ew.lb.SYMBOLS):
        d=pd.read_csv(ew.OUT/f'transition_matrix_{s}.csv')
        mat=d.pivot(index='previous_state',columns='next_state',values='probability').to_numpy()
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1)
        ax.grid(False)
        for i in range(8):
            for j in range(8):
                label=('0' if i==j else '<.01') if mat[i,j]<.005 else f'{mat[i,j]:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,
                        color='white' if mat[i,j]>.5 else '#222222')
        ticks=[f'{i}: {i:03b}' for i in range(8)]
        ax.set(xticks=np.arange(8),yticks=np.arange(8),xticklabels=ticks,yticklabels=ticks,
               xlabel='Next state',ylabel='Previous state',title=f'{s.upper()} · {d.fit_count.sum():,} fit transitions')
        ax.tick_params(axis='x',labelrotation=45)
    fig.colorbar(im,ax=axes,label='Conditional transition probability',shrink=.85)
    fig.suptitle('Multinomial logistic transition model · 1–7 December 2025 UTC\nBinary digits: 10 / 5 / 2.5 bps trend directions')
    return save(fig,'transition_matrices.png')


def performance_figure():
    style();m=pd.read_csv(ew.OUT/'model_metrics.csv').query("split=='test'")
    b=pd.read_csv(ew.OUT/'paired_day_bootstrap.csv').query("split=='test'").set_index('horizon_s').reindex(ew.HORIZONS)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(len(ew.HORIZONS))
    for model,label,color,marker in [('full','With W and L',BLUE,'o'),
             ('without_liquidity_score','Without W and L',ORANGE,'s'),
             ('constant_train_prevalence','Event prevalence',GRAY,'x')]:
        y=m[m.model.eq(model)].set_index('horizon_s').reindex(ew.HORIZONS).average_precision.to_numpy()
        axes[0].plot(x,y,label=label,color=color,marker=marker)
    axes[0].fill_between(x,b.full_ap_ci_low,b.full_ap_ci_high,color=BLUE,alpha=.14,label='95% day bootstrap, full')
    axes[0].set(ylabel='Average precision (AP)',ylim=(0,None),xlabel='Forecast horizon, seconds',
                xticks=x,xticklabels=ew.HORIZONS,title='Ranking of new bursts')
    axes[0].legend(fontsize=9)
    axes[1].errorbar(b.ap_difference,x,xerr=np.vstack([b.ap_difference-b.difference_ci_low,
        b.difference_ci_high-b.ap_difference]),fmt='o',color=BLUE,capsize=4)
    axes[1].axvline(0,color=GRAY,lw=1,ls='--')
    axes[1].set(yticks=x,yticklabels=[f'{h}s' for h in ew.HORIZONS],
                xlabel='AP with W,L minus AP without W,L',title='Incremental information from surprise')
    fig.suptitle('Historical test · 14–28 February 2026 UTC · both instruments\nPaired bootstrap of 15 UTC days; pointwise intervals, not independent-event confidence')
    return save(fig,'model_performance.png')


def calibration_figure():
    style();d=pd.read_csv(ew.OUT/'probability_calibration.csv').query("split=='test' and model=='full'")
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    colors=[BLUE,ORANGE,OLIVE,'#A95585',GRAY]
    high=max(d.predicted.max(),d.observed.max())*1.05
    ax.plot([0,high],[0,high],color=GRAY,ls='--',label='Ideal calibration')
    for h,c in zip(ew.HORIZONS,colors):
        g=d[d.horizon_s.eq(h)]
        ax.plot(g.predicted,g.observed,marker='o',color=c,label=f'{h}s')
    ax.set(xlim=(0,high),ylim=(0,high),xlabel='Mean model probability',
           ylabel='Observed fraction with a future burst',title='Probability calibration · historical test\nTen quantile bins per horizon; diagnostic only, no test-set calibration')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3)
    return save(fig,'probability_calibration.png')


def episode_figure(symbol):
    style()
    cat=pd.read_parquet(ew.OUT/f'forecast_episodes_{symbol}.parquet')
    test=cat[cat.split.eq('test')]
    selected=test[test.has_potential_cascade].sort_values('onset_timestamp')
    if selected.empty:selected=test.sort_values('onset_timestamp')
    event=selected.iloc[0]
    onset=int(event.onset_timestamp)//1_000_000
    start,end=onset-180,onset+180
    sec=pl.read_parquet(ew.lb.OUT/f'seconds_{symbol}.parquet',columns=['second','mid_close'])
    q=np.arange(start-ew.START,end-ew.START+1)
    mid=sec['mid_close'].to_numpy()[q-1]
    base=sec['mid_close'][onset-ew.START-1]
    f=pd.read_parquet(ew.OUT/f'market_states_{symbol}.parquet')
    f=f[f.timestamp.between(start*1_000_000,end*1_000_000)]
    p=pd.read_parquet(ew.OUT/'predictions.parquet',filters=[('symbol','==',symbol),('horizon_s','==',20),('model','==','full')])
    p=p[p.timestamp.between(start*1_000_000,end*1_000_000)]
    config=pd.read_json(ew.OUT/'frozen_models.json',typ='series').to_dict()
    threshold=next(m['threshold'] for m in config['models'] if m['variant']=='full' and m['horizon_s']==20)
    fig,ax=plt.subplots(4,1,figsize=(12,9),sharex=True,layout='constrained')
    ax[0].plot(np.arange(start,end+1)-onset,np.log(mid/base)*10000,color=BLUE)
    ax[0].set_ylabel('Midpoint change\nfrom onset, bps')
    x=f.timestamp.to_numpy()/1_000_000-onset
    ax[1].plot(x,f.rv_5s_bps/np.sqrt(5),color=BLUE,label='RMS 5s')
    ax[1].plot(x,f.rv_60s_bps/np.sqrt(60),color=ORANGE,label='RMS 60s')
    ax[1].set_ylabel('Realized RMS\nbps / √s');ax[1].legend(loc='upper left',ncol=2)
    ax[2].plot(x,f.liquidity_L,color=OLIVE,marker='.',ms=3)
    ax[2].set(ylabel='Liquidity L\nlow = rare path',ylim=(-.03,1.03))
    ax[3].scatter(p.timestamp.to_numpy()/1_000_000-onset,p.probability,color=BLUE,s=13,label='P(new burst within 20s)')
    ax[3].axhline(threshold,color=ORANGE,ls='--',label='Frozen validation threshold')
    ax[3].set(ylabel='Model probability',xlabel='Seconds relative to retrospective onset (0)')
    ax[3].yaxis.set_major_formatter(PercentFormatter(1));ax[3].legend(loc='upper left',ncol=2,fontsize=9)
    for a in ax:
        a.axvline(0,color=GRAY,lw=1)
        a.axvline(5,color=GRAY,lw=1,ls='--')
        a.axvspan(0,min(180,int(event.end_timestamp)//1_000_000-onset),color=GRAY,alpha=.09)
    date=pd.to_datetime(event.onset_timestamp,unit='us',utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')
    fig.suptitle(f'{symbol.upper()} · first potential-cascade episode in historical test\n{date} · solid: onset; dashed: first detection; dots: eligible DC predictions')
    return save(fig,f'example_{symbol}.png')


def all_figures():
    for fn in [transition_figure,performance_figure,calibration_figure]:
        fig=fn();plt.close(fig)
    for s in ew.lb.SYMBOLS:
        fig=episode_figure(s);plt.close(fig)


if __name__=='__main__':all_figures()
