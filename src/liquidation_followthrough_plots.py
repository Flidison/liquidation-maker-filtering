"""Editable figures for the post-liquidation event study notebook."""
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import liquidation_followthrough as lf

LABELS = dict(zip(lf.CLASSES, ["Normal activity", "Liquidation burst", "Potential cascade"]))
COLORS = dict(zip(lf.CLASSES, ["#687D90", "#BC762A", "#295FA2"]))
ANCHOR_LABELS = {"onset": "Начало\nвсплеска", "first_detection": "Первое\nобнаружение",
                 "cascade_detection": "Подтверждение\nпотенц. каскада", "episode_end": "Окончание\nэпизода"}


def setup_style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
        "legend.frameon": False, "figure.facecolor": "white"})
    (lf.OUT/"figures").mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(lf.OUT/"figures"/f"{name}.png", bbox_inches="tight")
    return fig


def post_end_price():
    data = pd.read_csv(lf.OUT/"price_summary.csv")
    f = data.loc[data.phase.eq("validation") & data.anchor.eq("episode_end") & data.side.ne("ALL")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for i, symbol in enumerate(lf.lb.SYMBOLS):
        for j, h in enumerate(lf.HORIZONS):
            ax = axes[i,j]
            for offset, side, color, marker in [(-.13, "buy", "#295FA2", "o"), (.13, "sell", "#BC762A", "s")]:
                g = f.loc[f.symbol.eq(symbol) & f.horizon_s.eq(h) & f.side.eq(side)].set_index("classification").reindex(lf.CLASSES)
                m = g.mean_post_return_bps.to_numpy()
                y = np.arange(3)+offset
                ax.errorbar(m, y, xerr=np.vstack([m-g.raw_mean_ci95_low, g.raw_mean_ci95_high-m]),
                    fmt=marker, color=color, capsize=3, markersize=6, lw=1.3)
                for x, yy, n in zip(m, y, g.valid):
                    ax.annotate(f"n={int(n):,}", (x, yy), xytext=(0, -12 if side=="sell" else 10),
                                textcoords="offset points", ha="center", fontsize=8, color=color)
            ax.axvline(0, color="#303030", lw=1)
            ax.set_title(f"{symbol.upper()} · после окончания +{h} с")
            ax.set_yticks(range(3), [LABELS[c] for c in lf.CLASSES])
            ax.set_ylim(2.6, -.65)
            ax.set_xlabel("Средний log return, bps · + рост, − падение")
    fig.legend([Line2D([],[],color="#295FA2",marker="o",ls=""), Line2D([],[],color="#BC762A",marker="s",ls="")],
               ["Buy liquidations", "Sell liquidations"], loc="upper center", ncol=2, bbox_to_anchor=(.5,1.01))
    fig.suptitle("Validation: февраль 2026 · от последнего отмеченного окна\nИнтервалы: 95% bootstrap по UTC-дням, 2 000 повторов", y=1.10, fontsize=12)
    fig.tight_layout()
    return save(fig, "post_end_price_by_side")


def anchor_comparison():
    p = pd.read_csv(lf.OUT/"price_summary.csv")
    p = p.loc[p.side.eq("ALL") & p.horizon_s.eq(20)]
    anchors = ["onset", "first_detection", "cascade_detection", "episode_end"]
    fig, axes = plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for i, symbol in enumerate(lf.lb.SYMBOLS):
        for j, phase in enumerate(["train", "validation"]):
            ax = axes[i,j]
            for k, classification in enumerate(lf.CLASSES):
                g=p.loc[p.symbol.eq(symbol)&p.phase.eq(phase)&p.classification.eq(classification)].set_index("anchor")
                present=[a for a in anchors if a in g.index]
                x=np.array([anchors.index(a) for a in present])+(k-1)*.08
                g=g.loc[present]
                m=g.mean_post_signed_return_bps.to_numpy()
                ax.errorbar(x,m,yerr=np.vstack([m-g.signed_mean_ci95_low,g.signed_mean_ci95_high-m]),
                    marker=["o","s","D"][k],color=COLORS[classification],ls="none",
                    capsize=3,label=LABELS[classification])
            ax.axhline(0,color="#303030",lw=1)
            ax.set_xticks(range(4),[ANCHOR_LABELS[a] for a in anchors])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(f"{symbol.upper()} · {phase}")
            ax.set_ylabel("Средний signed return за +20 с, bps")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,1.02))
    fig.suptitle("+ продолжение по стороне ликвидаций; − движение против стороны\nИтоговый класс эпизода ретроспективен · интервалы: 95% bootstrap по дням",y=1.11,fontsize=12)
    fig.tight_layout()
    return save(fig,"anchor_comparison_20s")


def volatility_changes():
    vol=pd.read_csv(lf.OUT/"volatility_summary.csv")
    during=pd.read_csv(lf.OUT/"during_summary.csv")
    vol=vol.loc[vol.phase.eq("validation")&vol.side.eq("ALL")&vol.anchor.eq("episode_end")]
    during=during.loc[during.phase.eq("validation")&during.side.eq("ALL")]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for i,symbol in enumerate(lf.lb.SYMBOLS):
        for j,baseline in enumerate(["pre_anchor","pre_onset"]):
            a=np.array([[during.loc[during.symbol.eq(symbol)&during.classification.eq(c),"increased_share"].iloc[0] for c in lf.CLASSES],
                *[[vol.loc[vol.symbol.eq(symbol)&vol.baseline.eq(baseline)&vol.classification.eq(c)&vol.horizon_s.eq(h),"increased_share"].iloc[0] for c in lf.CLASSES] for h in lf.HORIZONS]])*100
            ax=axes[i,j]
            im=ax.imshow(a,cmap="Blues",vmin=0,vmax=100,aspect="auto")
            for y in range(3):
                for x in range(3):
                    ax.text(x,y,f"{a[y,x]:.1f}%",ha="center",va="center",color="white" if a[y,x]>65 else "#19324D",fontsize=12)
            ax.set_xticks(range(3),["Normal\nactivity","Liquidation\nburst","Potential\ncascade"])
            ax.set_yticks(range(3),["Во время эпизода*","После конца: 5 с","После конца: 20 с"])
            ax.set_title(symbol.upper()+" · "+("сразу до / после" if j==0 else "относительно до начала"))
            ax.grid(False)
    fig.colorbar(im,ax=axes,label="Эпизоды с увеличением RV, %",shrink=.85)
    fig.suptitle("Validation: доля эпизодов, в которых реализованная волатильность увеличилась\n* Во время: сравнение с предшествующим интервалом той же длительности",fontsize=12)
    return save(fig,"volatility_increase_shares")


def return_distribution():
    o=pl.scan_parquet(lf.OUT/"all_outcomes.parquet").filter((pl.col("phase")=="validation")&
        (pl.col("anchor")=="episode_end")).select("symbol","classification","horizon_s","post_signed_return_bps").collect().to_pandas()
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,sharey=True)
    for i,symbol in enumerate(lf.lb.SYMBOLS):
        for j,h in enumerate(lf.HORIZONS):
            ax=axes[i,j]
            arrays=[o.loc[o.symbol.eq(symbol)&o.horizon_s.eq(h)&o.classification.eq(c),"post_signed_return_bps"].dropna().to_numpy() for c in lf.CLASSES]
            artists=ax.boxplot(arrays,orientation="horizontal",whis=(5,95),showfliers=False,
                showmeans=True,patch_artist=True,meanprops={"marker":"D","markerfacecolor":"#252525","markeredgecolor":"#252525","markersize":4},
                medianprops={"color":"#222222","linewidth":1.5})
            for patch,c in zip(artists["boxes"],lf.CLASSES):
                patch.set_facecolor(COLORS[c]);patch.set_alpha(.6)
            ax.axvline(0,color="#303030",lw=1)
            ax.set_yticks([1,2,3],[LABELS[c] for c in lf.CLASSES])
            ax.set_ylim(3.6,.4)
            ax.set_title(f"{symbol.upper()} · +{h} с после окончания")
            ax.set_xlabel("Signed log return, bps · − отскок")
    fig.suptitle("Validation: распределение исходов, один вес на эпизод\nКоробка: 25–75%; линия: медиана; ромб: среднее; усы: 5–95%, хвосты скрыты",y=1.06,fontsize=12)
    fig.tight_layout()
    return save(fig,"post_end_return_distribution")


def selected_examples():
    o=pd.read_parquet(lf.OUT/"burst_episode_outcomes.parquet")
    o=o.loc[o.phase.eq("validation")&o.classification.eq(lf.CLASSES[2])&o.anchor.eq("episode_end")&o.horizon_s.eq(20)]
    ids=[]
    for symbol in lf.lb.SYMBOLS:
        g=o.loc[o.symbol.eq(symbol)]
        ids.append(g.loc[(g.post_signed_return_bps-g.post_signed_return_bps.median()).abs().idxmin(),"episode_id"])
    return ids


def episode_example(episode_id):
    cat=pd.read_parquet(lf.OUT/"episode_catalog.parquet")
    e=cat.set_index("episode_id").loc[episode_id]
    seconds=pl.read_parquet(lf.lb.OUT/f"seconds_{e.symbol}.parquet")
    path=lf.PricePath(seconds)
    onset=int(e.onset_timestamp//1_000_000)
    end=int(e.end_timestamp//1_000_000)
    times=np.arange(onset-25,end+26)
    mid=path.mid[times-path.start]
    price=(np.log(mid)-np.log(path.mid[onset-path.start]))*1e4
    rv=path.interval(times-5,times)["rv_bps"]
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    axes[0].plot(times-onset,price,color="#295FA2",lw=1.8)
    axes[0].axhline(0,color="#555555",lw=.8)
    axes[0].set_ylabel("Цена относительно начала, bps")
    axes[1].plot(times-onset,rv,color="#BC762A",lw=1.8)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Trailing 5-секундная RV, bps")
    axes[1].set_xlabel("Секунды относительно начала первого отмеченного окна")
    boundaries=[(0,"Начало","--"),(int(e.first_detection_timestamp//1_000_000)-onset,"Обнаружение",":"),
                (int(e.first_cascade_detection_timestamp//1_000_000)-onset,"Потенц. каскад","-."),(end-onset,"Конец","-")]
    for ax in axes:
        ax.axvspan(0,end-onset,color="#D6E2EF",alpha=.4)
        for x,label,ls in boundaries:
            ax.axvline(x,color="#454545",lw=.9,ls=ls,label=label if ax is axes[0] else None)
    for h,marker in [(5,"o"),(20,"s")]:
        i=end+h-(onset-25)
        for ax,values in zip(axes,[price,rv]):
            ax.scatter([end+h-onset],[values[i]],marker=marker,color="#222222",zorder=5)
        axes[0].annotate(f"конец +{h} с",(end+h-onset,price[i]),xytext=(3,8),textcoords="offset points",fontsize=9)
    axes[0].legend(ncol=4,loc="lower left",bbox_to_anchor=(0,1.02))
    date=pd.to_datetime(e.onset_timestamp,unit="us",utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.suptitle(f"{episode_id} · {e.symbol.upper()} · {e.exchange} {e.side} · {date}\nПример, ближайший к медиане signed return +20 с своего инструмента",y=1.08,fontsize=12)
    fig.tight_layout()
    return save(fig,f"followthrough_{episode_id}")
