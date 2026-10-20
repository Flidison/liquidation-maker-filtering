"""Editable Matplotlib figures for Liquidation Bursts.ipynb."""
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from liquidation_bursts import OUT, SYMBOLS, VALIDATION

COLORS = {"buy": "#167A6C", "sell": "#C2413A", "normal": "#74869A", "burst": "#CC7A20"}


def setup_style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18,
                         "legend.frameon": False, "figure.facecolor": "white"})
    (OUT / "figures").mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight")
    return fig


def daily_events():
    episodes = pd.read_csv(OUT / "episodes.csv")
    episodes["day"] = pd.to_datetime(episodes.first_detection_timestamp, unit="us", utc=True).dt.floor("D")
    days = pd.date_range("2025-12-15", "2026-02-28", tz="UTC")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey="row")
    for col, symbol in enumerate(SYMBOLS):
        e = episodes.loc[episodes.symbol == symbol]
        for row, label in enumerate(["All burst episodes", "Potential cascade episodes"]):
            f = e if row == 0 else e.loc[e.classification == "Potential liquidation cascade"]
            n = f.groupby("day").size().reindex(days, fill_value=0)
            ax = axes[row,col]
            ax.bar(days, n, width=.8, color=COLORS["burst"] if row==0 else "#7454A3")
            ax.axvline(pd.Timestamp("2026-02-01",tz="UTC"), color="#333333", ls="--", lw=1)
            ax.axvspan(pd.Timestamp("2026-02-01",tz="UTC"), days[-1]+pd.Timedelta(days=1), color="#DCE7F2", alpha=.3)
            ax.set_title(f"{symbol.upper()} · {label}")
            ax.set_ylabel("Episodes / day")
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle("Separate venue/side episodes · shaded period = validation", y=1.02)
    fig.tight_layout()
    return save(fig, "daily_episode_counts")


def market_association():
    data = pd.read_csv(OUT / "market_comparison.csv")
    fig, axes = plt.subplots(2,2,figsize=(11,6))
    order = [("train","Normal active"),("train","Burst (liquidations only)"),
             ("validation","Normal active"),("validation","Burst (liquidations only)")]
    for row,symbol in enumerate(SYMBOLS):
        part = data.loc[data.symbol == symbol].set_index(["phase","group"])
        for col,(metric,ylabel) in enumerate([
            ("median_abs_return_bps","Median |5-second log return|, bps"),
            ("median_trade_ratio","Median trade notional / past mean")]):
            ax=axes[row,col]
            values=[part.loc[key,metric] for key in order]
            bars=ax.bar(range(4),values,color=[COLORS["normal"],COLORS["burst"]]*2,width=.62)
            ax.bar_label(bars,fmt="%.2f",padding=3)
            ax.set_xticks(range(4),["Train\nnormal active","Train\nburst","Validation\nnormal active","Validation\nburst"])
            ax.set_ylabel(ylabel)
            ax.set_title(symbol.upper())
            ax.set_ylim(0,max(values)*1.22)
    fig.suptitle("Burst selection uses liquidations only · one observation per instrument and time",y=1.02)
    fig.tight_layout()
    return save(fig,"market_association")


def robustness():
    data=pd.read_csv(OUT/"sensitivity.csv")
    data=data.loc[(data.variant=="primary") & (data.offset_s==0) & (data.event_type=="potential_cascades")]
    count_max = max(1, data.episodes.max())
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,split in enumerate(["train","validation"]):
        selected=data.loc[data.phase==split]
        for col,(value,title) in enumerate([("episodes","Potential cascade episodes"),
                                           ("reference_match_share","Reference cascades matched, %")]):
            pivot=selected.pivot(index="baseline_s",columns="window_s",values=value).sort_index().sort_index(axis=1)
            a=pivot.to_numpy()*(100 if col else 1)
            ax=axes[row,col]
            ax.imshow(a,cmap="YlGnBu",vmin=0,vmax=100 if col else count_max,aspect="auto")
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    ax.text(j,i,f"{a[i,j]:.0f}"+("%" if col else ""),ha="center",va="center",
                            color="white" if a[i,j]>(65 if col else count_max*.65) else "#17324D",fontsize=12)
            ax.set_xticks(range(3),pivot.columns.astype(str))
            ax.set_yticks(range(3),(pivot.index//60).astype(str))
            ax.set_xlabel("Observation window, seconds")
            ax.set_ylabel("Historical baseline, minutes")
            ax.set_title(f"{split.title()} · {title}")
            ax.grid(False)
    fig.suptitle("Same calibration recipe; ±10 s episode overlap tolerance\n2/5 s windows require ≥10 s persistence; 10 s windows require ≥20 s", y=1.03)
    fig.tight_layout()
    return save(fig,"sensitivity_grid")


def selected_episodes():
    episodes=pd.read_csv(OUT/"episodes.csv")
    cascade=episodes.loc[episodes.classification=="Potential liquidation cascade"]
    chosen=cascade.sort_values("liquidation_notional",ascending=False).groupby(["symbol","phase"]).head(1)
    isolated=episodes.loc[(episodes.classification=="Liquidation burst") & (episodes.flagged_windows==1)
                          & (episodes.phase=="validation")]
    chosen=pd.concat([chosen,isolated.sort_values("liquidation_notional",ascending=False).groupby("symbol").head(1)])
    return chosen.sort_values(["classification","symbol","phase"]).reset_index(drop=True)


def episode_plot(episode):
    symbol=episode.symbol
    start=int(episode.onset_timestamp)-180_000_000
    end=int(episode.end_timestamp)+180_000_000
    market=pl.scan_parquet(OUT/f"market_{symbol}.parquet").filter(pl.col("timestamp").is_between(start,end)).collect().to_pandas()
    times=pd.to_datetime(market.timestamp,unit="us",utc=True)
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True,gridspec_kw={"height_ratios":[1,1,1.3,1.2]})
    for exchange in ("binance","bybit"):
        for side in ("buy","sell"):
            f=pl.scan_parquet(OUT/f"features_{symbol}_{exchange}_{side}.parquet").filter(
                pl.col("timestamp").is_between(start,end)).select("timestamp","liquidation_count","liquidation_notional").collect().to_pandas()
            x=pd.to_datetime(f.timestamp,unit="us",utc=True)
            style="-" if exchange=="binance" else "--"
            axes[0].step(x,f.liquidation_count,where="pre",color=COLORS[side],ls=style,lw=1.3,label=f"{exchange} {side}")
            axes[1].step(x,f.liquidation_notional/1e6,where="pre",color=COLORS[side],ls=style,lw=1.3)
    axes[0].set_ylabel("Liquidation records\nper 5 s")
    axes[0].legend(ncol=4,loc="upper left",fontsize=9)
    axes[1].set_ylabel("Liquidation notional\nUSDT million / 5 s")
    # Invalid/stale prices remain visible as gaps.
    axes[2].plot(times,market.mid_close.where(market.price_valid),color="#21496D",lw=1.4)
    axes[2].set_ylabel("Binance BBO mid\nUSDT")
    axes[3].bar(times-pd.Timedelta(seconds=2.5),market.trade_notional/1e6,width=4/86400,color="#74869A",alpha=.85,label="5 s trade notional")
    axes[3].plot(times,market.trade_historical_q95/1e6,color="#B6761A",lw=1,label="Historical q95")
    axes[3].set_ylabel("Binance trades\nUSDT million / 5 s")
    axes[3].legend(loc="upper left",fontsize=9)
    detection=pd.to_datetime(int(episode.first_detection_timestamp),unit="us",utc=True)
    onset=pd.to_datetime(int(episode.onset_timestamp),unit="us",utc=True)
    finish=pd.to_datetime(int(episode.end_timestamp),unit="us",utc=True)
    confirmation=(pd.to_datetime(int(episode.first_cascade_detection_timestamp),unit="us",utc=True)
                  if pd.notna(episode.first_cascade_detection_timestamp) else None)
    for ax in axes:
        ax.axvspan(onset,finish,color="#C8D2DC",alpha=.3)
        ax.axvline(detection,color="#2563A4",ls=":",lw=1.4)
        if confirmation is not None:
            ax.axvline(confirmation,color="#7454A3",ls="--",lw=1.4)
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5,maxticks=9))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[3].set_xlabel(f"UTC · {detection:%Y-%m-%d} · dotted blue = first burst detection; dashed purple = first cascade confirmation")
    fig.suptitle(f"{symbol.upper()} · {episode.exchange.upper()} {episode.side.upper()} · {episode.phase}\n"
                 f"{episode.classification} · {episode.episode_id} · {episode.liquidation_notional/1e6:.2f}m USDT in flagged windows",fontsize=13,y=1.01)
    fig.tight_layout()
    return save(fig,f"episode_{episode.episode_id}")
