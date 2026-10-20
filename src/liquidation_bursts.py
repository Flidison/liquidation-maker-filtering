"""Causal, interpretable liquidation-burst research. No model or trading signal.

All intervals are [t-window, t), and t is the first detection/availability time.
The notebook documents the methodology; this module contains reproducible computations.
"""
from pathlib import Path
import gc
import hashlib
import json
import math

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from research_paths import ROOT, resolve_legacy, DATA_ROOT
OUT = ROOT / "artifacts/results/bursts"
DATA = DATA_ROOT
AUDIT = ROOT / "artifacts/results/quality"
START = int(pd.Timestamp("2025-12-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp())
FREEZE = int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())
VALIDATION = int(pd.Timestamp("2026-02-01", tz="UTC").timestamp())
SYMBOLS = ("btcusdt", "ethusdt")
EXCHANGES = ("binance", "bybit")
SIDES = ("buy", "sell")
VERSION = "2026-10-11-bursts-v1"


def source_path(symbol, kind):
    filename = ("" if kind.startswith("bybit") else "perp_") + symbol + ".parquet"
    return DATA / kind / filename


def utc(seconds):
    return pd.to_datetime(seconds, unit="s", utc=True)


def prior_mean(x, bins, min_samples=None):
    return pl.Series(x).shift(1).rolling_mean(
        bins, min_samples=bins if min_samples is None else min_samples
    ).to_numpy()


def prior_quantile(x, bins, q, min_samples=None):
    # NaNs are missing observations, not zero returns/volume.
    s = pl.Series(x).fill_nan(None)
    return s.shift(1).rolling_quantile(
        q, window_size=bins, min_samples=bins if min_samples is None else min_samples,
        interpolation="linear",
    ).to_numpy()


def prior_nanmean(x, bins, min_samples):
    return pl.Series(x).fill_nan(None).shift(1).rolling_mean(
        bins, min_samples=min_samples
    ).to_numpy()


def verify_sources():
    """Reuse the full-row audit only after verifying all source bytes."""
    OUT.mkdir(exist_ok=True)
    manifest = pd.read_csv(AUDIT / "source_manifest.csv")
    summary = pd.read_csv(AUDIT / "summary.csv")
    checks = []
    for row in manifest.to_dict("records"):
        path = DATA.joinpath(*Path(row["source_file"]).parts[-2:])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Source changed since the full audit: {path}")
        if pq.ParquetFile(path).metadata.num_rows != row["rows"]:
            raise ValueError(f"Row count changed: {path}")
        checks.append({"source": str(path.relative_to(ROOT)), "rows": row["rows"],
                       "sha256": digest, "matches_full_audit": True})
    pd.DataFrame(checks).to_csv(OUT / "source_verification.csv", index=False)
    summary["source"] = summary["file"].map(lambda p: "/".join(Path(p).parts[-2:]))
    summary.to_csv(OUT / "quality_summary.csv", index=False)
    return summary


def build_second_cache(symbol):
    """Only completed-second audit aggregates; no event-evidence hindsight fields."""
    target = OUT / f"seconds_{symbol}.parquet"
    # Rebuild this small cache on every pipeline run after source verification.
    n = END - START
    seconds = np.arange(START, END, dtype=np.int64)
    trade_path = AUDIT / f"data__binance_trades__perp_{symbol}" / "seconds.parquet"
    bbo_path = AUDIT / f"data__binance_booktickers__perp_{symbol}" / "seconds.parquet"
    tr = pl.read_parquet(trade_path, columns=["second", "count", "notional"])
    bbo = pl.read_parquet(bbo_path, columns=["second", "count", "last_price"])
    manifest = pd.read_csv(AUDIT / "source_manifest.csv")
    for kind, frame in [("binance_trades", tr), ("binance_booktickers", bbo)]:
        match = manifest[manifest.source_file.str.endswith(f"{kind}/perp_{symbol}.parquet")]
        assert frame["count"].sum() == int(match.iloc[0]["rows"])
        assert frame["second"].n_unique() == frame.height
        assert frame["second"].is_sorted()
    volume = np.zeros(n, dtype=np.float64)
    trade_count = np.zeros(n, dtype=np.int64)
    ti = tr["second"].to_numpy() - START
    volume[ti] = tr["notional"].to_numpy()
    trade_count[ti] = tr["count"].to_numpy()
    mid = np.full(n, np.nan)
    bi = bbo["second"].to_numpy() - START
    mid[bi] = bbo["last_price"].to_numpy()
    # At second s+1, last_price[s] is fully available. Fill forward only.
    quote_index = np.maximum.accumulate(np.where(np.isfinite(mid), np.arange(n), -1))
    filled_mid = mid[np.maximum(quote_index, 0)]
    filled_mid[quote_index < 0] = np.nan
    age_upper = np.where(quote_index >= 0, np.arange(n) + 1 - quote_index, n)
    trade_index = np.maximum.accumulate(np.where(trade_count > 0, np.arange(n), -1))
    trade_age = np.arange(n) + 1 - trade_index
    frame = pl.DataFrame({"second": seconds, "trade_notional": volume,
                          "trade_count": trade_count, "mid_close": filled_mid,
                          "quote_age_upper_s": age_upper.astype(np.int32),
                          "trade_age_upper_s": trade_age.astype(np.int32)})
    frame.write_parquet(target, compression="zstd")
    return frame


def load_liquidations(symbol, seconds):
    frames, quality = {}, []
    for exchange in EXCHANGES:
        f = pl.read_parquet(source_path(symbol, f"{exchange}_liquidations")).to_pandas()
        f["source_row"] = np.arange(len(f), dtype=np.int64)
        raw_columns = ["timestamp", "ticker", "side", "price", "amount"]
        duplicate = f.duplicated(raw_columns, keep="first")
        clear = (f[raw_columns].isna().all(axis=1) | f.timestamp.isna()
                 | (f.timestamp < START * 1_000_000) | (f.timestamp >= END * 1_000_000)
                 | ~np.isfinite(f.price) | (f.price <= 0)
                 | ~np.isfinite(f.amount) | (f.amount < 0) | ~f.side.isin(SIDES))
        if clear.any():
            raise ValueError("Unexpected invalid liquidation rows; refresh the full audit")
        f["raw_timestamp"] = f.timestamp
        f["effective_timestamp"] = f.timestamp + (200_000 if exchange == "bybit" else 0)
        f["notional"] = f.price * f.amount
        f["duplicate_extra_copy"] = duplicate
        # Strictly preceding completed second: its last quote was known at the event.
        idx = f.effective_timestamp.to_numpy() // 1_000_000 - START - 1
        in_range = (idx >= 0) & (idx < seconds.height)
        safe_idx = np.clip(idx, 0, seconds.height - 1)
        ref = seconds["mid_close"].to_numpy()[safe_idx]
        fresh = in_range & (seconds["quote_age_upper_s"].to_numpy()[safe_idx] <= 2)
        ratio = f.price.to_numpy() / ref
        f["suspect_price"] = fresh & ((ratio < 0.5) | (ratio > 1.5))
        f["reference_mid"] = np.where(fresh, ref, np.nan)
        quality.append({"symbol": symbol, "exchange": exchange, "rows": len(f),
                        "missing_cells": int(f[raw_columns].isna().sum().sum()),
                        "invalid_rows": int(clear.sum()), "duplicate_extra_copies": int(duplicate.sum()),
                        "timestamp_reversals": int((f.timestamp.diff() < 0).sum()),
                        "suspect_price_rows": int(f.suspect_price.sum()),
                        "suspect_notional": float(f.loc[f.suspect_price, "notional"].sum()),
                        "price_reference_missing": int((~fresh).sum())})
        f = f.sort_values("effective_timestamp", kind="stable").reset_index(drop=True)
        frames[exchange] = f
    return frames, quality


def market_windows(seconds, window=5, baseline=1200, offset=0):
    if baseline % window:
        raise ValueError("Baseline must consist of complete observation windows")
    n = (seconds.height - offset) // window
    lo, hi = offset, offset + n * window
    t = START + offset + np.arange(1, n + 1, dtype=np.int64) * window
    volume = seconds["trade_notional"].to_numpy()[lo:hi].reshape(n, window).sum(axis=1)
    counts = seconds["trade_count"].to_numpy()[lo:hi].reshape(n, window).sum(axis=1)
    ends = lo + np.arange(1, n + 1) * window - 1
    mid = seconds["mid_close"].to_numpy()[ends]
    ages = seconds["quote_age_upper_s"].to_numpy()
    end_age = ages[ends]
    prior_mid = np.r_[np.nan if lo == 0 else seconds["mid_close"][lo - 1], mid[:-1]]
    prior_age = np.r_[np.inf if lo == 0 else ages[lo - 1], end_age[:-1]]
    gap = ages[lo:hi].reshape(n, window).max(axis=1) > 2
    trade_gap = (seconds["trade_age_upper_s"].to_numpy()[lo:hi]
                 .reshape(n, window).max(axis=1) > 10)
    valid_price = np.isfinite(prior_mid) & np.isfinite(mid) & (prior_age <= 2) & ~gap
    log_return = np.where(valid_price, np.log(mid / prior_mid), np.nan)
    valid_volume = np.where(trade_gap, np.nan, volume)
    bins = baseline // window
    enough = math.ceil(0.9 * bins)
    mean_volume = prior_nanmean(valid_volume, bins, enough)
    volume_q95 = prior_quantile(valid_volume, bins, .95, enough)
    ret_q99 = prior_quantile(np.abs(log_return), bins, .99, enough)
    # Full elapsed history is mandatory even when missing observations are tolerated.
    ready = np.arange(n) >= bins
    for a in [mean_volume, volume_q95, ret_q99]:
        a[~ready] = np.nan
    ratio = volume / np.maximum(mean_volume, 1.)
    strong_trade = (~trade_gap & ready & (volume > volume_q95) & (ratio >= 2.))
    strong_price = (valid_price & ready & (np.abs(log_return) > ret_q99)
                    & (np.abs(log_return) >= 2e-4 * np.sqrt(window / 5)))
    return pd.DataFrame({"timestamp": t * 1_000_000,
                         "window_start_timestamp": (t - window) * 1_000_000,
                         "trade_notional": volume, "trade_count": counts,
                         "mid_start": prior_mid, "mid_close": mid,
                         "quote_age_upper_s": end_age,
                         "price_valid": valid_price, "trade_gap": trade_gap,
                         "log_return": log_return, "log_return_bps": log_return * 10000,
                         "trade_baseline_mean": mean_volume, "trade_historical_q95": volume_q95,
                         "trade_intensity_ratio": ratio, "abs_return_historical_q99": ret_q99,
                         "strong_trade": strong_trade, "strong_price": strong_price,
                         "history_ready": ready})


def liquidation_windows(frame, side, n, window=5, offset=0, variant="primary"):
    use = frame.side.eq(side)
    if variant == "liq_quality":
        use &= ~frame.duplicate_extra_copy & ~frame.suspect_price
    f = frame.loc[use]
    indices = (f.effective_timestamp.to_numpy() // 1_000_000 - START - offset) // window
    keep = (indices >= 0) & (indices < n)
    indices = indices[keep].astype(np.int64)
    counts = np.bincount(indices, minlength=n).astype(np.int32)
    volume = np.bincount(indices, weights=f.notional.to_numpy()[keep], minlength=n)
    suspect = np.bincount(indices, weights=f.suspect_price.to_numpy()[keep], minlength=n)
    return counts, volume, suspect.astype(np.int32)


def calibrate(counts, volume, timestamps, window, symbol, exchange, side):
    active = (timestamps < FREEZE * 1_000_000) & (counts > 0)
    count_q = np.quantile(counts[active], [.5, .9, .95, .99])
    volume_q = np.quantile(volume[active], [.5, .9, .95, .99])
    return {"symbol": symbol, "exchange": exchange, "side": side, "window_s": window,
            "calibration_active_windows": int(active.sum()),
            "count_p50": float(count_q[0]), "count_p90": float(count_q[1]),
            "count_p95": float(count_q[2]), "count_p99": float(count_q[3]),
            "notional_p50": float(volume_q[0]), "notional_p90": float(volume_q[1]),
            "notional_p95": float(volume_q[2]), "notional_p99": float(volume_q[3]),
            "minimum_count": max(2, math.ceil(3 * window / 5), math.ceil(count_q[1])),
            "minimum_notional": max(25_000 * window / 5, float(volume_q[1])),
            "ratio_threshold": 5., "historical_quantile": .99,
            "minimum_consecutive_windows": max(2, math.ceil(10 / window)),
            "frozen_at_timestamp": FREEZE * 1_000_000}


def liquidation_features(counts, volume, suspect, market, baseline, window):
    bins = baseline // window
    cmean = prior_mean(counts.astype(float), bins)
    vmean = prior_mean(volume, bins)
    cq = prior_quantile(counts.astype(float), bins, .99)
    vq = prior_quantile(volume, bins, .99)
    cr = counts / np.maximum(cmean, max(1., window / 5))
    vr = volume / np.maximum(vmean, 1000. * window / 5)
    # Raw ratios remain undefined when the baseline is zero; no inf-based flags.
    craw = np.divide(counts, cmean, out=np.full(len(counts), np.nan), where=cmean > 0)
    vraw = np.divide(volume, vmean, out=np.full(len(volume), np.nan), where=vmean > 0)
    return pd.DataFrame({"liquidation_count": counts, "liquidation_notional": volume,
                         "suspect_price_count": suspect,
                         "count_baseline_mean": cmean, "notional_baseline_mean": vmean,
                         "count_historical_q99": cq, "notional_historical_q99": vq,
                         "count_intensity_ratio": cr, "notional_intensity_ratio": vr,
                         "raw_count_ratio": craw, "raw_notional_ratio": vraw,
                         "near_zero_baseline": (cmean < max(1., window / 5))
                                               | (vmean < 1000. * window / 5)})


def classify(features, market, threshold, side, apply_freeze=True, ratio=None, floor_q=.9):
    ratio = threshold["ratio_threshold"] if ratio is None else ratio
    count_floor = threshold["minimum_count"]
    volume_floor = threshold["minimum_notional"]
    if floor_q == .95:
        count_floor = max(count_floor, math.ceil(threshold["count_p95"]))
        volume_floor = max(volume_floor, threshold["notional_p95"])
    count_jump = ((features.count_intensity_ratio >= ratio)
                  & (features.liquidation_count > features.count_historical_q99))
    volume_jump = ((features.notional_intensity_ratio >= ratio)
                   & (features.liquidation_notional > features.notional_historical_q99))
    eligible = market.history_ready & np.isfinite(features.count_baseline_mean)
    if apply_freeze:
        eligible &= market.timestamp >= FREEZE * 1_000_000
    burst = (eligible & (features.liquidation_count >= count_floor)
             & (features.liquidation_notional >= volume_floor) & (count_jump | volume_jump)).to_numpy()
    indices = np.arange(len(burst))
    run = np.where(burst, indices - np.maximum.accumulate(np.where(~burst, indices, -1)), 0)
    aligned = market.log_return.to_numpy() * (1 if side == "buy" else -1) > 0
    persistent = run >= threshold["minimum_consecutive_windows"]
    cascade = (burst & persistent & aligned & market.strong_trade.to_numpy()
               & market.strong_price.to_numpy())
    return pd.DataFrame({"eligible": eligible, "count_anomaly": count_jump,
                         "notional_anomaly": volume_jump, "burst": burst,
                         "consecutive_burst_windows": run.astype(np.int32),
                         "persistent": persistent, "direction_aligned": aligned,
                         "potential_cascade": cascade,
                         "classification": np.where(~eligible, "Not evaluated",
                             np.where(cascade, "Potential liquidation cascade",
                                      np.where(burst, "Liquidation burst", "Normal activity")))})


def make_episodes(flagged, window):
    if flagged.empty:
        return pd.DataFrame()
    f = flagged.copy()
    # No gap bridging: sums cannot double-count an overlapping rolling window.
    new = (f.timestamp.diff().ne(window * 1_000_000) | f.phase.ne(f.phase.shift()))
    f["episode_group"] = new.cumsum()
    rows = []
    for _, g in f.groupby("episode_group", sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        peak = g.loc[g.liquidation_notional.idxmax()]
        confirmation = g.loc[g.potential_cascade]
        rows.append({"symbol": first.symbol, "exchange": first.exchange, "side": first.side,
                     "phase": first.phase, "onset_timestamp": int(first.window_start_timestamp),
                     "first_detection_timestamp": int(first.timestamp),
                     "first_cascade_detection_timestamp": (int(confirmation.timestamp.iloc[0])
                                                              if len(confirmation) else None),
                     "end_timestamp": int(last.timestamp),
                     "classification": ("Potential liquidation cascade" if len(confirmation)
                                        else "Liquidation burst"),
                     "duration_s": (last.timestamp - first.window_start_timestamp) / 1_000_000,
                     "flagged_windows": len(g), "cascade_windows": int(g.potential_cascade.sum()),
                     "liquidation_count": int(g.liquidation_count.sum()),
                     "liquidation_notional": float(g.liquidation_notional.sum()),
                     "suspect_price_count": int(g.suspect_price_count.sum()),
                     "max_count_intensity_ratio": float(g.count_intensity_ratio.max()),
                     "max_notional_intensity_ratio": float(g.notional_intensity_ratio.max()),
                     "episode_trade_notional": float(g.trade_notional.sum()),
                     "episode_log_return": (float(np.log(last.mid_close / first.mid_start))
                                            if g.price_valid.all() else np.nan),
                     "peak_timestamp": int(peak.timestamp),
                     "peak_liquidation_count": int(peak.liquidation_count),
                     "peak_liquidation_notional": float(peak.liquidation_notional),
                     "peak_trade_notional": float(peak.trade_notional),
                     "peak_log_return": float(peak.log_return)})
    result = pd.DataFrame(rows)
    result["first_cascade_detection_timestamp"] = pd.array(
        result.first_cascade_detection_timestamp, dtype="Int64")
    return result


def phase(t):
    return np.where(t < FREEZE * 1_000_000, "calibration",
                    np.where(t < VALIDATION * 1_000_000, "train", "validation"))


def calibrate_and_explore():
    """Only rows before FREEZE enter calibration and threshold diagnostics."""
    thresholds, exploration, quality = [], [], []
    for symbol in SYMBOLS:
        seconds = build_second_cache(symbol)
        liquidations, qc = load_liquidations(symbol, seconds)
        quality.extend(qc)
        for window in (2, 5, 10):
            market = market_windows(seconds.head(FREEZE - START), window, 1200)
            # Explicitly truncate before any threshold inspection.
            ncal = int((market.timestamp < FREEZE * 1_000_000).sum())
            market_cal = market.iloc[:ncal].copy()
            for exchange, liq in liquidations.items():
                for side in SIDES:
                    count, volume, suspect = liquidation_windows(liq, side, len(market), window)
                    th = calibrate(count, volume, market.timestamp.to_numpy(), window,
                                   symbol, exchange, side)
                    thresholds.append(th)
                    if window == 5:
                        feat = liquidation_features(count[:ncal], volume[:ncal], suspect[:ncal],
                                                    market_cal, 1200, window)
                        for ratio in (3., 5., 10.):
                            for fq in (.9, .95):
                                flags = classify(feat, market_cal, th, side, False, ratio, fq)
                                exploration.append({"symbol": symbol, "exchange": exchange,
                                    "side": side, "ratio": ratio, "absolute_floor_quantile": fq,
                                    "eligible_windows": int(flags.eligible.sum()),
                                    "active_windows": int(((count[:ncal] > 0) & flags.eligible).sum()),
                                    "burst_windows": int(flags.burst.sum()),
                                    "persistent_windows": int((flags.persistent & flags.burst).sum())})
            del market
        del seconds, liquidations
        gc.collect()
    pd.DataFrame(thresholds).to_csv(OUT / "thresholds.csv", index=False)
    pd.DataFrame(exploration).to_csv(OUT / "train_threshold_exploration.csv", index=False)
    pd.DataFrame(quality).to_csv(OUT / "liquidation_quality.csv", index=False)
    config = {"version": VERSION, "observation_window_s": 5, "baseline_s": 1200,
              "calibration_start": str(utc(START)), "threshold_freeze": str(utc(FREEZE)),
              "validation_start": str(utc(VALIDATION)), "interval": "[t-window,t)",
              "ratio_threshold": 5, "absolute_floor_quantile": .9,
              "minimum_trade_ratio": 2, "trade_quantile": .95, "return_quantile": .99,
              "minimum_abs_return_bps_5s": 2, "max_quote_age_upper_s": 2,
              "minimum_persistence_s": 10, "minimum_persistence_windows": 2,
              "near_zero_count_floor_5s": 1, "near_zero_notional_floor_5s": 1000,
              "choice_basis": "Interpretability and calibration-window rarity only; no market outcomes or validation tuning."}
    (OUT / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return pd.DataFrame(thresholds), pd.DataFrame(exploration)


def run_scenario(window=5, baseline=1200, offset=0, variant="primary", save=False):
    thresholds = pd.read_csv(OUT / "thresholds.csv")
    events, episodes, summary, comparison, daily = [], [], [], [], []
    for symbol in SYMBOLS:
        seconds = pl.read_parquet(OUT / f"seconds_{symbol}.parquet")
        liqs, _ = load_liquidations(symbol, seconds)
        market = market_windows(seconds, window, baseline, offset)
        market["symbol"] = symbol
        market["phase"] = phase(market.timestamp.to_numpy())
        if save:
            pl.from_pandas(market).write_parquet(OUT / f"market_{symbol}.parquet", compression="zstd")
        any_active = np.zeros(len(market), bool)
        any_burst = np.zeros(len(market), bool)
        for exchange, liq in liqs.items():
            for side in SIDES:
                th = thresholds[(thresholds.symbol == symbol) & (thresholds.exchange == exchange)
                                & (thresholds.side == side) & (thresholds.window_s == window)].iloc[0].to_dict()
                counts, volume, suspect = liquidation_windows(liq, side, len(market), window, offset, variant)
                features = liquidation_features(counts, volume, suspect, market, baseline, window)
                flags = classify(features, market, th, side)
                any_active |= counts > 0
                any_burst |= flags.burst.to_numpy()
                for split in ("train", "validation"):
                    selected = market.phase.eq(split) & flags.eligible
                    summary.append({"symbol": symbol, "exchange": exchange, "side": side,
                        "phase": split, "eligible_windows": int(selected.sum()),
                        "active_windows": int((selected & (counts > 0)).sum()),
                        "burst_windows": int((selected & flags.burst).sum()),
                        "potential_cascade_windows": int((selected & flags.potential_cascade).sum()),
                        "normal_windows": int((selected & ~flags.burst).sum()),
                        "price_invalid_windows": int((selected & ~market.price_valid).sum()),
                        "near_zero_baseline_windows": int((selected & features.near_zero_baseline).sum()),
                        "near_zero_baseline_bursts": int((selected & flags.burst & features.near_zero_baseline).sum()),
                        "aligned_burst_windows": int((selected & flags.burst & flags.direction_aligned).sum()),
                        "strong_trade_burst_windows": int((selected & flags.burst & market.strong_trade).sum()),
                        "strong_price_burst_windows": int((selected & flags.burst & market.strong_price).sum()),
                        "persistent_burst_windows": int((selected & flags.burst & flags.persistent).sum())})
                if save:
                    # Market columns are stored once per symbol; join on timestamp for complete features.
                    full_features = pd.concat([market[["timestamp", "phase"]], features, flags], axis=1)
                    pl.from_pandas(full_features).write_parquet(
                        OUT / f"features_{symbol}_{exchange}_{side}.parquet", compression="zstd")
                f = pd.concat([market.loc[flags.burst].copy(), features.loc[flags.burst],
                               flags.loc[flags.burst]], axis=1)
                f["exchange"], f["side"] = exchange, side
                if len(f):
                    ep = make_episodes(f, window)
                    episodes.append(ep)
                    if save:
                        events.append(f)
                del features, flags
                gc.collect()
        if save:
            # Deduplicate market observations across liquidation venues and sides.
            eligible = (market.phase != "calibration") & market.history_ready
            for split in ("train", "validation"):
                for group, selection in [("Normal active", any_active & ~any_burst),
                                         ("Burst (liquidations only)", any_burst)]:
                    g = market.loc[eligible & market.phase.eq(split) & selection]
                    comparison.append({"symbol": symbol, "phase": split, "group": group,
                        "windows": len(g), "price_valid_windows": int(g.price_valid.sum()),
                        "median_abs_return_bps": float(g.log_return_bps.abs().median()),
                        "p95_abs_return_bps": float(g.log_return_bps.abs().quantile(.95)),
                        "median_trade_notional": float(g.trade_notional.median()),
                        "median_trade_ratio": float(g.trade_intensity_ratio.median()),
                        "strong_trade_share": float(g.strong_trade.mean()),
                        "strong_price_share": float(g.strong_price.mean())})
                    day = g.assign(day=utc(g.timestamp.to_numpy() // 1_000_000).floor("D"),
                                   abs_return_bps=g.log_return_bps.abs())
                    d = day.groupby("day").agg(windows=("timestamp", "size"),
                        abs_return_bps=("abs_return_bps", "median"),
                        trade_ratio=("trade_intensity_ratio", "median")).reset_index()
                    d["symbol"], d["phase"], d["group"] = symbol, split, group
                    daily.append(d)
        del seconds, liqs, market
        gc.collect()
    ep = pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame()
    if len(ep):
        ep = ep.sort_values(["first_detection_timestamp", "symbol", "exchange", "side"]).reset_index(drop=True)
        ep.insert(0, "episode_id", [f"LB{i:06d}" for i in range(1, len(ep)+1)])
    if save:
        ev = pd.concat(events, ignore_index=True).sort_values(["timestamp", "symbol", "exchange", "side"])
        pl.from_pandas(ev).write_parquet(OUT / "event_windows.parquet", compression="zstd")
        ev.assign(datetime_utc=pd.to_datetime(ev.timestamp, unit="us", utc=True)).to_csv(
            OUT / "event_windows.csv", index=False)
        ep.to_csv(OUT / "episodes.csv", index=False)
        pl.from_pandas(ep).write_parquet(OUT / "episodes.parquet", compression="zstd")
        pd.DataFrame(summary).to_csv(OUT / "classification_summary.csv", index=False)
        pd.DataFrame(comparison).to_csv(OUT / "market_comparison.csv", index=False)
        pd.concat(daily).to_csv(OUT / "daily_market_comparison.csv", index=False)
    return ep, pd.DataFrame(summary)


def episode_match(reference, alternative, tolerance_s=10):
    """Fraction with a same-stream interval overlap, allowing fixed 10-second jitter."""
    if reference.empty:
        return np.nan
    matched = 0
    keys = ["symbol", "exchange", "side", "phase"]
    for key, ref in reference.groupby(keys):
        alt = alternative
        for col, value in zip(keys, key):
            alt = alt.loc[alt[col] == value]
        if alt.empty:
            continue
        alt = alt.sort_values("onset_timestamp")
        starts = alt.onset_timestamp.to_numpy()
        ends = alt.end_timestamp.to_numpy()
        for row in ref.itertuples():
            j = np.searchsorted(ends, row.onset_timestamp - tolerance_s * 1_000_000, side="left")
            matched += int(j < len(alt) and starts[j] <= row.end_timestamp + tolerance_s * 1_000_000)
    return matched / len(reference)


def sensitivity():
    base = pd.read_csv(OUT / "episodes.csv")
    rows = []
    configurations = [(w, b, 0, "primary") for w in (2,5,10) for b in (600,1200,2400)]
    configurations += [(5,1200,2,"primary"), (5,1200,0,"liq_quality")]
    for window, baseline, offset, variant in configurations:
        print(f"Sensitivity: window={window}, baseline={baseline}, offset={offset}, {variant}", flush=True)
        ep = base if (window,baseline,offset,variant) == (5,1200,0,"primary") else run_scenario(
            window, baseline, offset, variant)[0]
        for split in ("train", "validation"):
            for label in ("all_bursts", "potential_cascades"):
                ref = base.loc[base.phase == split]
                alt = ep.loc[ep.phase == split]
                if label == "potential_cascades":
                    ref = ref.loc[ref.classification == "Potential liquidation cascade"]
                    alt = alt.loc[alt.classification == "Potential liquidation cascade"]
                rows.append({"window_s": window, "baseline_s": baseline, "offset_s": offset,
                             "variant": variant, "phase": split, "event_type": label,
                             "episodes": len(alt), "reference_episodes": len(ref),
                             "reference_match_share": episode_match(ref, alt),
                             "alternative_match_share": episode_match(alt, ref),
                             "flagged_duration_s": float(alt.duration_s.sum()),
                             "minimum_persistence_s": window * max(2, math.ceil(10/window))})
        pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    return pd.DataFrame(rows)


def read_raw_window(symbol, kind, start_us, end_us):
    return pl.scan_parquet(source_path(symbol, kind)).filter(
        (pl.col("timestamp") >= int(start_us)) & (pl.col("timestamp") < int(end_us))
    ).collect(engine="streaming")


def reconcile_and_check_duplicates():
    """Independent raw-window reconstruction, including matched historical windows."""
    events = pl.read_parquet(OUT / "event_windows.parquet").to_pandas()
    ep = pd.read_csv(OUT / "episodes.csv")
    checks, duplicates = [], []
    for symbol in SYMBOLS:
        # Distinct most intense cascade and burst per split; deterministic selection.
        selected = ep.loc[ep.symbol == symbol].sort_values("liquidation_notional", ascending=False)
        selected = selected.groupby(["phase", "classification"], sort=True).head(2)
        for row in selected.itertuples():
            t = int(row.peak_timestamp)
            w = events.loc[(events.timestamp == t) & events.symbol.eq(symbol)
                           & events.exchange.eq(row.exchange) & events.side.eq(row.side)].iloc[0]
            tr = read_raw_window(symbol, "binance_trades", t - 1205_000_000, t)
            current = tr.filter(pl.col("timestamp") >= t - 5_000_000)
            raw_notional = current.select((pl.col("price") * pl.col("amount")).sum()).item()
            assert np.isclose(raw_notional, w.trade_notional, rtol=1e-10)
            # Last raw BBO strictly before each endpoint, obtained from a bounded earlier slice.
            bbo = read_raw_window(symbol, "binance_booktickers", t-600_000_000, t)
            mids = []
            for end in (t-5_000_000, t):
                last = bbo.filter(pl.col("timestamp") < end).tail(1)
                mids.append((last["bid_price"][0] + last["ask_price"][0])/2)
            raw_return = np.log(mids[1] / mids[0])
            if w.price_valid:
                assert np.isclose(raw_return, w.log_return, atol=1e-13)
            delay = 200_000 if row.exchange == "bybit" else 0
            liq = read_raw_window(symbol, row.exchange + "_liquidations", t-5_000_000-delay, t-delay)
            liq = liq.filter(pl.col("side") == row.side)
            ln = liq.select((pl.col("price") * pl.col("amount")).sum()).item()
            assert liq.height == w.liquidation_count
            assert np.isclose(ln, w.liquidation_notional, rtol=1e-10)
            hist_liq = read_raw_window(symbol, row.exchange+"_liquidations", t-1205_000_000-delay, t-5_000_000-delay).filter(pl.col("side") == row.side)
            assert np.isclose(hist_liq.height/240, w.count_baseline_mean, atol=1e-10)
            checks.append({"episode_id": row.episode_id, "symbol": symbol, "timestamp": t,
                           "trade_notional_difference": raw_notional - w.trade_notional,
                           "log_return_difference": raw_return - w.log_return,
                           "liquidation_count_difference": liq.height - w.liquidation_count,
                           "liquidation_notional_difference": ln - w.liquidation_notional,
                           "count_baseline_difference": hist_liq.height/240 - w.count_baseline_mean})
            # Hypothetical collapse of identical records. This is not proof of bad executions.
            unique = tr.unique(subset=["timestamp","ticker","side","price","amount"])
            bucket = ((pl.col("timestamp")-(t-1205_000_000))//5_000_000).alias("bin")
            values = []
            for data in (tr, unique):
                b = data.with_columns(bucket).group_by("bin").agg((pl.col("price")*pl.col("amount")).sum().alias("notional"))
                a = np.zeros(241)
                a[b["bin"].to_numpy()] = b["notional"].to_numpy()
                mean, q95 = a[:240].mean(), np.quantile(a[:240], .95)
                values.append((a[-1], a[-1]/max(mean,1), (a[-1] > q95) and (a[-1] >= 2*max(mean,1))))
            duplicates.append({"episode_id": row.episode_id, "symbol": symbol, "phase": row.phase,
                               "classification": row.classification,
                               "raw_rows_20m5s": tr.height, "unique_rows_20m5s": unique.height,
                               "raw_trade_notional_5s": values[0][0], "dedup_trade_notional_5s": values[1][0],
                               "raw_trade_ratio": values[0][1], "dedup_trade_ratio": values[1][1],
                               "raw_strong_trade": values[0][2], "dedup_strong_trade": values[1][2]})
            del tr, unique, bbo
            gc.collect()
    pd.DataFrame(checks).to_csv(OUT / "raw_reconciliation.csv", index=False)
    pd.DataFrame(duplicates).to_csv(OUT / "trade_duplicate_sensitivity_sample.csv", index=False)
    return pd.DataFrame(checks), pd.DataFrame(duplicates)


def check_all_cascade_trade_confirmations():
    """Alternate exact-row-dedup hypothesis at each first cascade confirmation."""
    episodes = pd.read_csv(OUT / "episodes.csv")
    episodes = episodes.loc[episodes.classification == "Potential liquidation cascade"]
    rows = []
    for episode in episodes.itertuples():
        t = int(episode.first_cascade_detection_timestamp)
        raw = read_raw_window(episode.symbol, "binance_trades", t-1205_000_000, t)
        unique = raw.unique(subset=["timestamp", "ticker", "side", "price", "amount"])
        row = {"episode_id": episode.episode_id, "symbol": episode.symbol,
               "exchange": episode.exchange, "side": episode.side, "phase": episode.phase,
               "first_cascade_detection_timestamp": t}
        for label, frame in [("raw", raw), ("dedup", unique)]:
            binned = frame.with_columns(((pl.col("timestamp")-(t-1205_000_000))//5_000_000).alias("bin")).group_by("bin").agg(
                (pl.col("price")*pl.col("amount")).sum().alias("notional"))
            volume = np.zeros(241)
            volume[binned["bin"].to_numpy()] = binned["notional"].to_numpy()
            mean, q95 = volume[:240].mean(), np.quantile(volume[:240], .95)
            row[label+"_trade_ratio"] = volume[-1]/max(mean, 1)
            row[label+"_strong_trade"] = bool(volume[-1] > q95 and volume[-1] >= 2*max(mean, 1))
            row[label+"_trade_notional"] = volume[-1]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "trade_duplicate_confirmation_sensitivity.csv", index=False)
    return result


def direction_comparison():
    rows = []
    for symbol in SYMBOLS:
        market = pl.read_parquet(OUT/f"market_{symbol}.parquet", columns=["price_valid", "log_return"])
        valid = market["price_valid"].to_numpy()
        flat = market["log_return"].to_numpy() == 0
        for exchange in EXCHANGES:
            for side in SIDES:
                f = pl.read_parquet(OUT/f"features_{symbol}_{exchange}_{side}.parquet",
                                   columns=["phase", "eligible", "liquidation_count", "burst", "direction_aligned"]).to_pandas()
                for split in ("train", "validation"):
                    for group, burst in [("Normal active", False), ("Burst (liquidations only)", True)]:
                        chosen = f.eligible & f.phase.eq(split) & (f.liquidation_count > 0) & f.burst.eq(burst) & valid
                        rows.append({"symbol": symbol, "exchange": exchange, "side": side,
                                     "phase": split, "group": group, "valid_windows": int(chosen.sum()),
                                     "aligned_windows": int((chosen & f.direction_aligned).sum()),
                                     "flat_windows": int((chosen & flat).sum())})
    result = pd.DataFrame(rows)
    result["alignment_share"] = result.aligned_windows / result.valid_windows
    result.to_csv(OUT/"direction_comparison.csv", index=False)
    return result


def write_run_manifest():
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    files = sorted([*OUT.glob("*.csv"), *OUT.glob("*.parquet"), OUT / "frozen_config.json"])
    metadata = {"version": VERSION,
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "python_modules": {p.name: digest(p) for p in [resolve_legacy("liquidation_bursts.py"), resolve_legacy("test_liquidation_bursts.py")]},
                "artifacts": {p.name: digest(p) for p in files}}
    (OUT/"run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_saved_run():
    """Reject stale outputs after source, computation, or stored-result changes."""
    metadata = json.loads((OUT/"run_manifest.json").read_text())
    for folder, key in [(ROOT, "python_modules"), (OUT, "artifacts")]:
        for name, expected in metadata[key].items():
            with (resolve_legacy(name) if folder == ROOT else folder/name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Changed {name}; set RECOMPUTE=True to refresh the analysis")
    # Read-only source verification; avoid rewriting files protected by the manifest.
    for row in pd.read_csv(OUT/"source_verification.csv").to_dict("records"):
        with (resolve_legacy(row["source"])).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"Changed source: {row['source']}; refresh the source audit first")
    return metadata
