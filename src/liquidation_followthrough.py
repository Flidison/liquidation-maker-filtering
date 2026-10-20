"""Descriptive post-event price and realized-volatility study; no detector refit.

Future outcomes live only in this separate module/artifact directory. All prices
are Binance BBO midpoints strictly before UTC integer-second boundaries.
"""
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import polars as pl

import liquidation_bursts as lb

from research_paths import ROOT, resolve_legacy
OUT = ROOT / "artifacts/results/followthrough"
CLASSES = ["Normal activity", "Liquidation burst", "Potential liquidation cascade"]
ANCHORS = {"onset": "onset_timestamp", "first_detection": "first_detection_timestamp",
           "episode_end": "end_timestamp", "cascade_detection": "first_cascade_detection_timestamp"}
HORIZONS = (5, 20)
ZERO_BPS = 1e-10


class PricePath:
    """Vectorized interval lookup, with complete fresh 1-second return coverage."""
    def __init__(self, seconds):
        self.start = int(seconds["second"][0])
        self.n = seconds.height
        assert np.array_equal(seconds["second"].to_numpy(), np.arange(self.start, self.start+self.n))
        self.mid = np.r_[np.nan, seconds["mid_close"].to_numpy()]
        self.fresh = np.r_[False, seconds["quote_age_upper_s"].to_numpy() <= 2]
        self.fresh &= np.isfinite(self.mid) & (self.mid > 0)
        self.logmid = np.log(self.mid)
        r = np.diff(self.logmid)
        valid = self.fresh[1:] & self.fresh[:-1] & np.isfinite(r)
        # Prefix sums let every requested interval be checked without filling gaps.
        self.sq = np.r_[0., np.cumsum(np.where(valid, r*r, 0.))]
        self.bad = np.r_[0, np.cumsum(~valid)]

    def interval(self, start_s, end_s):
        start_s, end_s = np.broadcast_arrays(np.asarray(start_s, dtype=np.int64),
                                            np.asarray(end_s, dtype=np.int64))
        lo, hi = start_s-self.start, end_s-self.start
        in_range = (lo >= 0) & (hi <= self.n) & (hi > lo)
        a, b = np.clip(lo, 0, self.n), np.clip(hi, 0, self.n)
        valid = in_range & ((self.bad[b]-self.bad[a]) == 0)
        var = np.maximum(0., self.sq[b]-self.sq[a])
        ret = np.where(valid, (self.logmid[b]-self.logmid[a])*1e4, np.nan)
        rv = np.where(valid, np.sqrt(var)*1e4, np.nan)
        rms = rv / np.sqrt(np.maximum(1, end_s-start_s))
        return {"return_bps": ret, "rv_bps": rv, "rms_bps": rms, "valid": valid}


def direction(values):
    a = np.asarray(values)
    return np.select([~np.isfinite(a), a > ZERO_BPS, a < -ZERO_BPS],
                     ["missing", "up", "down"], default="unchanged")


def change(after, before):
    after, before = np.asarray(after), np.asarray(before)
    return np.select([~np.isfinite(after) | ~np.isfinite(before),
                      after-before > ZERO_BPS, after-before < -ZERO_BPS],
                     ["missing", "increased", "decreased"], default="unchanged")


def safe_ratio(after, before):
    after, before = np.asarray(after), np.asarray(before)
    return np.divide(after, before, out=np.full(after.shape, np.nan),
                     where=np.isfinite(after) & np.isfinite(before) & (before > ZERO_BPS))


def catalogs(symbol, flagged):
    """Normal controls are contiguous active bins with no symbol-wide burst.

    Future outcomes do not influence control eligibility. Adjacent bins are
    grouped within each venue/side and phase, just as in the burst catalog.
    """
    cols = ["timestamp", "phase", "eligible", "liquidation_count", "liquidation_notional", "burst"]
    streams = {(ex, side): pl.read_parquet(lb.OUT/f"features_{symbol}_{ex}_{side}.parquet",
                    columns=cols).to_pandas() for ex in lb.EXCHANGES for side in lb.SIDES}
    any_burst = np.logical_or.reduce([f.burst.to_numpy() for f in streams.values()])
    controls = []
    for (exchange, side), f in streams.items():
        g = f.loc[f.eligible & (f.liquidation_count > 0) & ~any_burst].copy()
        group = (g.timestamp.diff().ne(5_000_000) | g.phase.ne(g.phase.shift())).cumsum()
        c = g.groupby(group, sort=False).agg(first_detection_timestamp=("timestamp", "min"),
                end_timestamp=("timestamp", "max"), phase=("phase", "first"),
                flagged_windows=("timestamp", "size"), liquidation_count=("liquidation_count", "sum"),
                liquidation_notional=("liquidation_notional", "sum")).reset_index(drop=True)
        c["onset_timestamp"] = c.first_detection_timestamp - 5_000_000
        c["duration_s"] = (c.end_timestamp-c.onset_timestamp)/1e6
        c["symbol"], c["exchange"], c["side"] = symbol, exchange, side
        c["classification"] = CLASSES[0]
        c["first_cascade_detection_timestamp"] = pd.array([None]*len(c), dtype="Int64")
        c["episode_id"] = [f"N_{symbol}_{exchange}_{side}_{i:06}" for i in range(1, len(c)+1)]
        controls.append(c)
    keep = ["episode_id", "symbol", "exchange", "side", "phase", "classification",
            *ANCHORS.values(), "duration_s", "flagged_windows", "liquidation_count", "liquidation_notional"]
    cat = pd.concat([flagged.loc[flagged.symbol.eq(symbol), keep],
                     *[c[keep] for c in controls]], ignore_index=True)
    cat["window_count"] = (cat.duration_s/5).astype(np.int64)
    cat.loc[cat.classification.eq(CLASSES[0]), "flagged_windows"] = 0
    # End of the last marked bin is a retrospective boundary. It can be recognized
    # after the next 5-second bin closes; retain that separate time explicitly.
    cat["end_recognition_timestamp"] = cat.end_timestamp + 5_000_000
    return cat, any_burst


def phase_end(frame):
    return np.where(frame.phase.eq("train"), lb.VALIDATION, lb.END)


def measure_catalog(cat, path):
    """Before/during activity with duration-normalized volatility for unequal runs."""
    c = cat.copy()
    a = c.onset_timestamp.to_numpy(np.int64)//1_000_000
    b = c.end_timestamp.to_numpy(np.int64)//1_000_000
    d = b-a
    before, during, background = path.interval(a-d, a), path.interval(a, b), path.interval(a-1200, a)
    for prefix, vals in [("before_same_duration", before), ("during", during), ("background_20m", background)]:
        for metric in ("return_bps", "rv_bps", "rms_bps"):
            c[f"{prefix}_{metric}"] = vals[metric]
    c["during_signed_return_bps"] = c.during_return_bps*np.where(c.side.eq("buy"), 1, -1)
    c["during_rv_change"] = change(c.during_rv_bps, c.before_same_duration_rv_bps)
    c["during_rv_ratio"] = safe_ratio(c.during_rv_bps, c.before_same_duration_rv_bps)
    c["during_vs_background_rms_ratio"] = safe_ratio(c.during_rms_bps, c.background_20m_rms_bps)
    return c


def measure_outcomes(cat, path, any_burst):
    exposure = np.r_[0, np.cumsum(np.repeat(any_burst, 5))]
    output = []
    for anchor, col in ANCHORS.items():
        c = cat.loc[cat[col].notna()].copy()
        a = c[col].to_numpy(np.int64)//1_000_000
        onset = c.onset_timestamp.to_numpy(np.int64)//1_000_000
        end = c.end_timestamp.to_numpy(np.int64)//1_000_000
        for h in HORIZONS:
            o = c[["episode_id", "symbol", "exchange", "side", "phase", "classification",
                   "onset_timestamp", "first_detection_timestamp", "end_timestamp",
                   "first_cascade_detection_timestamp", "end_recognition_timestamp"]].copy()
            o["anchor"], o["horizon_s"] = anchor, h
            o["anchor_timestamp"] = a*1_000_000
            o["target_timestamp"] = (a+h)*1_000_000
            o["class_known_at_anchor"] = np.where(anchor == "onset", "Not yet detected",
                np.where(c.classification.eq(CLASSES[0]), CLASSES[0],
                    np.where(c.first_cascade_detection_timestamp.le(a*1_000_000).fillna(False),
                             CLASSES[2], CLASSES[1])))
            bounded = (a+h <= phase_end(c))
            o["target_within_phase"] = bounded
            for name, start, stop in [("pre_anchor", a-h, a), ("post", a, a+h),
                                      ("pre_onset", onset-h, onset), ("from_onset", onset, a+h)]:
                vals = path.interval(start, stop)
                for metric in ("return_bps", "rv_bps", "rms_bps"):
                    v = vals[metric]
                    if name in ("post", "from_onset"):
                        v = np.where(bounded, v, np.nan)
                    o[f"{name}_{metric}"] = v
            o["anchor_mid"] = path.mid[np.clip(a-path.start, 0, path.n)]
            o["target_mid"] = np.where(bounded & np.isfinite(o.post_return_bps),
                path.mid[np.clip(a+h-path.start, 0, path.n)], np.nan)
            o["post_signed_return_bps"] = o.post_return_bps*np.where(o.side.eq("buy"), 1, -1)
            o["pre_anchor_signed_return_bps"] = o.pre_anchor_return_bps*np.where(o.side.eq("buy"), 1, -1)
            o["post_price_direction"] = direction(o.post_return_bps)
            o["post_liquidation_direction"] = np.select([
                ~np.isfinite(o.post_signed_return_bps), o.post_signed_return_bps > ZERO_BPS,
                o.post_signed_return_bps < -ZERO_BPS], ["missing", "continuation", "reversal"], "unchanged")
            o["post_price_direction_1bps"] = np.select([~np.isfinite(o.post_return_bps),
                o.post_return_bps > 1, o.post_return_bps < -1], ["missing", "up", "down"], "within_1bps")
            for baseline in ("pre_anchor", "pre_onset"):
                o[f"rv_vs_{baseline}_change"] = change(o.post_rv_bps, o[f"{baseline}_rv_bps"])
                o[f"rv_vs_{baseline}_ratio"] = safe_ratio(o.post_rv_bps, o[f"{baseline}_rv_bps"])
            o["post_vs_background_rms_ratio"] = safe_ratio(o.post_rms_bps, c.background_20m_rms_bps)
            lo = np.clip(a-path.start, 0, len(exposure)-1)
            hi = np.clip(a+h-path.start, 0, len(exposure)-1)
            o["post_burst_exposure_s"] = np.where(bounded, exposure[hi]-exposure[lo], np.nan)
            o["post_overlaps_own_episode_s"] = np.maximum(0, np.minimum(a+h, end)-np.maximum(a, onset))
            # This field lets a reader inspect the distinct total move relative to
            # the price BEFORE the event, as opposed to the move AFTER its end.
            output.append(o)
    return pd.concat(output, ignore_index=True)


def grouped(frame, extra=()):
    keys = ["phase", "symbol", "classification", *extra]
    for names, g in frame.groupby(keys, observed=True, sort=True):
        base = dict(zip(keys, names))
        for side in ("ALL", "buy", "sell"):
            s = g if side == "ALL" else g.loc[g.side.eq(side)]
            if len(s):
                yield {**base, "side": side}, s


def cluster_mean_ci(g, column, seed=20261011, draws=2000):
    v = g.loc[np.isfinite(g[column]), ["anchor_timestamp", column]].copy()
    if v.empty:
        return np.nan, np.nan, 0
    v["day"] = v.anchor_timestamp//86_400_000_000
    days = v.groupby("day")[column].agg(["sum", "count"])
    n = len(days)
    if n < 5:
        return np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (draws, n))
    means = days["sum"].to_numpy()[idx].sum(axis=1)/days["count"].to_numpy()[idx].sum(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return lo, hi, n


def summaries(outcomes, catalog):
    price_rows, vol_rows, during_rows = [], [], []
    for key, g in grouped(outcomes, ("anchor", "horizon_s")):
        v = g.loc[np.isfinite(g.post_return_bps)]
        row = {**key, "episodes": len(g), "valid": len(v), "missing": len(g)-len(v)}
        for col in ("post_return_bps", "post_signed_return_bps", "pre_anchor_return_bps", "from_onset_return_bps"):
            row[f"mean_{col}"] = v[col].mean()
        row.update(median_post_return_bps=v.post_return_bps.median(),
                   median_post_signed_return_bps=v.post_signed_return_bps.median(),
                   p05_post_return_bps=v.post_return_bps.quantile(.05),
                   p95_post_return_bps=v.post_return_bps.quantile(.95),
                   up_share=v.post_price_direction.eq("up").mean(),
                   down_share=v.post_price_direction.eq("down").mean(),
                   unchanged_share=v.post_price_direction.eq("unchanged").mean(),
                   within_1bps_share=v.post_price_direction_1bps.eq("within_1bps").mean(),
                   continuation_share=v.post_liquidation_direction.eq("continuation").mean(),
                   reversal_share=v.post_liquidation_direction.eq("reversal").mean(),
                   post_has_burst_share=v.post_burst_exposure_s.gt(0).mean())
        for column, name in [("post_return_bps", "raw"), ("post_signed_return_bps", "signed")]:
            lo, hi, ndays = cluster_mean_ci(v, column)
            row[f"{name}_mean_ci95_low"], row[f"{name}_mean_ci95_high"] = lo, hi
            row["valid_utc_days"] = ndays
        price_rows.append(row)
        for baseline in ("pre_anchor", "pre_onset"):
            vc = g.loc[np.isfinite(g.post_rv_bps) & np.isfinite(g[f"{baseline}_rv_bps"])]
            status = vc[f"rv_vs_{baseline}_change"]
            vol_rows.append({**key, "baseline": baseline, "episodes": len(g), "valid_pairs": len(vc),
                "mean_before_rv_bps": vc[f"{baseline}_rv_bps"].mean(),
                "mean_after_rv_bps": vc.post_rv_bps.mean(),
                "mean_delta_rv_bps": (vc.post_rv_bps-vc[f"{baseline}_rv_bps"]).mean(),
                "median_ratio": vc[f"rv_vs_{baseline}_ratio"].median(),
                "increased_share": status.eq("increased").mean(),
                "decreased_share": status.eq("decreased").mean(),
                "unchanged_share": status.eq("unchanged").mean(),
                "zero_baseline_pairs": int(vc[f"{baseline}_rv_bps"].le(ZERO_BPS).sum()),
                "median_post_vs_background_rms_ratio": vc.post_vs_background_rms_ratio.median()})
    for key, g in grouped(catalog):
        v = g.loc[np.isfinite(g.during_rv_bps) & np.isfinite(g.before_same_duration_rv_bps)]
        during_rows.append({**key, "episodes": len(g), "valid_pairs": len(v),
            "mean_duration_s": g.duration_s.mean(), "median_duration_s": g.duration_s.median(),
            "mean_before_rms_bps": v.before_same_duration_rms_bps.mean(),
            "mean_during_rms_bps": v.during_rms_bps.mean(),
            "mean_during_return_bps": v.during_return_bps.mean(),
            "mean_during_signed_return_bps": v.during_signed_return_bps.mean(),
            "median_rv_ratio": v.during_rv_ratio.median(),
            "increased_share": v.during_rv_change.eq("increased").mean(),
            "decreased_share": v.during_rv_change.eq("decreased").mean(),
            "unchanged_share": v.during_rv_change.eq("unchanged").mean(),
            "median_during_vs_background_rms_ratio": v.during_vs_background_rms_ratio.median()})
    return pd.DataFrame(price_rows), pd.DataFrame(vol_rows), pd.DataFrame(during_rows)


def auxiliary_summaries(outcomes):
    """Venue breakdown, raw-sign bins, and explicitly retrospective isolation check."""
    keys = ["phase", "symbol", "exchange", "side", "classification", "anchor", "horizon_s"]
    venue = outcomes.groupby(keys, observed=True).agg(episodes=("episode_id", "size"),
        valid=("post_return_bps", "count"), mean_return_bps=("post_return_bps", "mean"),
        median_return_bps=("post_return_bps", "median"),
        mean_signed_return_bps=("post_signed_return_bps", "mean")).reset_index()
    rows = []
    e = outcomes.loc[outcomes.anchor.eq("episode_end")]
    for key, g in grouped(e, ("horizon_s",)):
        for selection in ("all", "no_burst_in_future_horizon"):
            v = g if selection == "all" else g.loc[g.post_burst_exposure_s.eq(0)]
            rows.append({**key, "selection": selection, "episodes": len(v),
                "valid": v.post_return_bps.count(), "mean_return_bps": v.post_return_bps.mean(),
                "mean_signed_return_bps": v.post_signed_return_bps.mean(),
                "median_rv_ratio": v.rv_vs_pre_anchor_ratio.median()})
    return venue, pd.DataFrame(rows)


def write_episode_lookup(catalog, outcomes):
    """One row per original burst episode, with easily inspected UTC labels."""
    c = catalog.loc[catalog.classification.ne(CLASSES[0])].copy()
    for col in ["onset_timestamp", "first_detection_timestamp", "end_timestamp",
                "first_cascade_detection_timestamp", "end_recognition_timestamp"]:
        c[col.replace("timestamp", "utc")] = pd.to_datetime(c[col], unit="us", utc=True)
    metrics = ["post_return_bps", "post_signed_return_bps", "post_price_direction",
               "from_onset_return_bps", "post_rv_bps", "rv_vs_pre_anchor_change",
               "rv_vs_pre_onset_change", "rv_vs_pre_anchor_ratio", "rv_vs_pre_onset_ratio"]
    f = outcomes.loc[outcomes.classification.ne(CLASSES[0])]
    wide = f.pivot(index="episode_id", columns=["anchor", "horizon_s"], values=metrics)
    wide.columns = [f"{anchor}_{h}s_{metric}" for metric, anchor, h in wide.columns]
    result = c.merge(wide, on="episode_id", how="left", validate="one_to_one")
    result.to_csv(OUT/"burst_episode_summary.csv", index=False)
    return result


def reconcile_raw(outcomes):
    """Independent raw BBO as-of prices for 16 stratified event rows and each 1s step."""
    flagged = outcomes.loc[outcomes.classification.ne(CLASSES[0]) & outcomes.anchor.eq("episode_end")
                           & outcomes.horizon_s.eq(20)]
    checks = []
    for (_, _, _), g in flagged.groupby(["symbol", "classification", "phase"]):
        selected = g.sort_values("episode_id").iloc[[0, len(g)//2]]
        for row in selected.itertuples():
            anchor = row.anchor_timestamp//1_000_000
            boundaries = np.arange(anchor-20, anchor+21, dtype=np.int64)*1_000_000
            raw = lb.read_raw_window(row.symbol, "binance_booktickers", int(boundaries[0]-5_000_000),
                                     int(boundaries[-1])).to_pandas().sort_values("timestamp", kind="stable")
            ti = raw.timestamp.to_numpy()
            pos = np.searchsorted(ti, boundaries, side="left")-1
            if np.any(pos < 0):
                raise AssertionError("Insufficient raw BBO context in reconciliation")
            mid = ((raw.bid_price+raw.ask_price)/2).to_numpy()[pos]
            r = np.diff(np.log(mid))
            actual_ret = np.log(mid[-1]/mid[20])*1e4
            actual_rv = np.sqrt(np.sum(r[20:]**2))*1e4
            prior_rv = np.sqrt(np.sum(r[:20]**2))*1e4
            checks.append({"episode_id": row.episode_id, "symbol": row.symbol, "phase": row.phase,
                "classification": row.classification,
                "return_error_bps": actual_ret-row.post_return_bps,
                "post_rv_error_bps": actual_rv-row.post_rv_bps,
                "pre_rv_error_bps": prior_rv-row.pre_anchor_rv_bps})
    result = pd.DataFrame(checks)
    assert result[["return_error_bps", "post_rv_error_bps", "pre_rv_error_bps"]].abs().max().max() < 1e-5
    result.to_csv(OUT/"raw_bbo_reconciliation.csv", index=False)
    return result


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_manifest():
    manifest = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "parent_manifest_sha256": digest(lb.OUT/"run_manifest.json"),
        "modules": {p: digest(resolve_legacy(p)) for p in ["liquidation_followthrough.py", "test_liquidation_followthrough.py",
                                               "liquidation_followthrough_plots.py"]},
        "artifacts": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.suffix in (".csv", ".parquet")}}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_saved():
    lb.verify_saved_run()
    m = json.loads((OUT/"run_manifest.json").read_text())
    assert digest(lb.OUT/"run_manifest.json") == m["parent_manifest_sha256"]
    for folder, key in [(ROOT, "modules"), (OUT, "artifacts")]:
        for file, checksum in m[key].items():
            if digest(resolve_legacy(file) if folder == ROOT else folder/file) != checksum:
                raise ValueError(f"Changed {file}; recompute follow-through artifacts")
    return m


def run():
    OUT.mkdir(exist_ok=True)
    lb.verify_saved_run()
    flagged = pd.read_parquet(lb.OUT/"episodes.parquet")
    catalogs_all, outcomes_all = [], []
    for symbol in lb.SYMBOLS:
        print(f"Measuring {symbol}", flush=True)
        cat, bursts = catalogs(symbol, flagged)
        path = PricePath(pl.read_parquet(lb.OUT/f"seconds_{symbol}.parquet"))
        cat = measure_catalog(cat, path)
        outcomes = measure_outcomes(cat, path, bursts)
        catalogs_all.append(cat)
        outcomes_all.append(outcomes)
        del path
        gc.collect()
    cat = pd.concat(catalogs_all, ignore_index=True)
    out = pd.concat(outcomes_all, ignore_index=True)
    assert cat.episode_id.is_unique
    assert not out.duplicated(["episode_id", "anchor", "horizon_s"]).any()
    assert cat.classification.ne(CLASSES[0]).sum() == len(flagged)
    for name, f in [("episode_catalog", cat), ("all_outcomes", out),
                    ("burst_episode_outcomes", out.loc[out.classification.ne(CLASSES[0])])]:
        pl.from_pandas(f).write_parquet(OUT/f"{name}.parquet", compression="zstd")
        if name != "all_outcomes":
            f.to_csv(OUT/f"{name}.csv", index=False)
    write_episode_lookup(cat, out)
    print(f"Summarizing {len(cat):,} episodes, {len(out):,} outcomes", flush=True)
    price, vol, during = summaries(out, cat)
    venue, isolation = auxiliary_summaries(out)
    for name, f in [("price_summary", price), ("volatility_summary", vol),
                    ("during_summary", during), ("venue_summary", venue), ("isolation_sensitivity", isolation)]:
        f.to_csv(OUT/f"{name}.csv", index=False)
    qc = out.groupby(["phase", "symbol", "classification", "anchor", "horizon_s"]).agg(
        episodes=("episode_id", "size"), valid_price=("post_return_bps", "count"),
        within_phase=("target_within_phase", "sum")).reset_index()
    qc["missing_price"] = qc.episodes-qc.valid_price
    qc.to_csv(OUT/"outcome_coverage.csv", index=False)
    reconcile_raw(out)
    write_manifest()
    print("Completed", flush=True)
    return price, vol, during


if __name__ == "__main__":
    run()
