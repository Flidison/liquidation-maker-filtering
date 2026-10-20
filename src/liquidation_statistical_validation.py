"""Independent validation of descriptive liquidation follow-through claims.

Run with .data_quality_work/venv/bin/python liquidation_statistical_validation.py.
Reads saved episode-level outcomes; does not refit or change the detector.
Primary family is fixed below before any p-values are computed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from research_paths import ROOT, resolve_legacy
SOURCE = ROOT / "artifacts/results/followthrough"
OUT = ROOT / "artifacts/results/statistics"
DAY_US = 86_400_000_000
SEED = 20261011
BOOTSTRAP_DRAWS = 19999
SIGNFLIP_DRAWS = 199999
CLASSES = ["Normal activity", "Liquidation burst", "Potential liquidation cascade"]
SYMBOLS = ["btcusdt", "ethusdt"]
HORIZONS = [5, 20]
METRICS = ["post_end_signed_return_bps", "post_minus_pre_end_rv_bps",
           "post_minus_pre_onset_rv_bps", "during_minus_pre_equal_duration_rms_bps"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fixed_family():
    return [{"symbol": s, "classification": c, "metric": m, "horizon_s": h}
            for s in SYMBOLS for c in CLASSES for m in METRICS
            for h in ([0] if m == METRICS[-1] else HORIZONS)]


def holm(pvalues):
    """Holm step-down adjusted p-values; every fixed-family test is retained."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("All predeclared tests must have finite valid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def signflip_p(sums, seed=SEED):
    """Two-sided test of zero aggregate mean via whole-cluster sign flips.

    Exactness requires independent cluster sums symmetric about zero under H0.
    Merely E[sum]=0 does not guarantee symmetry. Counts are held fixed, so testing
    sum equals zero is equivalent to testing the episode-weighted mean equals zero.
    """
    sums = np.asarray(sums, dtype=float)
    sums = sums[sums != 0]
    g = len(sums)
    if g == 0:
        return 1., "exact", 1, 0.
    observed = abs(sums.sum())
    tolerance = max(1e-12, observed * 1e-12)
    if g <= 18:
        # Fix the first sign positive: the other half is identical under |sum|.
        n = 1 << (g - 1)
        exceed = 0
        for start in range(0, n, 8192):
            bits = ((np.arange(start, min(n, start + 8192), dtype=np.uint64)[:, None]
                     >> np.arange(g - 1, dtype=np.uint64)) & 1).astype(float)
            statistics = abs(sums[0] + (2 * bits - 1) @ sums[1:])
            exceed += np.count_nonzero(statistics >= observed - tolerance)
        return exceed / n, "exact", n, 0.
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, SIGNFLIP_DRAWS, 8192):
        n = min(8192, SIGNFLIP_DRAWS - start)
        signs = 2 * rng.integers(0, 2, size=(n, g), dtype=np.int8) - 1
        exceed += np.count_nonzero(abs(signs @ sums) >= observed - tolerance)
    p = (1 + exceed) / (1 + SIGNFLIP_DRAWS)
    return p, "monte_carlo_plus_one", SIGNFLIP_DRAWS, np.sqrt(p * (1-p) / SIGNFLIP_DRAWS)


def cluster_inference(values, days, calendar_days, seed=SEED):
    v = pd.DataFrame({"value": values, "day": days}).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        raise ValueError("Predeclared population has no valid observations")
    d = v.groupby("day").value.agg(["sum", "count"]).reindex(calendar_days, fill_value=0)
    sums, counts = d["sum"].to_numpy(), d["count"].to_numpy()
    mean = float(sums.sum()/counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (BOOTSTRAP_DRAWS, len(d)))
    numer = sums[idx].sum(axis=1)
    denom = counts[idx].sum(axis=1)
    estimates = numer[denom > 0]/denom[denom > 0]
    ci = np.quantile(estimates, [.025, .975])
    p, method, draws, mcse = signflip_p(sums, seed)
    scores = sums - counts*mean
    se = np.sqrt(len(d)/(len(d)-1)*np.sum(scores**2))/counts.sum()
    tstat = mean/se if se > 0 else (np.sign(mean)*np.inf if mean else 0.)
    tp = 2*student_t.sf(abs(tstat), len(d)-1)
    active = counts > 0
    loo = (sums.sum()-sums[active])/(counts.sum()-counts[active])
    loo = loo[np.isfinite(loo)]
    n_days = int(active.sum())
    notes = []
    if n_days < 10:
        notes.append("fewer_than_10_active_UTC_days; inference_very_fragile")
    elif n_days < 20:
        notes.append("fewer_than_20_active_UTC_days; small_cluster_sample")
    if len(v) < 50:
        notes.append("fewer_than_50_episodes")
    if counts.max()/counts.sum() > .25:
        notes.append("one_day_contains_over_25pct_of_episodes")
    if len(loo) and np.any(np.sign(loo) != np.sign(mean)):
        notes.append("leave_one_day_out_changes_mean_sign")
    return {
        "n_valid_episodes": len(v), "n_valid_utc_days": n_days,
        "n_calendar_days": len(calendar_days), "effect_mean": mean,
        "median_episode_effect": float(v.value.median()),
        "positive_episode_share": float(v.value.gt(0).mean()),
        "negative_episode_share": float(v.value.lt(0).mean()),
        "mean_ci95_low": float(ci[0]), "mean_ci95_high": float(ci[1]),
        "p_signflip_two_sided": p, "signflip_method": method,
        "signflip_draws_or_exact_patterns": draws, "p_monte_carlo_standard_error": float(mcse),
        "cluster_sandwich_se": float(se), "cluster_t_statistic_diagnostic": float(tstat),
        "cluster_t_p_diagnostic": float(tp),
        "loo_mean_min": float(loo.min()) if len(loo) else np.nan,
        "loo_mean_max": float(loo.max()) if len(loo) else np.nan,
        "loo_same_sign_share": float(np.mean(np.sign(loo) == np.sign(mean))) if len(loo) else np.nan,
        "max_day_episode_share": float(counts.max()/counts.sum()),
        "notes": "; ".join(notes),
    }, d


def input_specification():
    now = datetime.now(timezone.utc).isoformat()
    previous_path = OUT/"statistical_test_specification.json"
    original_time = now
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["primary_family"] != fixed_family():
            raise ValueError("Refusing to overwrite a different predeclared primary family")
        original_time = previous.get("initial_protocol_written_utc", previous["written_before_pvalue_computation_utc"])
    return {
        "written_before_pvalue_computation_utc": now,
        "initial_protocol_written_utc": original_time,
        "scope": "validation 2026-02-01 through 2026-02-28 UTC only",
        "primary_family": fixed_family(), "primary_family_size": 42,
        "primary_null": "episode-weighted mean of named signed return or paired volatility difference equals zero",
        "primary_test": "two-sided whole-UTC-day sign flip; exact through 18 active nonzero days, otherwise 199999 Monte Carlo flips with plus-one correction",
        "clustering_day": "UTC day of episode end for post-end outcomes; UTC day of episode onset for during vs before comparisons. The current catalog has no validation onset outside February; assert rather than silently discard any future mismatch.",
        "assumptions": [
            "UTC-day cluster sums are mutually independent and symmetric about zero under the null; a zero mean alone is insufficient for exact sign-flip validity.",
            "Within a UTC day all dependence between sides, venues and overlapping events is allowed.",
            "The percentile cluster bootstrap assumes exchangeable calendar days and enough independent days. It preserves all episodes and their episode weights inside a resampled day.",
            "28 calendar days do not establish independence across days; 2-day nonoverlapping blocks are a sensitivity check, not a proof.",
            "Normal controls and cascade selection are observational, and cascade classification includes price/volume conditions; no causal interpretation.",
            "Episode end is retrospective and becomes identifiable only after the next 5-second bin. Post-end results are descriptive and are not an executable prediction or maker markout.",
        ],
        "multiplicity": "Holm across all 42 primary validation hypotheses, alpha 0.05. Family includes all symbols/classes/horizons and all four named metric types, fixed before these p-values.",
        "confidence_intervals": "19999 whole-calendar-day bootstrap draws; 95% pointwise percentile intervals, NOT simultaneous familywise intervals; empty bootstrap samples discarded",
        "limitations": "The descriptive patterns were viewed before this validation request. This is a fixed analysis protocol for retrospective follow-up, not preregistration or an untouched confirmatory holdout.",
        "sensitivity": "Returns only: median, 1/99 percent winsorized mean, 1 percent trimmed mean, equal-day mean, leave-one-day-out, 2-day block p/CI, by-side descriptive means. Sensitivity p-values are exploratory/unadjusted and do not replace primary inference. Cluster-sandwich t p-values receive a separate Holm correction over the same42 hypotheses as a diagnostic comparison; they are not a second chance to declare rejection.",
        "volatility_definition": "RV = sqrt(sum of 1-second log returns squared) * 10000; RMS = RV/sqrt(seconds). All paired comparisons use equal-duration before/after windows within each pair.",
        "source_sha256": {name: sha256(SOURCE/name) for name in ["all_outcomes.parquet", "episode_catalog.parquet"]},
        "script_sha256": sha256(Path(__file__)), "seed": SEED,
    }


def main():
    OUT.mkdir(exist_ok=True)
    spec = input_specification()
    (OUT/"statistical_test_specification.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    outcomes = pd.read_parquet(SOURCE/"all_outcomes.parquet")
    catalog = pd.read_parquet(SOURCE/"episode_catalog.parquet")
    outcomes = outcomes.loc[outcomes.phase.eq("validation") & outcomes.anchor.eq("episode_end")].copy()
    catalog = catalog.loc[catalog.phase.eq("validation")].copy()
    assert not outcomes.duplicated(["episode_id", "horizon_s"]).any()
    assert not catalog.episode_id.duplicated().any()
    first_day = pd.Timestamp("2026-02-01", tz="UTC").value//1_000//DAY_US
    calendar = np.arange(first_day, first_day+28)
    primary, sensitivity, daily, sides = [], [], [], []
    for i, key in enumerate(fixed_family()):
        if key["metric"] == METRICS[-1]:
            g = catalog.loc[catalog.symbol.eq(key["symbol"]) & catalog.classification.eq(key["classification"])].copy()
            g["value"] = g.during_rms_bps-g.before_same_duration_rms_bps
            g["cluster_day"] = g.onset_timestamp//DAY_US
        else:
            g = outcomes.loc[outcomes.symbol.eq(key["symbol"]) & outcomes.classification.eq(key["classification"])
                             & outcomes.horizon_s.eq(key["horizon_s"])].copy()
            if key["metric"] == METRICS[0]:
                g["value"] = g.post_signed_return_bps
            elif key["metric"] == METRICS[1]:
                g["value"] = g.post_rv_bps-g.pre_anchor_rv_bps
            else:
                g["value"] = g.post_rv_bps-g.pre_onset_rv_bps
            g["cluster_day"] = g.anchor_timestamp//DAY_US
        assert g.cluster_day.isin(calendar).all(), "Unexpected validation-day boundary"
        result, d = cluster_inference(g.value, g.cluster_day, calendar, seed=SEED+i)
        result.update(key, n_episodes_total=len(g), phase="validation")
        result["n_missing_episodes"] = len(g)-result["n_valid_episodes"]
        primary.append(result)
        for day, r in d.iterrows():
            daily.append({**key, "utc_day": pd.to_datetime(day*86400, unit="s", utc=True).isoformat(),
                          "sum_effect": r["sum"], "episode_count": int(r["count"])})
        if key["metric"] == METRICS[0]:
            v = g.loc[np.isfinite(g.value)].copy()
            q1, q99 = v.value.quantile([.01, .99])
            winsor = v.value.clip(q1, q99)
            ordered = np.sort(v.value.to_numpy())
            trim = int(len(ordered)*.01)
            trimmed = ordered[trim:len(ordered)-trim] if trim else ordered
            active = d["count"].gt(0)
            blocks = (v.cluster_day-first_day)//2
            b_result, _ = cluster_inference(v.value, blocks, np.arange(14), seed=SEED+100+i)
            sensitivity.append({**key, "n_valid_episodes": len(v),
                "primary_mean": result["effect_mean"], "median": float(v.value.median()),
                "winsorized_1_99_mean": float(winsor.mean()), "trimmed_1pct_mean": float(trimmed.mean()),
                "winsor_q01": float(q1), "winsor_q99": float(q99),
                "equal_active_day_mean": float((d.loc[active, "sum"]/d.loc[active, "count"]).mean()),
                "loo_mean_min": result["loo_mean_min"], "loo_mean_max": result["loo_mean_max"],
                "loo_same_sign_share": result["loo_same_sign_share"],
                "two_day_active_blocks": b_result["n_valid_utc_days"],
                "two_day_bootstrap_ci95_low": b_result["mean_ci95_low"],
                "two_day_bootstrap_ci95_high": b_result["mean_ci95_high"],
                "two_day_signflip_p_exploratory": b_result["p_signflip_two_sided"]})
            for side, sg in v.groupby("side"):
                sides.append({**key, "side": side, "n_valid_episodes": len(sg),
                              "n_valid_utc_days": sg.cluster_day.nunique(),
                              "mean_signed_return_bps": float(sg.value.mean()),
                              "mean_raw_return_bps": float(sg.post_return_bps.mean()),
                              "median_signed_return_bps": float(sg.value.median())})
    primary = pd.DataFrame(primary)
    assert len(primary) == spec["primary_family_size"] == 42
    primary["p_holm_42"] = holm(primary.p_signflip_two_sided)
    primary["cluster_t_p_holm_42_diagnostic"] = holm(primary.cluster_t_p_diagnostic)
    primary["reject_h0_holm_05_under_assumptions"] = primary.p_holm_42.le(.05)
    primary["ci_is_pointwise_not_simultaneous"] = True
    for name, frame in [("statistical_tests_validation", primary),
                        ("return_robustness_validation", pd.DataFrame(sensitivity)),
                        ("statistical_daily_aggregates", pd.DataFrame(daily)),
                        ("return_by_side_validation", pd.DataFrame(sides))]:
        frame.to_csv(OUT/f"{name}.csv", index=False)
    summary = {
        "primary_tests": len(primary),
        "rejects_holm_05_under_assumptions": int(primary.reject_h0_holm_05_under_assumptions.sum()),
        "counts_by_metric": primary.groupby("metric").reject_h0_holm_05_under_assumptions.sum().to_dict(),
        "small_cluster_groups": int(primary.n_valid_utc_days.lt(20).sum()),
        "not_a_predictive_or_causal_test": True,
        "output_sha256": {p.name: sha256(p) for p in OUT.glob("*.csv") if p.name.startswith(("statistical_", "return_"))},
    }
    (OUT/"statistical_validation_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(primary.loc[:, ["symbol", "classification", "metric", "horizon_s", "n_valid_episodes",
                          "n_valid_utc_days", "effect_mean", "mean_ci95_low", "mean_ci95_high",
                          "p_signflip_two_sided", "p_holm_42"]].to_string(index=False))


if __name__ == "__main__":
    main()
