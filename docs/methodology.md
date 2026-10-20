# Methodology and experiment boundaries

## Time and labels

The authoritative axis is `int64` Unix microseconds, UTC. Bybit liquidation availability is timestamp +200 ms.
Feature intervals are `[t-window,t)` and exclude equal-time records. Maker state is constructed strictly before
`floor(trade_timestamp / 1e6)` and held for that second; exact 5s liquidation alignment is an additional trade-time feature.
Stable sorting and output row restoration preserve the submission contract.

Maker markout uses taker sign, trade price, +0.5 bps rebate, and the last midpoint at or before `t+h`.
The lookup is backward as-of at the **future label time**, not a forward search for a quote after that time.
Targets beyond global BBO bounds are invalid. Stale forward-filled labels remain in the official metric; fresh-label
diagnostics are reported separately. Each trade's weight is `min(price*amount,100000)`.

`PnL_all`, `PnL_kept`, and `PnL_filtered` are weighted average markouts in bps. Score is `PnL_kept − PnL_all`.
The weighted USD-markout sum is `sum(kept_weight*markout)/10000`; it assumes hypothetical participation and is not
realized portfolio PnL. The original requirement is an **average daily** kept clipped turnover of at least $500,000.

## Burst and warning research

The detector compares completed 5s activity with 240 earlier bins over 20 minutes; the current bin is excluded from
the baseline. Counts and notional use explicit ratio/floor/quantile rules calibrated before December 15.
Price direction, persistence and trade/price intensity identify *potential cascades*. These are observational labels,
not proof of a structural cascade. The original sensitivity covers alternative windows, offsets and cleaning choices.

Follow-through studies retrospective onset/detection/cascade/end anchors. End-of-episode timing is known after the
next completed bin. Signed post-end returns cannot be used as an available-at-onset predictor. The 42-test retrospective
family uses whole-UTC-day sign-flip inference, Holm adjustment and day/block sensitivity. Its assumptions remain material.

Historical warning experiments observe directional changes at 1s resolution. Threshold states 2.5/5/10 bps feed a
first-order transition approximation, W (mean of 20 transition surprises), and a calibrated L transform.
December 1–7 fits transitions; December 8–14 calibrates nuisance scores. Supervised partitions are 60/15/15 complete days:
train before January 30, validation January 30–February 13, and historical test February 14–28. The 305s purge covers
maximum horizon and detection delay. The historical “test” was already inspected in earlier EDA.

Logistic/LightGBM classifier metrics include binary log loss, probability MSE/Brier, average precision, precision/recall,
and F1. The reduced 5s model has five trees, at most depth eight and five lags. Its saved negative test-MSE result is retained.

## Maker fits and controlled ablations

Maker train: December 1–January 31. Validation: February 1–28. The effective regression fit begins December 22
after warmup. The January policy holdout begins January 22. Every fit mask requires `second+301 < boundary`.

The separate side-specific Logistic Regression architecture predicts a new burst onset in `(t,t+5s]` on a 5s clock.
It uses a uniform deterministic 20% clock sample, StandardScaler fit on past data, C=0.01 and no class weights.
Weekly expanding out-of-fold predictions begin December 22; the target-availability purge is 10s.
DC-population predictions are not substituted as calibrated probabilities on the maker clock.

Maker fitting samples 1/16 of feature-equivalence groups using the original stable hash. Weights are summed clipped
notionals ×16, normalized to mean one within a fit; the group target is `sum(w*y)/sum(w)` without target clipping.
Ridge uses standardized features and alpha=100; LightGBM uses the frozen fixed architecture. Feature-family ablations
E4–E7 change only their declared feature inputs. E9 uses the exact E3 sample, features, targets, weights and boundary mask
with the fixed CatBoost parameters. No February labels enter fitting. Full-trade squared error adds the constant
within-group residual variance omitted from the grouped fit objective.

## Policy calibration and uncertainty

Budget matching sorts model scores with deterministic group/trade tie keys and binary decisions. The corrected group key
is uint64 with disjoint symbol/time/side/alignment namespaces. The old colliding float-policy artifacts are omitted.
Budget overshoot is bounded by one clipped trade (at most $100,000 over the whole evaluation period).
Threshold calibration uses scores/weights, not markout outcomes. Matching on February nevertheless uses its turnover
distribution; model/strategy choice also reused February outcomes. Neither is an independent prospective evaluation.

Primary E5/E7 paired maker tests have nine Bonferroni comparisons. The later source-informed expanded family has 24
combined comparisons. E9−E3 has its own three-horizon family. Pointwise Score intervals and paired differences are not
interchangeable. The saved 1-day and 5-day block checks preserve paired calendar dependence, but cannot remove
selection bias. Undefined empty-retention resamples yield missing nominal intervals; conditional quantiles are labeled.

## Reference implementation differences

Historical reference fitted models and exact ranking/cleaning code were unavailable. Current E2/E3 are fixed
reconstructions, not recovered historical V4 weights. E1's initial 5s window differs from the documented 60s rule,
which is retained as E1_60; E1_60_strength adds a current flow-strength ranking.
The reference comparison used a different sampling/walk-forward scheme and per-symbol turnover budget.
Its absolute results are not directly comparable with this project's combined-budget February evaluation.
Flagged liquidation price cleaning is explicitly unresolved; the published sensitivity does not silently redefine it.
