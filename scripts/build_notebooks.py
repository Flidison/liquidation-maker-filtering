"""Author the six concise research notebooks from verified aggregate evidence."""
from pathlib import Path
import textwrap
import inspect
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
SETUP='''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from research_paths import artifact_dir
import research_report as report
report.style()
def show(fig, name):
    path=ROOT/'artifacts/figures'/name
    fig.savefig(path, dpi=150)
    display(Image(filename=str(path)))
    plt.close(fig)
'''


def md(s):return nbf.v4.new_markdown_cell(inspect.cleandoc(s))
def code(s):return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
def write(name,cells):
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3.11','language':'python'},
                 'language_info':{'name':'python','version':'3.11'},
                 'publication':{'mode':'frozen aggregate replay; no model fitting','raw_data_included':False}}
    nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/name)


def build():
    write('01_data_exploration.ipynb',[
    md('''# 01 · Data Exploration & Quality
    Can the supplied feeds support a causal liquidation study without silently cleaning away market events?

    Eight Parquet sources cover BTCUSDT and ETHUSDT perpetual futures, December 2025–February 2026:
    Binance trades, BBO and liquidations, plus Bybit liquidations. This notebook replays the original full-row audit
    from `artifacts/results/quality/`; it does not load the private ticks. The separate API example uses artificial data.'''),
    code(SETUP),
    md('''## Coverage and schemas
    `summary.csv` and `source_manifest.csv` were saved by the original audit. The manifest retains source SHA-256 values;
    local absolute paths have been replaced by repository-relative paths. Equal timestamps and exact duplicate records
    are distinct diagnostics. Identical trade records are not proof of duplicate executions.'''),
    code('''quality=report.table('quality','summary.csv')
manifest=report.table('quality','source_manifest.csv')
assert quality.rows.sum()==1_315_384_304
assert manifest.rows.sum()==quality.rows.sum()
display(quality[['file','rows','columns','date range']])
print(f"Full audit: {quality.rows.sum():,} records in {len(quality)} files")'''),
    code('''fig,ax=plt.subplots(figsize=(10,4))
labels=quality.file.str.replace('data/raw/','',regex=False).str.replace('.parquet','',regex=False)
ax.barh(labels,quality.rows,color='#24659a')
ax.set(xscale='log',xlabel='Records (log scale)',title='Source populations · December 2025–February 2026')
show(fig,'source_coverage.png')'''),
    md('''## Missingness, duplicates and gaps
    The audit uses integer UTC microseconds. Bybit availability is delayed by 200 ms for cross-feed analysis.
    Long liquidation inter-arrivals can reflect an inactive feed; a missing BBO interval requires a different interpretation.
    Suspect observations and market-event candidates remain separate from clear schema errors.'''),
    code('''cols=['file','NaN count','timestamp reversals','duplicate timestamps','exact duplicate rows',
      'invalid prices','invalid amounts','largest time gap','number of removed observations']
display(quality[cols])
display(report.table('quality','window_reconciliation.csv'))
integrity=report.table('quality','source_integrity_verification.csv')
assert integrity.unchanged.all()
assert (integrity.sha256_before==integrity.sha256_after).all()'''),
    md('''The saved audit contains no missing cells or invalid basic prices/amounts. Bybit has 63 BTC and 46 ETH timestamp
    reversals in physical file order. It also contains exact duplicate trades and long BBO gaps. These findings do not establish that all observed prices
    are economically plausible. The maker study retains flagged liquidation prices and reports a cleaning sensitivity;
    the warning study has its own explicitly documented predictor cleaning.'''),
    md('''## Portable data access
    These eight CSV fixtures are generated independently of the market dataset. They exercise the public schema,
    timestamp units and submission interface; their prices and outcomes are not evidence about the strategies.'''),
    code('''sample=ROOT/'data/sample'
trades=pd.read_csv(sample/'binance_trades/perp_btcusdt.csv')
bbo=pd.read_csv(sample/'binance_booktickers/perp_btcusdt.csv')
display(trades.head(4))
display(bbo.head(4))
assert trades.timestamp.dtype==np.dtype('int64')
assert set(trades.side)=={'buy','sell'}'''),
    md('''## Conclusion
    The supplied sources passed the recorded structural checks, but duplicates, staleness and liquidation price flags
    still matter. Raw-data redistribution rights are undocumented, so no original rows are published.
    `python -m data_quality_audit` performs the original full audit when authorized files are supplied;
    see [data access](../data/README.md) and [rerun instructions](../docs/reproducibility.md).''')])

    write('02_liquidation_features.ipynb',[
    md('''# 02 · Liquidation Activity & Features
    How does short-window liquidation activity differ from its recent background, and what is available at decision time?

    Historical tables come from the original quality, burst and follow-through experiments.
    A small artificial-feed calculation below demonstrates the working feature pipeline.'''),code(SETUP),
    md('''## Counts, notional and burst detection
    The main detector compares a completed 5-second interval with the preceding 1,200 seconds (240 disjoint bins).
    Windows are half-open: `[t − window, t)`. Counts and `price × amount` notional are separate signals.
    Calibration ends on December 15; price/trade confirmations define potential cascades, not independent price predictions.'''),
    code('''config=json.loads((artifact_dir('bursts')/'frozen_config.json').read_text())
display(pd.Series(config,name='frozen value').to_frame())
thresholds=report.table('bursts','thresholds.csv')
display(thresholds.loc[thresholds.window_s.eq(5)])
classes=report.table('bursts','classification_summary.csv')
display(classes.groupby(['symbol','phase'])[['active_windows','burst_windows','potential_cascade_windows']].sum())'''),
    md('''## Distributions from the full audit
    Histogram bins and counts are extracted from the original exact audit summaries.
    The signed `log10(1 + abs(x))` transform is unchanged; liquidation notional here is positive.
    These are aggregate distributions over all dates, not a newly sampled subset.'''),
    code("show(report.liquidation_distribution_figure(),'liquidation_notional.png')"),
    md('''## Causal feature construction
    Maker market state is held for one second, using records strictly before that second boundary.
    The 5-second liquidation alignment is additionally computed at the exact trade timestamp.
    Bybit timestamps receive the 200 ms delay once, inside `liquidation_arrays`.
    Features include book imbalance/spread, lagged returns, volatility, trade flow, liquidation ages/counts/notional,
    intensity ratios and acceleration. No future midpoint or label is allowed into the feature schema.'''),
    code('''from maker_hybrid.core import liquidation_arrays, liquidation_window
from maker_hybrid.submission import market_batches
from maker_hybrid.features import trade_features, feature_columns
sample=ROOT/'data/sample'
tr=pd.read_csv(sample/'binance_trades/perp_btcusdt.csv')
bb=pd.read_csv(sample/'binance_booktickers/perp_btcusdt.csv')
bn=pd.read_csv(sample/'binance_liquidations/perp_btcusdt.csv')
by=pd.read_csv(sample/'bybit_liquidations/btcusdt.csv')
batch=next(market_batches(tr,bb,bn,by,'btcusdt'))
x=trade_features(batch[2],batch[4])
display(x[['second','spread_bps','book_imbalance','return_5s_bps','rv_20s_bps',
           'liq_binance_buy_log_count_5s','liq_binance_buy_log_notional_5s','alignment']].tail(6))
print('E3 feature count:',len(feature_columns('E3')))'''),
    md('''## Window sensitivity and follow-through
    Alternative window lengths and offsets are retained in `sensitivity.csv`. Post-episode returns are descriptive:
    the episode end is retrospective. They must not be fed back into a feature that claims to be known at onset.'''),
    code('''sensitivity=report.table('bursts','sensitivity.csv')
display(sensitivity.query("phase=='validation' and event_type=='all_bursts'"))
follow=report.table('followthrough','price_summary.csv').query(
    "phase=='validation' and side=='ALL' and horizon_s==20 and anchor=='episode_end'")
display(follow[['symbol','classification','valid','mean_post_signed_return_bps','reversal_share',
                'signed_mean_ci95_low','signed_mean_ci95_high']])'''),
    md('''Negative signed post-end returns describe reversals relative to the liquidation side.
    Selection on cascade price/volume and overlapping episodes limits interpretation.
    The follow-through statistical tables apply Holm adjustment across 42 tests; post-cascade return patterns do not
    survive that adjustment. This is not evidence for a tradable post-end rule.'''),
    md('''## Conclusion
    The detector and causal maker features are implemented. The distinction between an available event,
    a retrospective episode label and a future maker markout is essential. Next, test burst prediction separately
    before using predicted probabilities as maker features.''')])

    write('03_liquidation_prediction.ipynb',[
    md('''# 03 · Liquidation Prediction
    Do observable market states predict a new liquidation burst, and do these probabilities help the maker filter?

    This notebook distinguishes the historical directional-change experiments (`evt/`, `early_warning/`) from the newer
    side-specific Logistic Regression stacking experiment (`maker/`). Their populations, targets and splits differ.'''),code(SETUP),
    md('''## Historical DC / Markov / W–L experiment
    Directional changes use completed-second midpoint prices and thresholds 2.5/5/10 bps.
    A first-order transition matrix produces transition surprise; W averages the last 20 surprises and L is a calibrated
    normal-CDF transform. The Markov assumption is an approximation. Open interest is absent and is not replaced by volume.

    Historical splits: train through January 29, validation January 30–February 13, historical test February 14–28.
    The first 14 days fit nuisance estimators; a 305-second purge protects horizon boundaries.
    The final February segment had already appeared in EDA. It is not an independent hidden test.'''),
    code('''display(report.table('early_warning','calendar_split.csv').groupby(['split','role']).agg(days=('day','size')))
metrics=report.table('evt','modeling/test_metrics.csv')
display(metrics[['horizon_s','model','n','positives','mse','average_precision','precision','recall']])'''),
    code('''fig,axes=plt.subplots(1,2,figsize=(11,4))
for model,color in [('logistic','#d18a29'),('lightgbm','#24659a'),('constant_train_prevalence','#687681')]:
    q=metrics[metrics.model.eq(model)].sort_values('horizon_s')
    axes[0].plot(q.horizon_s,q.mse,marker='o',label=model,color=color)
    axes[1].plot(q.horizon_s,q.average_precision,marker='o',label=model,color=color)
axes[0].set(ylabel='Brier score (probability MSE)',xlabel='Horizon (seconds)',title='Historical February segment')
axes[1].set(ylabel='Average precision',xlabel='Horizon (seconds)',title='Burst prediction · DC event population')
axes[1].legend(fontsize=8);fig.tight_layout();show(fig,'burst_prediction.png')'''),
    md('''LightGBM improves historical Brier score and average precision relative to logistic across the saved horizons.
    Classification thresholds were selected on the earlier validation segment. Precision and recall refer to those frozen
    thresholds, not to a common alarm budget. Binary log loss is the classification training objective.'''),
    md('''## W–L ablation and reduced 5-second model
    The warning model's `full` versus `without_liquidity_score` comparison is distinct from E3/E4.
    The later 5-tree, depth-at-most-8, 5-lag revision did not improve the original test MSE;
    the negative result remains in the saved comparison.'''),
    code('''warning=report.table('early_warning','model_metrics.csv')
display(warning.query("split=='test' and horizon_s==5")[
    ['model','n','brier','average_precision','precision','recall']])
reduced=report.table('evt','reduced_5s/test_evaluation/comparison_by_split.csv')
display(reduced.query("split=='test'")[['model','mse','average_precision','precision','recall','brier_skill_vs_train_prior']])'''),
    md('''## Side-specific probabilities for maker models
    A separate StandardScaler + LogisticRegression(C=0.01) pipeline predicts buy/sell burst onset in `(t, t+5s]`.
    It trains on a deterministic 20% sample of the 5-second clock. Expanding weekly out-of-fold fits and a 10-second purge
    prevent probability features from using their own future labels. Historical DC probabilities are not reused as
    calibrated trade-clock probabilities.'''),
    code('''lr=report.table('maker','lr_validation_metrics.csv')
display(lr)
display(report.table('maker','lr_brier_comparison.csv').query("symbol=='combined' and block_days==5"))
display(report.table('maker','lr_upper_probability_bins.csv').query("symbol=='combined'"))'''),
    md('''For combined buy targets, Brier is about 0.001073 versus 0.001069 for the train-prior constant; there is no improvement.
    For sell targets it is about 0.001325 versus 0.001345. The saved block-bootstrap comparison supports the sell-side
    Brier improvement, but this does not establish an economic maker benefit. E5/E7 answer that separate question.'''),
    md('''## Conclusion
    Burst prediction is implemented and has useful ranking signal in these historical populations.
    Probability quality, event recall and maker Score are different evaluations. The reduced-model and W–L ablations
    do not establish a universal gain. No unseen test period has been evaluated.''')])

    write('04_maker_strategies.ipynb',[
    md('''# 04 · Maker Strategies
    Can liquidation features improve hypothetical maker trade selection at comparable clipped turnover?

    Original maker experiments use Binance trades/BBO and both liquidation feeds. Training is December–January;
    February is repeatedly inspected validation. Tables are frozen results over all eligible raw trades, not the fit sample.'''),code(SETUP),
    md('''## Labels, weights and baselines
    For taker sign `s`, trade price `p` and backward-as-of midpoint `m(t+h)`:
    `y = −s × (m(t+h) − p) / p × 10000 + 0.5` bps; `w = min(p × amount, 100000)`.
    A quote exactly at the target timestamp is allowed; targets outside global BBO bounds are excluded.
    The filter is `0 = keep, 1 = filter`. Score is weighted `PnL_kept − PnL_all`.

    E0 keeps all eligible trades. E0_random is a separate budget-matched control.
    E1 is the initial 5-second reconstruction; E1_60 is the corrected 60-second signed-flow rule;
    E1_60_strength additionally ranks eligible trades by prior-boundary absolute net liquidation notional.
    The latter is the selected default submission, a source-informed extension with no claim of baseline authorship.'''),
    code('''display(report.table('maker','experiment_matrix.csv'))
display(report.table('maker','baseline_summary.csv').query("split=='validation' and model=='E0'")[
    ['symbol','tau','pnl_all','score','kept_turnover_per_day']])'''),
    md('''## Ridge, LightGBM and loss
    E2 is standardized Ridge (alpha=100). E3 is a fixed 32-tree LightGBM regressor; E4 removes liquidation features
    and alignment. E5 adds chronological Logistic Regression probabilities; E6 adds six observable A–S risk proxies;
    E7 adds both. E8 is not identified: inventory, own fills, queue state and execution intensity are unavailable.

    Regression fits a deterministic 1/16 sample of feature-equivalence groups, with summed clipped-notional weights
    corrected by 16 and normalized to mean one. The target is the group's weighted mean markout, without clipping.
    Effective training starts December 22; `second + 301 < boundary` is enforced.
    Ridge minimizes weighted squared error plus its L2 penalty; LightGBM uses weighted squared error.
    Official Score is evaluated separately.'''),
    code('''spec=json.loads((artifact_dir('maker')/'experiment_specification.json').read_text())
display(pd.Series(spec['LGBMRegressor'],name='fixed parameter').to_frame())
display(report.primary(['E1_60_strength','E2','E3','E4','E5','E6','E7'])[
    ['model','tau','score','pnl_all','pnl_kept','pnl_filtered','kept_turnover_per_day']])'''),
    code("show(report.score_figure(['E1_60_strength','E2','E3','E4','E5','E6','E7']),'maker_scores.png')"),
    md('''## Controlled liquidation and stacking ablations
    E3−E4 isolates the presence of liquidation features under the fixed setup, not a fully optimized model-family comparison.
    February E3−E4 Scores are −12.662645, −4.126121 and +1.319553 bps at 30/120/300 seconds.
    The 5-day block intervals in the expanded retrospective family include zero at all three horizons.'''),
    code('''paired=report.table('maker','expanded_paired_comparisons.csv').query("scope=='combined' and block_days==5")
display(paired.query("(model=='E3' and comparator=='E4') or (model=='E5' and comparator=='E3') or (model=='E6' and comparator=='E3')"))
show(report.family_importance_figure(),'feature_families.png')'''),
    code("show(report.top_importance_figure(),'lightgbm_importance.png')"),
    md('''Liquidation features account for almost all E3 gain, yet removing them improves the short-horizon point Score.
    Split gain is model usage, not causal evidence or an economic attribution. Logistic additions change the 30-second
    Score by +0.728006 bps, without a positive adjusted paired interval; 120/300-second predictions are unchanged.
    A–S proxies have zero split gain in this setup, and E6 equals E3 while E7 equals E5.'''),
    md('''## Inference contract
    The default rule uses the published fixed thresholds and needs no fitted model.
    E7/CatBoost weights are intentionally withheld pending clarification of upstream dataset rights.
    Their preserved interfaces use the documented keep-all fallback when a model is absent;
    that fallback must not be described as a successful learned-model run.'''),
    code('''from submission import signal
sample=ROOT/'data/sample'
tr=pd.read_csv(sample/'binance_trades/perp_btcusdt.csv')
bb=pd.read_csv(sample/'binance_booktickers/perp_btcusdt.csv')
bn=pd.read_csv(sample/'binance_liquidations/perp_btcusdt.csv')
by=pd.read_csv(sample/'bybit_liquidations/btcusdt.csv')
filters=signal(tr,bb,bn,by)
assert set(filters)=={30,120,300}
display(pd.DataFrame([{'horizon_s':h,'synthetic_rows':len(v),'synthetic_kept_rows':int((v==0).sum())}
                      for h,v in filters.items()]))'''),
    md('''## Conclusion
    Fixed regressors and ablations are implemented, but these results do not show a robust incremental advantage from
    LR stacking or A–S proxies. Ridge/E4 lead at 30 seconds; the flow-strength rule has higher 120/300-second point Scores.
    Independent evaluation is still needed before selecting a strategy prospectively.''')])

    write('05_catboost_research.ipynb',[
    md('''# 05 · CatBoost Research
    Does changing E3's estimator to CatBoost improve weighted markout prediction or the economic filtering Score?

    E9 uses the exact saved E3 sample and 81-feature schema, with identical targets, weights and purges.
    It adds no feature search, early stopping or February fitting. Read frozen tables under `catboost/v1/`.'''),code(SETUP),
    md('''## Fixed estimator and provenance
    Three CatBoostRegressor models predict 30/120/300-second markouts.
    Training uses weighted RMSE (equivalently the same squared-error optimum) on grouped markout means.
    Full-trade MSE is reconstructed with within-group residuals, so a training RMSE is not the official evaluation MSE.'''),
    code('''spec=json.loads((artifact_dir('catboost')/'v1/experiment_specification.json').read_text())
from maker_hybrid.features import feature_columns
assert spec['features']==feature_columns('E3')
assert not spec['validation_labels_used_for_fit'] and not spec['hyperparameter_search']
display(pd.Series(spec['parameters'],name='fixed value').to_frame())
print('Feature count:',len(spec['features']))
print('Frozen training-sample SHA-256:',spec['training_sample_sha256'])'''),
    md('''## Maker Score and regression error
    The main comparison matches $250m/day total across BTC and ETH on February, using each model's fixed rank scores.
    February matching uses scores and weights, not markout labels, but is still a retrospective turnover calibration.
    January-threshold transfer is examined separately in notebook 06.'''),
    code('''display(report.primary(['E3','E9'])[['model','tau','score','pnl_all','pnl_kept','pnl_filtered','kept_turnover_per_day']])
show(report.score_figure(['E3','E9']),'catboost_scores.png')'''),
    code('''mse=report.table('catboost','v1/regression_error_diagnostics.csv').query("period=='validation' and symbol=='combined'")
display(mse)
show(report.mse_figure(),'regression_mse.png')'''),
    md('''E9 Score is 4.691114 / 10.377586 / −0.096945 bps versus E3's 5.148496 / 6.735727 / 5.531530.
    Its better 120-second point Score does not establish a general estimator improvement.
    A low global weighted MSE need not produce a strong Score in the small retained tail.'''),
    md('''## Paired statistical comparison
    The primary family contains E9−E3 at three horizons, combined $250m/day.
    The table includes 1-day and 5-day paired resampling; the chart uses the 5-day sensitivity with Bonferroni coverage.'''),
    code('''paired=report.table('catboost','v1/paired_e9_e3.csv').query('primary_family_bonferroni and block_days==5')
display(paired[['tau','delta_score','ci_low','ci_high','confidence_level','finite_bootstrap_fraction']])
assert ((paired.ci_low<=0)&(paired.ci_high>=0)).all()
show(report.paired_figure(),'catboost_paired_intervals.png')'''),
    md('''All adjusted paired intervals include zero. Bootstrap uncertainty does not correct for repeated strategy selection
    on February. This is descriptive evidence from a reused validation month.'''),
    code("show(report.top_importance_figure('catboost'),'catboost_importance.png')"),
    md('''## Conclusion
    The CatBoost extension is implemented, with frozen parameter/sample receipts and exact weighted-error accounting.
    It improves the 120-second point Score but loses on 30 and 300 seconds. No adjusted paired comparison establishes
    an incremental gain. Native fitted weights remain local because the supplied data's redistribution licence is unresolved.''')])

    write('06_evaluation_robustness.ipynb',[
    md('''# 06 · Strategy Evaluation & Robustness
    Are the results stable across horizons, turnover budgets, calendar days and threshold transfer?

    The saved maker evaluations cover February 1–28, 2026. All official scores use every eligible raw trade,
    represented losslessly by sufficient statistics when decisions coincide. No synthetic observations enter these tables.'''),code(SETUP),
    md('''## Matched-turnover comparison
    $250m/day here is the combined BTC + ETH clipped turnover, not $250m for each symbol.
    E0 keep-all has Score zero by definition and a much larger turnover. E0_random is the matched-budget control.
    The original constraint is at least $500,000 average clipped turnover per day; minimum-day turnover is diagnostic.'''),
    code('''display(report.score_matrix())
show(report.score_figure(),'strategy_scores.png')'''),
    code('''primary=report.primary()
display(primary[['model','tau','score','pnl_all','pnl_kept','pnl_filtered','kept_turnover_per_day',
                 'ci_low','ci_high','bootstrap_finite_fraction']])
assert primary.official_turnover_constraint.all()'''),
    md('''## Confidence intervals and attribution limits
    Pointwise Score intervals use whole UTC days. Paired ablations include moving/circular 5-day block sensitivity and
    declared multiplicity adjustments. At very small budgets some resamples retain no trades; nominal intervals are then
    missing. Conditional quantiles are kept in the CSV, but are not plotted as nominal 95% confidence intervals.'''),
    code('''expanded=report.table('maker','expanded_paired_comparisons.csv')
display(expanded.query("scope=='combined' and block_days==5 and comparator=='E1_60_strength'"))
small=report.table('maker','validation_report.csv')
display(small.loc[small.bootstrap_finite_fraction.lt(1),[
    'model','symbol','tau','budget_per_day','ci_low','ci_high','ci_status']].head(8))'''),
    md('''## Turnover and daily stability
    Matching the full month's average does not guarantee steady daily participation. Daily Score is undefined on a day
    with no retained weight. Those gaps remain missing rather than being replaced by zero.'''),
    code("show(report.turnover_figure(),'score_turnover.png')\nshow(report.daily_figure(),'daily_stability.png')"),
    code('''display(primary[['model','tau','days_with_kept_trades','minimum_daily_kept_turnover',
                 'fraction_days_below_500k','largest_day_share_of_kept_turnover','score_without_best_day']])
daily=report.daily_primary()
assert not daily.duplicated(['model','day']).any()
# Independent reconstruction of the monthly Score from daily sufficient statistics.
for model,g in daily.groupby('model'):
    total=g[['kept_w','kept_wy','all_w','all_wy']].sum()
    score=total.kept_wy/total.kept_w-total.all_wy/total.all_w
    expected=primary.query('model==@model and tau==120').score.iloc[0]
    np.testing.assert_allclose(score,expected,rtol=1e-11)'''),
    md('''## January thresholds transferred to February
    This comparison freezes January holdout thresholds and reports actual February turnover.
    It cannot be read as another matched-$250m/day ranking. Fixed thresholds never guarantee future turnover.'''),
    code('''transfer=report.table('maker','threshold_transfer_report.csv').query("scope=='combined' and budget_per_day==250000000")
display(transfer[['model','tau','score','kept_turnover_per_day','official_turnover_constraint']])
display(report.table('catboost','v1/threshold_transfer_report.csv')[
    ['model','tau','score','kept_turnover_per_day','fraction_days_below_500k']])'''),
    md('''## Stale quotes and liquidation price sensitivity
    The official label forward-fills the last available BBO even when stale. A fresh-quote diagnostic and a separate
    liquidation-price cleaning sensitivity are reported without silently redefining the official scorer.'''),
    code('''display(primary.query("model in ['E3','E4','E7','E9']")[
    ['model','tau','score','fresh_future_quote_score','kept_stale_weight_fraction']])
display(report.table('maker','liquidation_price_audit.csv'))
display(report.table('maker','price_cleaning_sensitivity_summary.csv'))'''),
    md('''## Conclusion
    The fixed experiments, controlled ablations and retrospective robustness checks are complete in the original workspace.
    These notebooks replay the published aggregate evidence and API examples without refitting.
    A fresh hidden period and an execution-aware simulator remain future work. Historical maker markout assumes hypothetical
    participation in the kept prints; it is not realized trading profit and contains no inventory, queue position,
    latency-aware order placement or cancellation simulation.''')])


if __name__=='__main__':build()
