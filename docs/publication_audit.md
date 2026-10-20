# Publication audit

Prepared October 20, 2026 as a separate checkout with a new Git history. The original working project was not
reorganized, deleted or edited. Original source/notebook hashes and eight raw-source hashes were checked again.

## Sources inspected

All eight requested notebooks were inspected: Data Loading, Data Quality, Liquidation Bursts, Liquidation Follow-through,
Liquidation Early Warning, EVT_log_cond, Maker Hybrid Research, and CatBoost Maker Research. The current `maker_hybrid/`
sources, root submission entry points, tests, exact configs, daily reports and robustness outputs were reviewed.

`Liquidation_GPT_Full_Package_2026-10-18.zip` was available. Its inventory and research handoff context were checked:
415 entries, including seven notebooks, old snapshots, figures, tables, code and private historical-reference prose.
It predates the E9 extension. Current root sources/results take precedence. The sibling `mft_proj` is a separate study;
no files from it enter this release. No licence or redistribution receipt accompanied the supplied data.

## Notebook mapping

| Publication notebook | Original research |
|---|---|
| 01 Data Exploration & Quality | Data Loading + Data Quality audit summaries and synthetic loading example |
| 02 Liquidation Activity & Features | Bursts + follow-through; causal maker feature implementation |
| 03 Liquidation Prediction | Early Warning + EVT original/reduced models + maker OOF Logistic Regression |
| 04 Maker Strategies | Maker Hybrid model definitions, E0–E7 scores and controlled ablations |
| 05 CatBoost Research | Current E9 study and exact E3 comparison |
| 06 Strategy Evaluation & Robustness | Maker/E9 scores, paired intervals, daily/turnover/transfer/cleaning checks |

Each notebook has a question, source scope, executable table/figure calculations, interpretation and conclusion.
They are evidence companions using frozen aggregate tables, with artificial examples labeled separately.
They do not masquerade as fresh full-data experiment executions.

## Code changes

The existing computational modules are retained under `src/`; no new modeling framework was introduced.
`research_paths.py` supplies repository/data paths. Path-bearing source/receipt lookups were adapted to relocation.
The original quality notebook's reusable functions were extracted to `data_quality_audit.py` with explicit orchestration;
its file identifiers keep the original `data__...` convention.

248 original function/class ASTs compare identically, including maker labels/scoring, features, hash keys, regression,
classification and bootstrap computations. Their hashes are in `configs/numerical_function_hashes.json`.
Path-bearing function changes are limited to source/manifest verification, `burst_onsets` file location,
manifest generation and quality `file_id`. Imports/output roots are relocated outside function bodies.
`research_report.py` and notebook-authoring scripts are new presentation utilities, not new scientific metrics.

Original portable tests were retained. Full-data/model cases are explicitly marked `integration`, with default
deselection. The obsolete whole-workspace E9 byte-preservation assertion was omitted because this release intentionally
changes organization; original-workspace preservation was checked separately. New checks reconcile monthly Score from
daily sufficient statistics, preserve old rows in the E9 comparison, and verify histogram totals, sample schemas and ASTs.

## Retained and omitted

Retained: reusable algorithms, rules, Ridge/logistic/LightGBM/CatBoost implementations, inference, tests, exact
hyperparameters/threshold policies, compact aggregate results, audit hashes, and figures regenerated from those results.

Omitted: raw Parquets, per-event exports, prediction/training/group/state caches, serialized fitted models and LR
coefficient/scaler exports, superseded colliding-key policies, old notebook versions and snapshots, duplicate HTML/PDF,
archives/backups, old report generators/package scripts, third-party presentations/text, unrelated images, environments,
bytecode, debug logs and local machine paths. Omitted sources remain in the original project. No large data was added
to Git history, Git LFS, Releases or another host.

Frozen JSON configurations remain beside corresponding result tables because the original code reads that layout.
Historical model/sample hashes can reference intentionally absent objects; they record provenance, not availability.
Historical verification receipts report 147 maker tests and 156 E9-era tests with no failures/errors;
`configs/historical_validation.json` records their hashes. These are distinguished from the new 117 portable checks.

## Checks and remaining gaps

- Six notebook formats validated and executed top-to-bottom without raw data or model fitting.
- 117 portable cases pass; 47 private-data/model cases are explicitly deselected.
- Monthly Scores reconcile with original daily aggregate sums; old comparison rows remain numerically unchanged.
- All eight local source Parquet hashes still match the original full audit (1,315,384,304 rows).
- The dependency set installs into a new Python 3.11 virtual environment; portable tests run there.
- Publication scanner checks relative links, all included hashes, notebook execution/error outputs, forbidden artifacts,
  private paths, credential patterns and every Git-history blob before push.
- Full raw-data re-audit/refitting was not repeated. Private models/data are intentionally not publicly redistributable
  based on the available evidence. Fresh independent evaluation and execution-aware simulation remain unperformed.

The publication validation receipt records final local/CI/render checks separately. Research-status claims are scoped
to the evidence actually present; negative and uncertain comparisons are retained.
