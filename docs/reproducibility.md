# Reproducibility

There are two distinct runs: **public aggregate replay** and **private full-data reproduction**.
They answer different questions. Replay checks the released evidence and presentation; it does not regenerate
the original model fits or full raw-data labels.

## Public replay

Use Python 3.11 and the pinned `requirements.txt`, then install the project editable as shown in the README.
Notebook kernels are explicitly launched with the executing Python interpreter. Installing JupyterLab is optional;
`nbclient` is enough to execute the six notebooks headlessly.

```bash
python -m pytest -q
python scripts/verify_publication.py
python scripts/execute_notebooks.py
# CI-style execution without overwriting the saved notebooks:
python scripts/execute_notebooks.py --check
```

This runs 117 portable checks; 47 full-data/model cases are explicitly deselected by the `integration` marker.
The default suite covers label signs/rebate/clipping, temporal boundaries, Bybit latency, future-prefix invariance,
feature allowlists, exact uint64 ties, grouped/full-trade loss equivalence, statistical routines and independent
monthly-Score reconstruction from released daily aggregates. It also checks the original AST hashes of 248 unchanged
functions/classes. The private raw data and model checks are not claimed to have run in a public clone.

Artificial fixtures are generated with `scripts/generate_sample.py`. The default `submission.signal` works with
these feeds and published threshold policies; it is E1_60_strength, with 0 keep / 1 filter. At least 1,205 seconds of
liquidation history and 61 seconds of trade/BBO context are needed for complete chunked features. Preserve every
equal-timestamp message when constructing chunks, and select current-chunk outputs after supplying the context.

## Original data and fitted models

Obtain authorization and the exact eight sources in [data/README.md](../data/README.md). Set:

```bash
export LMF_DATA_ROOT=/path/to/authorized/data
```

For original-data equality, compare source sizes, row counts and SHA-256 values with
`artifacts/results/quality/source_manifest.csv`. The publication audit verified all eight local source hashes again.
Schema-compatible data from another collection is a new experiment, not an exact reproduction.

If already-authorized local fitted objects are restored instead of refitted, keep them ignored by Git:

| Location in publication checkout | Required objects |
|---|---|
| `artifacts/results/maker/models/` | `lr_buy_final.joblib`, `lr_sell_final.joblib`, `E7_30_final.joblib`, `E7_120_final.joblib`, `E7_300_final.joblib` for E7 inference |
| same directory | E2–E6 final/holdout and dated LR objects for historical fit checks |
| `artifacts/results/catboost/v1/models/` | `E9_30_final.cbm`, `E9_120_final.cbm`, `E9_300_final.cbm` for E9 inference |

Native E9 model hashes are recorded in `training_complete.json` and `catboost_config.json`. Maker model hashes are in
`regressors_complete.json`. Restore trusted files only; joblib deserialization executes Python objects.
No historical market cache is needed just to run inference on caller-supplied frames with those weights.

```python
from submission import hybrid_signal
from catboost_submission import catboost_signal
# Both require restored trusted weights for actual learned-model decisions.
e7_filters = hybrid_signal(trades, bbo, liq_binance, liq_bybit)
e9_filters = catboost_signal(trades, bbo, liq_binance, liq_bybit)
```

Absent weights trigger the original documented keep-all fallback. Such an output is not evidence that E7/E9 has
been reproduced. Thresholds were calibrated on reused February scores/weights; future turnover is not guaranteed.

## Full rebuild in a separate workspace

The original research outputs are retained as frozen evidence. Do not train over those completion markers.
The helper below creates a **new private workspace**, copies code/tests, and seeds only the exact fixed maker/split
specifications. It does not copy historical completion markers, result tables or model objects.
The source data is read from the external path and never copied or modified.

First inspect the command plan; this performs no computation:

```bash
python scripts/reproduce.py \
  --workspace /path/to/new-private-reproduction \
  --data-root /path/to/authorized/data
```

Then run selected stages, in dependency order, with the same arguments plus `--execute`:

```bash
python scripts/reproduce.py \
  --workspace /path/to/new-private-reproduction \
  --data-root /path/to/authorized/data \
  --stage quality,bursts,followthrough,warning,evt,maker,catboost --execute
```

| Stage | Computation and dependencies |
|---|---|
| `quality` | Original all-row audit; exact distributions, disk-backed quantiles, issue/second-level caches |
| `bursts` | Verifies quality/source hashes; calibrates detector, builds primary windows, sensitivity, episodes and manifests |
| `followthrough` | Uses burst caches; measures retrospective episode outcomes and the 42-test statistical family |
| `warning` | Uses detector/second caches; fits DC/transition/calibration state and warning classifiers |
| `evt` | Creates verified source catalog; fits original logistic/LightGBM comparison, reduced model and historical test assessment |
| `maker` | Full-trade groups/labels, features, baselines, OOF LR, Ridge/LightGBM, all-row evaluation, transfer, diagnostics and configs |
| `catboost` | Uses the newly built E3 sample and models; trains/evaluates E9 in its own versioned directory |

The helper can resume its own marked workspace. Code/config/data changes require a fresh workspace/version; do not
bypass the original immutable-cache checks. The `--stage maker` path requires the preceding quality/burst products;
`--stage catboost` requires maker outputs. Separate historical warning/EVT fits are not required for the new maker LR,
but are needed to regenerate their classification reports.

Expect substantial disk use: sources are about 5.47 GB compressed, and the existing derived artifacts exceed 15 GB.
The quality audit also writes disk-backed temporary distributions. Allow generous additional scratch space.
No measured complete-run duration or minimum RAM requirement is claimed. Original routines cap several workers at four
and use a 4 GB DuckDB memory limit, but that is not a bound on total process memory.

**Publication validation did not execute this complete expensive rebuild.** The preserved computational routines,
portable tests, staging logic and dependency plan were checked; full fresh-data fitting remains a resource-dependent step.
Bitwise equality of new tree fits is not guaranteed across hardware/library builds, even with the recorded seeds.
Frozen aggregate results were not changed to match a new fit.

To exercise restored or freshly regenerated private integration checks explicitly:

```bash
cd /path/to/new-private-reproduction
PYTHONPATH=src:. python -m pytest -q -m integration
```

These tests expect the full historical derived artifacts and matching lineage receipts. Restoring only raw sources
or a few final model files is insufficient. The obsolete assertion that every old workspace file is unchanged was
omitted from the publication suite; this is intentionally a reorganization. Original sources remain preserved locally.
