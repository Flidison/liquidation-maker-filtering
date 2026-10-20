# Data access and schemas

Only **independently generated synthetic CSV fixtures** are distributed in `sample/`.
They contain no sampled or transformed original market observations. Regenerate them with:

```bash
python scripts/generate_sample.py
```

The artificial feeds cover 1,801 seconds for each symbol, include a constructed liquidation burst and support
causal feature/submission checks. They are neither a market simulation nor a source of research findings.

## Original sources

The local project contains eight supplied Parquet files, 1,315,384,304 rows, approximately 5.47 GB on disk.
They cover BTCUSDT and ETHUSDT perpetual futures from December 1, 2025 to February 28, 2026 UTC.
The original vendor/download route and contractual redistribution permission were not included.
The full dataset, raw-derived per-event caches, and fitted model weights are **not in Git or Git LFS**.
No external data release has been created.

Place legally obtained originals under this ignored layout, or set `LMF_DATA_ROOT` to the equivalent external directory:

```text
data/raw/
├── binance_trades/perp_btcusdt.parquet
├── binance_trades/perp_ethusdt.parquet
├── binance_booktickers/perp_btcusdt.parquet
├── binance_booktickers/perp_ethusdt.parquet
├── binance_liquidations/perp_btcusdt.parquet
├── binance_liquidations/perp_ethusdt.parquet
├── bybit_liquidations/btcusdt.parquet
└── bybit_liquidations/ethusdt.parquet
```

| Feed | Columns and Arrow types |
|---|---|
| Trades / liquidations | `timestamp: int64`, `ticker: string`, `side: string`, `price: double`, `amount: double` |
| BBO/bookticker | `timestamp: int64`, `ticker: string`, `bid_price: double`, `bid_amount: double`, `ask_price: double`, `ask_amount: double` |

- `timestamp` is **Unix UTC microseconds**, not seconds, milliseconds or pandas nanoseconds.
- Binance ticker: `perp:btcusdt` / `perp:ethusdt`; Bybit: `btcusdt` / `ethusdt`.
  The inference interface also recognizes case-insensitive BTCUSDT/ETHUSDT.
- In trades, `side` is the **taker side**. A taker buy implies a hypothetical maker sell.
- In liquidations, `side` is the liquidation order side. A buy closes a short and implies upward pressure.
- `amount` is base-asset quantity; notional is `price × amount` in USDT. The scorer clips each trade at 100,000.
- The input timestamp is preserved. The feature pipeline adds **+200,000 µs to Bybit availability once**;
  do not pre-shift Bybit files yourself.
- Equal-time events are excluded from causal features. Future label lookup includes quotes equal to the target.

The [source manifest](../artifacts/results/quality/source_manifest.csv) supplies exact original row counts, sizes
and SHA-256 hashes. A different vendor export can implement the same schema without reproducing these results.
No account IDs, personal identifiers or credentials occur in these schemas; only synthetic rows are released.

## Obtaining data lawfully

[Binance Public Data](https://github.com/binance/binance-public-data) documents public futures trade archives and
checksums. The current [Binance Vision dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
include noncommercial licensing and conditions on derivative works. Review the terms applicable to your download;
an exchange's public availability does not establish the licence of the supplied Parquet files.

[Bybit All Liquidation](https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation) documents the public
liquidation stream. It is a live feed, not a promise that this exact historical sample can be recovered.
Bybit's [API terms](https://www.bybit.com/common-static/compliance/legal/BYBIT/df1923006718fbba8ba70d7d762b9866.pdf)
restrict repackaging/reselling Service Data and do not provide a general redistribution grant.

For the exact historical study, request the original files **and permission for your intended use** from their supplier,
or obtain a licensed historical archive covering all four feed types. No verified free source reproducing the exact
historical BBO and both liquidation feeds was established. Do not substitute aggregated trades or resampled candles
and call the original full-trade evaluation reproduced. See [rights review](../docs/rights_and_sources.md).
