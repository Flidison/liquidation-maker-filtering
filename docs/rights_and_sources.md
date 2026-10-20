# Rights and sources

Review date: **October 20, 2026**. This records the publication decision and evidence; it does not infer a licence
from the fact that exchange messages are observable.

## Supplied material

The research workspace and handoff archive contain original project Python code, notebooks, compact experiment outputs,
trained models and eight market-data Parquets. No `LICENSE`, supplier redistribution agreement, acquisition manifest or
data purchase contract was found for those inputs. The user identified the code/research as their project and authorized
public publication. The sibling `mft_proj` directory was inspected for context; it is a different study and contributes no files.

The new MIT licence covers project-authored code, documentation and artificial fixtures. It does not assert ownership
of established methods or licence third-party feeds. Historical baseline reference slides, a queue-model presentation,
private handoff prose and extracted third-party slide text are omitted. E2/E3 are described as current reconstructions;
the exact historical reference weights/cleaning code were not located.

## Exchange sources checked

The official [Binance public-data repository](https://github.com/binance/binance-public-data) documents futures trade
archives and checksums. Its [Vision dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
(version 1.0, August 26, 2026) specify CC BY-NC-SA 4.0 with additional conditions and refer to derivative analyses.
Those terms are not evidence that the supplied files came from Vision or were acquired under that version.

The official [Bybit liquidation documentation](https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation)
describes the live public channel. The checked [Bybit API terms](https://www.bybit.com/common-static/compliance/legal/BYBIT/df1923006718fbba8ba70d7d762b9866.pdf),
sections 5 and 6.7, do not supply a broad redistribution grant and prohibit repackaging/reselling Service Data.
No historic collection contract for this dataset was available.

## Release boundary

- Publish project implementations, parameters, research documentation, non-reconstructive aggregate findings and figures.
- Publish independently generated artificial feeds under the project's MIT scope.
- Withhold every original tick, per-event market-data export, raw-derived cache and fitted model object pending
  clarification of the supplied data's rights. No Git LFS or external mirror is used.
- Do not attach a broad data licence to aggregate results that purports to override upstream rights.
  Their inclusion permits inspection of the research; it is not a grant to reuse a third-party feed.

No account identifiers or credentials are present in the published fixtures. Local user paths are removed from copied
text and notebook outputs. The source manifests retain SHA-256 provenance without redistributing the original files.
This restricted scope permits publication of the research code and findings while leaving unresolved data rights open.

## Method references

The observable risk features are inspired by [Avellaneda and Stoikov, *High-frequency trading in a limit order book*](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf).
The implementation only uses observable volatility/spread proxies; it is not a calibration of that inventory/quoting model.
The DC/Markov experiment specification also records its [directional-change reference](https://arxiv.org/html/1402.2198).
These references are cited neutrally, without claiming authorship of their methods or redistributing the papers.
