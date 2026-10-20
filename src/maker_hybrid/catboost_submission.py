"""Independent E9 inference. The existing signal and hybrid_signal are untouched."""
from functools import lru_cache
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .core import ROOT, HORIZONS, SYMBOLS, as_pandas, policy_hash, digest
from .submission import market_batches, _symbol, _empty_liq
from .features import trade_features
from .policy import full_group_keep, trade_hash

DEFAULT_ARTIFACTS = ROOT/'artifacts/results/catboost'/'v1'


@lru_cache(maxsize=12)
def _load_model(path, expected_sha256):
    from catboost import CatBoostRegressor
    if digest(path) != expected_sha256:
        raise ValueError(f'E9 model checksum mismatch: {path}')
    return CatBoostRegressor().load_model(str(path))


def catboost_signal(trades, bbo, liq_binance, liq_bybit, *, artifact_dir=DEFAULT_ARTIFACTS, config=None):
    """Return horizon -> uint8 filter (0 keep / 1 filter), in original row order.

    No training, labels, future ranking, or raw-market disk reads. Supply at least
    1,205 seconds of liquidation history and 61 seconds of trades/BBO context.
    Missing configuration/model/BBO uses the same fixed keep-all fallback as E7.
    An explicit config permits frozen January thresholds or another version.
    """
    trades = as_pandas(trades)
    result = {h:np.zeros(len(trades), dtype=np.uint8) for h in HORIZONS}
    if not len(trades):
        return result
    artifact_dir = Path(artifact_dir)
    if config is None:
        try:
            config = json.loads((artifact_dir/'catboost_config.json').read_text())
        except (OSError, ValueError):
            warnings.warn('E9 configuration unavailable; fixed keep-all fallback', RuntimeWarning)
            return result
    if 'ticker' not in trades:
        raise ValueError('Official input requires ticker; refusing to guess BTC versus ETH')
    bbo = as_pandas(bbo) if bbo is not None else pd.DataFrame()
    bn = as_pandas(liq_binance) if liq_binance is not None else _empty_liq()
    by = as_pandas(liq_bybit) if liq_bybit is not None else _empty_liq()
    symbols = trades.ticker.astype(str).str.lower().str.replace('perp:', '', regex=False).to_numpy()
    for symbol in SYMBOLS:
        positions = np.flatnonzero(symbols == symbol)
        if not len(positions):
            continue
        tr, bb = trades.iloc[positions], _symbol(bbo, symbol)
        if bb is None or not len(bb):
            continue
        lb, ly = _symbol(bn, symbol), _symbol(by, symbol)
        if lb is None or not len(lb):lb = _empty_liq()
        if ly is None or not len(ly):ly = _empty_liq()
        active = {}
        for h in HORIZONS:
            ps = config['policies'].get(str(h), {})
            policy = ps.get(symbol, ps.get('combined'))
            if not policy or policy.get('model') == 'keep_all':
                continue
            if policy.get('model') != 'E9':
                raise ValueError('catboost_signal only accepts E9 policies')
            try:
                info = config['models'][str(h)]
                model = _load_model(str(artifact_dir/info['file']), info['sha256'])
                if model.feature_names_ != config['features']:
                    raise ValueError('E9 feature schema mismatch')
            except (ImportError, OSError, ValueError) as exc:
                warnings.warn(f'E9 {h}s model unavailable; fixed keep-all fallback ({type(exc).__name__})', RuntimeWarning)
                continue
            active[h] = (policy, model)
        if not active:
            continue
        for ix, raw, groups, inverse, states, _ in market_batches(tr, bb, lb, ly, symbol):
            x = trade_features(groups, states)[config['features']].astype('float32')
            key = policy_hash(groups.second, symbol, groups.taker_sign, groups.alignment)
            for h, (policy, model) in active.items():
                scores = model.predict(x, thread_count=4).astype('float32')
                kg, bg = full_group_keep(scores, key, groups.alignment, policy)
                keep, boundary = kg[inverse], bg[inverse]
                if boundary.any():
                    keep[boundary] = trade_hash(raw.loc[boundary], symbol) <= policy['trade_tie_cut']
                result[h][positions[ix]] = (~keep).astype(np.uint8)
    return result
