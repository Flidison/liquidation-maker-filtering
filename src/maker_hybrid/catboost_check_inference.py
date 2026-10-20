"""Real BTC/ETH raw-row checks against E9 group caches, including boundary ties."""
import copy
import json

import numpy as np
import pandas as pd
import polars as pl

from .core import OUT, HORIZONS, SYMBOLS, US, source, policy_hash
from .catboost_experiment import E9_OUT, immutable_json
from .catboost_submission import catboost_signal, _load_model
from .submission import market_batches
from .features import trade_features
from .policy import full_group_keep, trade_hash


def run():
    cfg = json.loads((E9_OUT/'catboost_config.json').read_text())
    receipts = []
    for symbol in SYMBOLS:
        day = '2026-02-03'
        start = int(pd.Timestamp(day+' 12:00',tz='UTC').timestamp())
        end = start+120
        def read(kind, lo, hi):
            return pl.scan_parquet(source(symbol,kind)).filter(
                (pl.col('timestamp')>=lo*US)&(pl.col('timestamp')<hi*US)).collect().to_pandas()
        tr = read('binance_trades', start-1300, end)
        b = read('binance_booktickers', start-1400, end+400)
        bn = pd.read_parquet(source(symbol,'binance_liquidations'))
        by = pd.read_parquet(source(symbol,'bybit_liquidations'))
        idx, raw, g, inverse, state, _ = list(market_batches(tr,b,bn,by,symbol))[0]
        x = trade_features(g,state)[cfg['features']].astype('float32')
        pred = pd.read_parquet(E9_OUT/'predictions'/'final'/symbol/(day+'.parquet'))
        compare = g.merge(pred,on=['second','taker_sign','alignment'],how='left',validate='one_to_one')
        check = g.second>=start
        key = policy_hash(g.second,symbol,g.taker_sign,g.alignment)
        full = catboost_signal(tr,b,bn,by)
        tie_config = copy.deepcopy(cfg)
        expected_tie = {}
        for h in HORIZONS:
            info = cfg['models'][str(h)]
            model = _load_model(str(E9_OUT/info['file']),info['sha256'])
            pp = model.predict(x,thread_count=4).astype('float32')
            np.testing.assert_array_equal(pp[check],compare.loc[check,f'E9_{h}'])
            policy = cfg['policies'][str(h)]['combined']
            kg,bg = full_group_keep(pp,key,g.alignment,policy)
            keep,boundary = kg[inverse],bg[inverse]
            if boundary.any():keep[boundary] = trade_hash(raw.loc[boundary],symbol)<=policy['trade_tie_cut']
            np.testing.assert_array_equal(full[h][idx],~keep)
            # Force an actual observed group onto the threshold to exercise raw ties.
            j = int(np.flatnonzero(check)[0])
            p = dict(policy,score_threshold=float(pp[j]),group_tie_cut=int(key[j]),trade_tie_cut=.5)
            tie_config['policies'][str(h)]['combined'] = p
            kg,bg = full_group_keep(pp,key,g.alignment,p)
            keep,boundary = kg[inverse],bg[inverse]
            assert boundary.any()
            keep[boundary] = trade_hash(raw.loc[boundary],symbol)<=.5
            expected_tie[h] = (~keep).astype(np.uint8)
        tied = catboost_signal(tr,b,bn,by,config=tie_config)
        for h in HORIZONS:np.testing.assert_array_equal(tied[h][idx],expected_tie[h])
        cutoff = (start+60)*US
        take = tr.timestamp<=cutoff
        short = catboost_signal(tr.loc[take],b[b.timestamp<=cutoff],bn[bn.timestamp<=cutoff],by[by.timestamp+200000<=cutoff])
        reordered = catboost_signal(tr.iloc[::-1],b,bn,by)
        for h in HORIZONS:
            np.testing.assert_array_equal(full[h][take],short[h])
            np.testing.assert_array_equal(full[h][::-1],reordered[h])
        receipts.append(dict(symbol=symbol,input_rows=len(tr),checked_prediction_groups=int(check.sum()),
                             cached_float32_predictions_equal=True,signal_matches_group_decisions=True,
                             raw_boundary_ties_equal=True,future_truncation_equal=True,reordered_output_equal=True))
        print(receipts[-1],flush=True)
    immutable_json(E9_OUT/'inference_integration_checks.json', receipts)


if __name__ == '__main__':run()
