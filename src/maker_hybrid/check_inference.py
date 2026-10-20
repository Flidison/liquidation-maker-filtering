"""Integration check: raw-frame inference agrees with the cached feature pipeline."""
from .core import *
from .submission import market_batches,signal_with_config,_model,signal
from .features import feature_columns,trade_features
from .policy import full_group_keep


def run():
    policies=json.loads((OUT/'budget_policies.json').read_text())
    cfg={'policies':{str(h):{'combined':next(p for p in policies if p['model']=='E7' and p['tau']==h and p['scope']=='combined' and p['calibration_period']=='validation' and p['budget_per_day']==250000000)} for h in HORIZONS},
         'features':{k:feature_columns(k) for k in ['LR','E7']}}
    receipts=[]
    for symbol in SYMBOLS:
        day='2026-02-03';start=int(pd.Timestamp(day+' 12:00',tz='UTC').timestamp());end=start+120
        def read(kind,lo,hi):return pl.scan_parquet(source(symbol,kind)).filter((pl.col('timestamp')>=lo*US)&(pl.col('timestamp')<hi*US)).collect().to_pandas()
        tr=read('binance_trades',start-1300,end)
        b=read('binance_booktickers',start-1400,end+400)
        bn=pd.read_parquet(source(symbol,'binance_liquidations'));by=pd.read_parquet(source(symbol,'bybit_liquidations'))
        cache=pd.read_parquet(OUT/'states'/symbol/(day+'.parquet')).set_index('second')
        idx,raw,g,inverse,state,arrays=list(market_batches(tr,b,bn,by,symbol))[0]
        check=state[state.second>=start].set_index('second')
        maximum=float(np.max(np.abs(check.to_numpy()-cache.loc[check.index].to_numpy())))
        np.testing.assert_allclose(check,cache.loc[check.index],rtol=2e-5,atol=2e-5)
        prob=state[['second']].copy()
        for side in ['buy','sell']:prob[f'p_{side}_burst_5s']=_model(f'lr_{side}_final.joblib').predict_proba(state[feature_columns('LR')])[:,1]
        xx=trade_features(g,state,prob)
        pred=pd.read_parquet(OUT/'predictions'/'final'/symbol/(day+'.parquet'))
        compare=g.merge(pred,on=['second','taker_sign','alignment'],how='left',validate='one_to_one')
        m=g.second>=start
        for h in HORIZONS:
            pp=_model(f'E7_{h}_final.joblib').predict(xx[feature_columns('E7')].astype('float32')).astype('float32')
            np.testing.assert_array_equal(pp[m],compare.loc[m,f'E7_{h}'])
        full=signal_with_config(tr,b,bn,by,cfg)
        cutoff=(start+60)*US
        take=tr.timestamp<=cutoff
        short=signal_with_config(tr.loc[take],b[b.timestamp<=cutoff],bn[bn.timestamp<=cutoff],by[by.timestamp+200000<=cutoff],cfg)
        for h in HORIZONS:np.testing.assert_array_equal(full[h][take],short[h])
        # Reorder distinct timestamps; output must follow the input positions.
        permutation=np.arange(len(tr))[::-1]
        reordered=signal_with_config(tr.iloc[permutation],b,bn,by,cfg)
        for h in HORIZONS:np.testing.assert_array_equal(reordered[h],full[h][permutation])
        default=signal(tr,b,bn,by)
        default_prefix=signal(tr.loc[take],b[b.timestamp<=cutoff],bn[bn.timestamp<=cutoff],by[by.timestamp+200000<=cutoff])
        for h in HORIZONS:np.testing.assert_array_equal(default[h][take],default_prefix[h])
        receipts.append({'symbol':symbol,'input_rows':len(tr),'checked_prediction_rows':int(m.sum()),
                         'maximum_feature_difference':maximum,'cached_float32_predictions_equal':True,
                         'future_truncation_equal':True,'reordered_output_equal':True,'default_rule_prefix_equal':True})
        print(receipts[-1],flush=True)
    save_json(OUT/'inference_integration_checks.json',receipts)


if __name__=='__main__':run()
