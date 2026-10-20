"""Research invariants that need only the released aggregates and synthetic feeds."""
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from research_paths import ROOT, artifact_dir
import research_report as report


def test_frozen_scores_reconcile_from_daily_sufficient_statistics():
    frames=[report.table('maker','policy_daily.csv'),report.table('maker','baseline60_daily.csv'),
            report.table('catboost','v1/policy_daily.csv')]
    for daily in frames:
        keys=['model','scope','tau','budget_per_day','calibration_period']
        totals=daily.groupby(keys)[['all_w','kept_w','all_wy','kept_wy']].sum().reset_index()
        original=report.table('catboost','v1/comparison_report.csv')
        merged=totals.merge(original,on=keys,validate='one_to_one',suffixes=('','_report'))
        assert len(merged)>0
        score=merged.kept_wy/merged.kept_w-merged.all_wy/merged.all_w
        np.testing.assert_allclose(score,merged.score,rtol=2e-10,atol=2e-10,equal_nan=True)
        np.testing.assert_allclose(merged.all_wy/merged.all_w,merged.pnl_all,rtol=2e-10,atol=2e-10)


def test_e3_numbers_in_catboost_comparison_are_unchanged():
    old=report.table('maker','all_validation_results.csv')
    new=report.table('catboost','v1/comparison_report.csv')
    legacy=new[new.model.ne('E9')]
    assert list(legacy)==list(old)
    pd.testing.assert_frame_equal(old.reset_index(drop=True),legacy.reset_index(drop=True))
    assert report.primary(['E9']).score.iloc[-1]<0  # Retain the negative 300s result.


def test_raw_source_counts_match_audit():
    q=report.table('quality','summary.csv')
    m=report.table('quality','source_manifest.csv')
    assert len(q)==len(m)==8
    assert m.rows.sum()==q.rows.sum()==1_315_384_304
    np.testing.assert_array_equal(q.sort_values('file').rows,m.sort_values('source_file').rows)


def test_extracted_histograms_preserve_every_count():
    h=report.table('quality','distribution_histograms.csv')
    q=report.table('quality','distribution_quantiles.csv')
    totals=h.groupby(['source_id','ticker','metric'])['count'].sum().sort_index()
    expected=q.set_index(['source_id','ticker','metric'])['count'].sort_index()
    np.testing.assert_array_equal(totals,expected)


def test_numeric_function_definitions_match_original_audit():
    manifest=json.loads((ROOT/'configs/numerical_function_hashes.json').read_text())
    for item in manifest:
        tree=ast.parse((ROOT/item['file']).read_text())
        fn=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==item['name'])
        actual=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
        assert actual==item['source_ast_sha256'],(item['file'],item['name'])


def test_sample_schema_submission_and_actual_rule_decisions():
    from submission import signal
    from maker_hybrid.submission import signal_with_config
    frames={}
    for sym in ['btcusdt','ethusdt']:
        paths={'trades':f'binance_trades/perp_{sym}.csv','bbo':f'binance_booktickers/perp_{sym}.csv',
               'bn':f'binance_liquidations/perp_{sym}.csv','by':f'bybit_liquidations/{sym}.csv'}
        for key,p in paths.items():frames.setdefault(key,[]).append(pd.read_csv(ROOT/'data/sample'/p))
    frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    r=signal(frames['trades'],frames['bbo'],frames['bn'],frames['by'])
    assert set(r)=={30,120,300}
    for arr in r.values():
        assert arr.dtype==np.uint8 and len(arr)==len(frames['trades'])
        assert np.isin(arr,[0,1]).all()
    # Demonstrate an active rule rather than allowing an absent-config fallback to pass.
    cfg={'policies':{str(h):{'combined':{'model':'E1_60','native':True}} for h in r}}
    native=signal_with_config(frames['trades'],frames['bbo'],frames['bn'],frames['by'],cfg)
    assert set(native[30])=={0,1}


def test_publication_has_no_raw_parquet_or_serialized_weights():
    assert not list((ROOT/'data').rglob('*.parquet'))
    for extension in ['*.joblib','*.cbm','*.pkl']:
        assert not list((ROOT/'artifacts').rglob(extension))


def test_rebuild_staging_copies_only_code_and_fixed_specifications(tmp_path):
    from scripts.reproduce import initialize
    work=tmp_path/'new-research';data=tmp_path/'authorized'
    initialize(work,data)
    assert (work/'src/maker_hybrid/core.py').exists()
    assert (work/'artifacts/results/maker/experiment_specification.json').exists()
    assert not list((work/'artifacts').rglob('*complete.json'))
    assert not list((work/'artifacts').rglob('*.csv'))
    assert not (work/'data/raw').exists()
    initialize(work,data)  # An explicitly marked workspace may resume.
    with pytest.raises(ValueError,match='Data root changed'):
        initialize(work,tmp_path/'different-input')


def test_rebuild_refuses_an_unrelated_existing_directory(tmp_path):
    from scripts.reproduce import initialize
    old=tmp_path/'existing';old.mkdir();sentinel=old/'important.txt';sentinel.write_text('preserve')
    with pytest.raises(ValueError,match='Refusing'):
        initialize(old,tmp_path/'data')
    assert sentinel.read_text()=='preserve'
