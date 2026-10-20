"""Plan or run the original expensive pipeline in a separate, private workspace.

This command does not run by default. --execute creates a new workspace, seeds
only fixed specifications, and starts the explicitly selected stages.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STAGES={
 'quality': [('module','data_quality_audit')],
 'bursts': [('code',"import liquidation_bursts as m; m.verify_sources(); m.calibrate_and_explore(); m.run_scenario(save=True); m.sensitivity(); m.reconcile_and_check_duplicates(); m.check_all_cascade_trade_confirmations(); m.direction_comparison(); m.write_run_manifest()")],
 'followthrough': [('module','liquidation_followthrough'),('module','liquidation_statistical_validation')],
 'warning': [('module','liquidation_early_warning'),('module','liquidation_early_warning_evaluation')],
 'evt': [('script','initialize_evt.py'),('module','evt_log_cond_modeling'),
         ('module','evt_log_cond_reduced_5s'),('module','evt_log_cond_test_reduced_5s'),
         ('module','evt_log_cond_calibration_5s')],
 'maker': [('module',x) for x in ['maker_hybrid.prepare','maker_hybrid.features','maker_hybrid.baselines',
          'maker_hybrid.logistic','maker_hybrid.regression','maker_hybrid.evaluate','maker_hybrid.prepare60',
          'maker_hybrid.baseline60','maker_hybrid.price_audit','maker_hybrid.price_sensitivity',
          'maker_hybrid.holdout_transfer','maker_hybrid.diagnostics','maker_hybrid.release',
          'maker_hybrid.check_inference']],
 'catboost': [('module','maker_hybrid.catboost_experiment'),('module','maker_hybrid.catboost_evaluate'),
              ('module','maker_hybrid.catboost_check_inference')],
}


def initialize(workspace,data_root):
    if workspace.exists():
        marker=workspace/'configs/rebuild_workspace.json'
        if not marker.exists():raise ValueError('Refusing an existing directory without a rebuild-workspace marker')
        meta=json.loads(marker.read_text())
        if meta['data_root']!=str(data_root):raise ValueError('Data root changed; create a fresh workspace')
        return
    workspace.mkdir(parents=True)
    for folder in ['src','tests','scripts']:
        shutil.copytree(ROOT/folder,workspace/folder,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for file in ['submission.py','catboost_submission.py','pyproject.toml','requirements.txt','README.md','.gitignore']:
        shutil.copy2(ROOT/file,workspace/file)
    (workspace/'configs').mkdir()
    for group in ['quality','bursts','followthrough','statistics','early_warning','evt/modeling','maker','catboost/v1']:
        (workspace/'artifacts/results'/group).mkdir(parents=True)
    # Do not copy completion markers, historical results or fitted artifacts.
    for file in ['maker/experiment_specification.json','evt/split_specification.json']:
        shutil.copy2(ROOT/'artifacts/results'/file,workspace/'artifacts/results'/file)
    (workspace/'configs/rebuild_workspace.json').write_text(json.dumps(
        {'data_root':str(data_root),'stages':list(STAGES),'purpose':'Fresh full-data reproduction; no old completion markers copied'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--stage',default='all',help='Comma-separated stages in dependency order, or all')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();workspace=a.workspace.expanduser().resolve();data=a.data_root.expanduser().resolve()
    if workspace==ROOT or ROOT in workspace.parents:
        raise ValueError('Choose a separate workspace outside the publication checkout')
    stages=list(STAGES) if a.stage=='all' else a.stage.split(',')
    if any(s not in STAGES for s in stages):p.error('Unknown stage')
    if a.execute:
        for kind in ['binance_trades','binance_booktickers','binance_liquidations','bybit_liquidations']:
            for sym in ['btcusdt','ethusdt']:
                file=data/kind/f'{"" if kind.startswith("bybit") else "perp_"}{sym}.parquet'
                if not file.is_file():raise FileNotFoundError(file)
        initialize(workspace,data)
    env=os.environ.copy();env.update(LMF_DATA_ROOT=str(data),PYTHONPATH=os.pathsep.join([str(workspace/'src'),str(workspace)]),
                                     POLARS_MAX_THREADS='4',MPLBACKEND='Agg')
    for stage in stages:
        print(f'[{stage}]',flush=True)
        for kind,value in STAGES[stage]:
            cmd=[sys.executable,'-m',value] if kind=='module' else ([sys.executable,'-c',value] if kind=='code' else [sys.executable,str(workspace/'scripts'/value)])
            print(' '.join(cmd),flush=True)
            if a.execute:subprocess.run(cmd,cwd=workspace,env=env,check=True)
    if not a.execute:print('Plan only; no workspace created and no training performed. Add --execute to run.')


if __name__=='__main__':main()
