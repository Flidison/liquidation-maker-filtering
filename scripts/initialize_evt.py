"""Rebuild the original eight-source catalog without an in-memory HFT load."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_paths import DATA_ROOT, artifact_dir


def main():
    quality=artifact_dir('quality')
    manifest=pd.read_csv(quality/'source_manifest.csv')
    summary=pd.read_csv(quality/'summary.csv')
    key=lambda p:'/'.join(Path(p).parts[-2:])
    old={key(r.source_file):r for r in manifest.itertuples()}
    audit={key(r['file']):r for r in summary.to_dict('records')}
    rows=[]
    for symbol in ['btcusdt','ethusdt']:
        for kind,label in [('binance_trades','trades'),('binance_booktickers','bbo'),
                           ('binance_liquidations','liq_binance'),('bybit_liquidations','liq_bybit')]:
            path=DATA_ROOT/kind/f'{"" if kind.startswith("bybit") else "perp_"}{symbol}.parquet'
            recorded=old[key(path)];a=audit[key(path)];f=pq.ParquetFile(path)
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            assert digest==recorded.sha256
            assert f.metadata.num_rows==recorded.rows==a['rows']
            assert a['number of removed observations']==0
            i=f.schema_arrow.get_field_index('timestamp')
            stats=[f.metadata.row_group(j).column(i).statistics for j in range(f.metadata.num_row_groups)]
            assert all(x is not None and x.has_min_max and x.null_count==0 for x in stats)
            delay=200_000 if label=='liq_bybit' else 0
            rows.append({'symbol':symbol,'dataset':label,'source':f'data/raw/{kind}/{path.name}',
                'rows':f.metadata.num_rows,'bytes':path.stat().st_size,'sha256':digest,
                'min_effective_timestamp':min(s.min for s in stats)+delay,
                'max_effective_timestamp':max(s.max for s in stats)+delay,'removed_rows':0,
                'exact_duplicate_extras':a['exact duplicate rows'],'timestamp_reversals':a['timestamp reversals'],
                'mode':'lazy' if label in ['trades','bbo'] else 'in memory'})
    catalog=pd.DataFrame(rows)
    spec=json.loads((artifact_dir('evt')/'split_specification.json').read_text())
    assert spec['total_source_rows']==catalog.rows.sum()
    assert spec['split_bounds_us']['train'][0]<=catalog.min_effective_timestamp.min()
    assert spec['split_bounds_us']['test'][1]>catalog.max_effective_timestamp.max()
    catalog.to_csv(artifact_dir('evt')/'source_catalog.csv',index=False)
    print('Verified and initialized EVT source catalog; split specification unchanged.')


if __name__=='__main__':main()
