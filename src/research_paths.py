"""Portable paths; relocation does not change any research definition."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('LMF_DATA_ROOT', ROOT / 'data/raw')).expanduser().resolve()
LEGACY_GROUPS = {
    'data_quality_artifacts': 'quality',
    'liquidation_burst_artifacts': 'bursts',
    'liquidation_followthrough_artifacts': 'followthrough',
    'liquidation_early_warning_artifacts': 'early_warning',
    'liquidation_validation_artifacts': 'statistics',
    'evt_log_cond_artifacts': 'evt',
    'maker_hybrid_artifacts': 'maker',
    'maker_hybrid_e9_artifacts': 'catboost',
}


def artifact_dir(group):
    return ROOT / 'artifacts/results' / group


def resolve_legacy(value):
    """Resolve paths in historical receipts, without trusting absolute local paths."""
    p = Path(value)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts[:2] == ('liquidation_task', 'data') or parts[:2] == ('data', 'raw'):
        return DATA_ROOT.joinpath(*parts[2:])
    if parts and parts[0] in LEGACY_GROUPS:
        return artifact_dir(LEGACY_GROUPS[parts[0]]).joinpath(*parts[1:])
    if len(parts) == 1 and p.suffix == '.py':
        return ROOT / ('tests' if p.name.startswith('test_') else 'src') / p
    return ROOT / p
