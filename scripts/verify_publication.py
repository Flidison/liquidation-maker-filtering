"""Deterministic content, notebook, link and secret checks before publication."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BAD=re.compile(rb'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|/Users/|/home/[^/\s]+/|[A-Za-z0-9._%+-]+@(?:gmail|yandex|mail|outlook)\.[A-Za-z]{2,}')
FORBIDDEN={'.parquet','.joblib','.cbm','.pkl','.zip','.pdf','.html','.pyc','.pem','.key'}


def main():
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','.venv','__pycache__','.pytest_cache','liquidation_maker_filtering.egg-info'])]
    for p in files:
        assert p.suffix not in FORBIDDEN, f'Unexpected artifact: {p.relative_to(ROOT)}'
        assert p.stat().st_size<10_000_000, f'Large file: {p.relative_to(ROOT)}'
        # Exclude this scanner's own literal signatures; actual credentials still cannot be present.
        if p.name!='verify_publication.py':assert not BAD.search(p.read_bytes()),f'Secret/private path: {p.relative_to(ROOT)}'
        if p.suffix=='.py':ast.parse(p.read_text())
    notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==6
    for p in notebooks:
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        assert any('image/png' in o.get('data',{}) for c in n.cells if c.cell_type=='code' for o in c.outputs), f'No rendered figures: {p.name}'
        for c in n.cells:
            if c.cell_type=='code':
                assert c.execution_count is not None,f'Unexecuted cell: {p.name}'
                assert not any(o.output_type=='error' for o in c.outputs),f'Failed output: {p.name}'
    for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md'),ROOT/'data/README.md',ROOT/'artifacts/README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https://','http://','#')):continue
            path=target.split('#')[0]
            assert (p.parent/path).exists(),f'Broken local link: {p.name} -> {target}'
    manifest=json.loads((ROOT/'artifacts/source_manifest.json').read_text())
    for row in manifest:
        if 'published_sha256' not in row:continue
        p=ROOT/row['published']
        assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==row['published_sha256'],row['published']
    # Audit every committed blob, including earlier commits. No old development history is imported.
    if (ROOT/'.git').exists():
        objects=subprocess.check_output(['git','rev-list','--objects','--all'],cwd=ROOT,text=True)
        for line in objects.splitlines():
            sha,_,name=line.partition(' ')
            if not name:continue
            kind=subprocess.check_output(['git','cat-file','-t',sha],cwd=ROOT,text=True).strip()
            if kind!='blob':continue
            content=subprocess.check_output(['git','cat-file','blob',sha],cwd=ROOT)
            assert len(content)<10_000_000, f'Large historical blob: {name}'
            assert Path(name).suffix not in FORBIDDEN, f'Forbidden historical blob: {name}'
            if name!='scripts/verify_publication.py':assert not BAD.search(content),f'Secret/private path in history: {name}'
    print(f'Publication checks passed: {len(notebooks)} executed notebooks; paths, manifest, content and Git blobs checked.')


if __name__=='__main__':main()
