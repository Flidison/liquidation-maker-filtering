"""Execute every publication notebook without raw data or model retraining."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true',help='execute in memory; do not rewrite notebooks')
    args=p.parse_args()
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        # Explicit interpreter avoids accidentally using a different global kernel.
        import sys
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        try:
            NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
            km.cleanup_resources()
        if not args.check:nbformat.write(nb,path)
        print(f'Executed {path.name}',flush=True)


if __name__=='__main__':main()
