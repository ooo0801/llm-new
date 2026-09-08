"""Explicit bounded evaluation phase launcher; never starts another search or expands sampling."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage2_six_source_pilot_20260905/evaluation'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','confirmation','heldout','report'])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    state=OUT/(args.phase+'_PROCESS.json')
    if state.exists():
        old=json.loads(state.read_text()); proc=Path(f"/proc/{old['pid']}/cmdline")
        if proc.exists() and str(ROOT).encode() in proc.read_bytes(): raise RuntimeError('Already running')
    env=dict(os.environ,HF_HOME='/root/autodl-tmp/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    cmd=[sys.executable,'-u',str(ROOT/'scripts/stage2_pilot_evaluate.py'),args.phase]
    with (OUT/(args.phase+'.log')).open('ab') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    payload={'phase':args.phase,'pid':proc.pid,'command':cmd}
    state.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload))


if __name__=='__main__': main()
