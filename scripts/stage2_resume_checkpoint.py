"""Validate the shutdown checkpoint before resuming existing frozen search and continuation."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from _bootstrap import project_path
from stage2_pilot import OUT,get_design,read
from llm_integrity.stage1_r1 import digest


def verify():
    p=get_design(); saved=read(OUT/'PAUSED_CHECKPOINT.json')
    if saved['design_sha256']!=digest(OUT/'DESIGN.json'): raise RuntimeError('Frozen design differs')
    mutable_status={'STATUS.json','search_PROCESS.json','CONTINUATION_PROCESS.json','CONTINUATION_STATUS.json'}
    for name,sha in saved['files_sha256'].items():
        path=project_path(name)
        if path.name in mutable_status: continue
        if not path.exists() or digest(path)!=sha: raise RuntimeError(f'Checkpoint file differs: {name}')
    # Every persisted result, including any newer one, must be complete before the original skip-existing resume.
    complete=[]; pending=[]
    for method in p['methods']:
        for source in p['prompts']:
            path=OUT/'search'/method/(source['id']+'.json')
            if not path.exists(): pending.append((method,source['id'])); continue
            row=read(path)
            if row.get('failure') or row['rounds_completed']!=row['rounds_requested'] or row['design_sha256']!=saved['design_sha256']:
                raise RuntimeError('Incomplete/failed result cannot be skipped')
            complete.append((method,source['id']))
    for f,meta in p['assets']['files'].items():
        path=Path(p['assets']['path'])/f
        if not path.exists() or digest(path)!=meta['sha256']: raise RuntimeError(f'Base model missing or changed: {f}')
    # Semantic encoder snapshot presence is checked without downloading or changing environments.
    from huggingface_hub import snapshot_download
    sem=Path(snapshot_download('BAAI/bge-small-zh-v1.5',revision='7999e1d3359715c523056ef9478215996d62a620',local_files_only=True))
    if digest(sem/'model.safetensors')!='354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026':
        raise RuntimeError('Semantic encoder changed')
    print({'verification':'PASS','completed':complete,'pending':pending},flush=True)
    return pending


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    os.environ['HF_HOME']='/root/autodl-tmp/huggingface'
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    pending=verify()
    if args.launch:
        if pending: subprocess.run([sys.executable,str(project_path('scripts/stage2_pilot_launch.py')),'search'],check=True)
        subprocess.run([sys.executable,str(project_path('scripts/stage2_finish_pilot.py')),'--launch'],check=True)
