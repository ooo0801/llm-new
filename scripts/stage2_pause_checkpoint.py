"""Pause at the currently running source-result boundary; archive portable research state."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from _bootstrap import project_path
from stage2_pilot import OUT, get_design, read
from llm_integrity.stage1_r1 import digest
from llm_integrity.stage1_r2 import atomic_json


def matches(pid, filename):
    path=Path(f'/proc/{pid}/cmdline')
    return path.exists() and str(project_path('scripts/'+filename)).encode() in path.read_bytes()


def pause():
    p=get_design(); state=read(OUT/'STATUS.json')
    if state['phase']!='search': raise RuntimeError('Unexpected phase; inspect manually before pause')
    method=state['method']; source=p['prompts'][state['source']-1]
    target=OUT/'search'/method/(source['id']+'.json')
    info=read(OUT/'search_PROCESS.json'); pid=info['pid']
    if not matches(pid,'stage2_pilot.py'): raise RuntimeError('Search identity mismatch')
    supervisor=read(OUT/'CONTINUATION_PROCESS.json')['pid']
    if matches(supervisor,'stage2_finish_pilot.py'): raise RuntimeError('Stop the continuation before boundary pause')
    atomic_json(OUT/'PAUSE_REQUEST.json',{'method':method,'source':source['id'],'pid':pid,'target':str(target),
                                        'requested_unix':time.time(),'policy':'Stop after this complete source, not a round-level resume'})
    start=time.monotonic()
    while True:
        if target.exists():
            try: result=read(target)
            except json.JSONDecodeError: time.sleep(.2); continue
            if result.get('failure') or result['rounds_completed']!=result['rounds_requested']:
                raise RuntimeError('Boundary result incomplete or failed; preserve for audit')
            if result['design_sha256']!=digest(OUT/'DESIGN.json'): raise RuntimeError('Boundary design mismatch')
            if matches(pid,'stage2_pilot.py'):
                os.kill(pid,signal.SIGSTOP)
                os.kill(pid,signal.SIGTERM)
                os.kill(pid,signal.SIGCONT)
            for _ in range(100):
                if not matches(pid,'stage2_pilot.py'): break
                time.sleep(.1)
            if matches(pid,'stage2_pilot.py'): raise RuntimeError('Search did not stop; do not shutdown yet')
            break
        if not matches(pid,'stage2_pilot.py'): raise RuntimeError('Search ended without requested result')
        if time.monotonic()-start>1200: raise RuntimeError('Boundary wait exceeded 20min; no automatic force kill')
        time.sleep(.5)
    os.sync()
    checkpoint=[]; pending=[]
    for method in p['methods']:
        for row in p['prompts']:
            path=OUT/'search'/method/(row['id']+'.json')
            if path.exists():
                result=read(path)
                if result.get('failure') or result['rounds_completed']!=result['rounds_requested']:
                    raise RuntimeError('Invalid saved result; refuse resume certificate')
                checkpoint.append({'method':method,'source_id':row['id'],'sha256':digest(path)})
            else: pending.append({'method':method,'source_id':row['id']})
    files={}
    for folder in ('src','scripts','configs'):
        for file in sorted(project_path(folder).rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts: files[str(file.relative_to(project_path('.')))]=digest(file)
    for file in sorted(OUT.rglob('*')):
        if file.is_file() and file.suffix in ('.json','.jsonl','.safetensors','.npy') and file.name not in ('PAUSED_CHECKPOINT.json','BACKUP_RECEIPT.json'):
            files[str(file.relative_to(project_path('.')))]=digest(file)
    checkpoint_manifest={'schema':'stage2-source-boundary-checkpoint-v1','status':'PAUSED_SAFE_AT_SOURCE_BOUNDARY',
             'paused_unix':time.time(),'design_sha256':digest(OUT/'DESIGN.json'),'completed':checkpoint,'pending':pending,
             'search_process_stopped':True,'continuation_process_stopped':not matches(supervisor,'stage2_finish_pilot.py'),
             'checkpoint_granularity':'complete source prompt; no in-memory round state promised','files_sha256':files,
             'model_assets':p['assets'],'model_weights_in_backup':False,
             'resume':'python scripts/stage2_resume_checkpoint.py --launch',
             'preserve_server_data_disk':True}
    atomic_json(OUT/'PAUSED_CHECKPOINT.json',checkpoint_manifest)
    # Whole isolated source/results folder, excluding only reproducible bytecode. No base model weights live here.
    archive=Path('/root/autodl-tmp/stage2_pause_backup_20260905.tar.gz')
    if archive.exists(): raise RuntimeError('Backup exists; refuse overwrite')
    root=project_path('.')
    def exclude(info):
        return None if '__pycache__' in Path(info.name).parts or '.pytest_cache' in Path(info.name).parts else info
    with tarfile.open(archive,'w:gz') as tar: tar.add(root,arcname=root.name,filter=exclude)
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),
             'checkpoint_sha256':digest(OUT/'PAUSED_CHECKPOINT.json'),'complete_sources':len(checkpoint),'pending_sources':len(pending)}
    atomic_json(OUT/'BACKUP_RECEIPT.json',receipt); os.sync(); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    if args.launch:
        env=dict(os.environ,PYTHONPATH=str(project_path('src')))
        with (OUT/'pause.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=project_path('.'),env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pause_pid':child.pid}))
    else: pause()
