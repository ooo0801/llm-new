"""One-shot, bounded continuation of the already-running six-source pilot.

No new searches, no retries on a failed phase, no additional samples, no shutdown.
Survives SSH disconnection and stops on a failed technical/statistical prerequisite.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import project_path
from stage2_pilot import OUT, get_design, read
from llm_integrity.stage1_r1 import digest
from llm_integrity.stage1_r2 import atomic_json, freeze_json


def validate_search_results(p):
    files=[OUT/'search'/method/(source['id']+'.json') for method in p['methods'] for source in p['prompts']]
    for path in files:
        row=read(path)
        if row.get('failure'): raise RuntimeError(f'Technical failure in {path.name}')
        if row['design_sha256']!=digest(OUT/'DESIGN.json'): raise RuntimeError('Search hash mismatch')
        if row['rounds_completed']!=row['rounds_requested']: raise RuntimeError('Search rounds incomplete')
    return files


def status(phase,**fields):
    value={'phase':phase,'unix_time':time.time(),**fields}
    atomic_json(OUT/'CONTINUATION_STATUS.json',value); print(json.dumps(value,ensure_ascii=False),flush=True)


def run_phase(phase):
    status('RUNNING_EVALUATION_PHASE',evaluation_phase=phase)
    cmd=[sys.executable,'-u',str(project_path('scripts/stage2_pilot_evaluate.py')),phase]
    code=subprocess.call(cmd,cwd=project_path('.'))
    if code: raise RuntimeError(f'Evaluation {phase} failed with exit code {code}; no automatic retry or expansion')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    if args.launch:
        path=OUT/'CONTINUATION_PROCESS.json'
        if path.exists():
            previous=read(path); proc=Path(f"/proc/{previous['pid']}/cmdline")
            if proc.exists() and str(Path(__file__).resolve()).encode() in proc.read_bytes(): raise RuntimeError('Continuation already running')
        env=dict(os.environ,HF_HOME='/root/autodl-tmp/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 PYTHONPATH=str(project_path('src')),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve())]
        with (OUT/'continuation.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=project_path('.'),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        payload={'pid':child.pid,'command':cmd}; atomic_json(path,payload); print(json.dumps(payload)); return
    import fcntl
    with (OUT/'CONTINUATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=get_design()
            files=[Path(__file__),project_path('scripts/stage2_pilot_evaluate.py'),project_path('scripts/stage2_eval_contract.py'),
                   project_path('scripts/run_stage1_r2.py'),project_path('scripts/build_discrete_candidate_portfolio.py')]
            binding={'design_sha256':digest(OUT/'DESIGN.json'),'files':{str(f.resolve()):digest(f) for f in files},
                     'behavior_ceiling':p['maximum_behavior_response_rows'],'utility_ceiling':p['maximum_utility_generations'],
                     'single_run':True,'retry':False,'budget_expansion':False,'shutdown':False}
            freeze_json(OUT/'CONTINUATION_PLAN.json',binding)
            started=time.monotonic(); previous=None
            while True:
                state=read(OUT/'STATUS.json'); phase=state['phase']
                if phase=='FAILED': raise RuntimeError('Search failed; preserve evidence and request audit')
                if phase=='SEARCH_COMPLETE': break
                procinfo=read(OUT/'search_PROCESS.json'); proc=Path(f"/proc/{procinfo['pid']}/cmdline")
                if not proc.exists() or b'stage2_pilot.py' not in proc.read_bytes(): raise RuntimeError('Search process absent before completion')
                marker=(phase,state.get('method'),state.get('source'))
                if marker!=previous:
                    status('WAITING_FOR_EXISTING_SEARCH',search_state=state); previous=marker
                if time.monotonic()-started>12*3600: raise RuntimeError('One-shot continuation wait limit reached; no search restarted')
                time.sleep(30)
            # Search publishes SEARCH_COMPLETE just before its lock/process is released.
            with (OUT/'RUN.lock').open('a') as search_lock:
                fcntl.flock(search_lock,fcntl.LOCK_EX)
                validate_search_results(p)
            for name,expected in binding['files'].items():
                if digest(name)!=expected: raise RuntimeError('Continuation code changed while waiting')
            run_phase('freeze')
            plan=read(OUT/'evaluation/PLAN.json')
            if plan['behavior_response_count']>binding['behavior_ceiling'] or plan['utility_response_count']>binding['utility_ceiling']:
                raise RuntimeError('Actual evaluation budget exceeds authorization')
            status('ACTUAL_BUDGET_FROZEN_BEFORE_SAMPLING',unique_prompts=len(plan['candidates']['prompts']),
                   behavior_responses=plan['behavior_response_count'],utility_responses=plan['utility_response_count'])
            for phase in ('confirmation','heldout','report'): run_phase(phase)
            status('LIMITED_PILOT_COMPLETE',full_stage2_complete=False,summary=str(OUT/'evaluation/SUMMARY.json'))
        except Exception as exc:
            status('STOPPED_REQUIRES_AUDIT',error_type=type(exc).__name__,error=str(exc)); raise


if __name__=='__main__': main()
