"""Continue evaluation after the already-running search; archive results."""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();status=root/'SUPERVISOR.json';scripts=Path(__file__).resolve().parent
    import fcntl
    with (root/'SUPERVISOR.lock').open('w') as supervisor_lock:
        fcntl.flock(supervisor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(status,dict(phase='WAITING_FOR_SEARCH',updated=time.time()))
        with (root/'RUN.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (root/'CANDIDATES.json').exists():
                write(status,dict(phase='FAILED',reason='Search exited without frozen candidates',updated=time.time()));return
        write(status,dict(phase='EVALUATING',updated=time.time()))
        with (root/'evaluation.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(scripts/'evaluate_prompt_budget.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(status,dict(phase='FAILED',reason='Evaluation returned nonzero',returncode=result.returncode,updated=time.time()));return
        archive=root.parent/(root.name+'-results.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(root,arcname=root.name)
            for name in ['run_prompt_budget.py','evaluate_prompt_budget.py','supervise_prompt_budget.py']:
                tar.add(scripts/name,arcname='code/'+name)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        write(status,dict(phase='COMPLETE_AWAITING_LOCAL_VERIFICATION',archive=str(archive),sha256=digest,bytes=archive.stat().st_size,updated=time.time()))


if __name__=='__main__':main()
