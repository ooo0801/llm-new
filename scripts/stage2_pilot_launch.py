import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/stage2_six_source_pilot_20260905"
parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=["prepare","search","evaluate"])
args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
state=OUT/(args.phase+"_PROCESS.json")
if state.exists():
    old=json.loads(state.read_text()); process=Path(f"/proc/{old['pid']}/cmdline")
    if process.exists() and str(ROOT).encode() in process.read_bytes(): raise RuntimeError("Already running")
env=dict(os.environ,HF_HOME="/root/autodl-tmp/huggingface",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
    PYTHONPATH=str(ROOT/"src"),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
cmd=[sys.executable,"-u",str(ROOT/"scripts/stage2_pilot.py"),args.phase]
with (OUT/(args.phase+".log")).open("ab") as f:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
payload={"phase":args.phase,"pid":proc.pid,"command":cmd}
state.write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload))
