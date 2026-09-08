"""Start a detached, bounded Stage2 phase; never resume an unrelated experiment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage2_qwen15b_20260905"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["bootstrap", "preflight", "preflight-v2", "attacks", "search", "evaluate"])
    p.add_argument("--assets")
    args = p.parse_args()
    out = ROOT / "results/stage2_qwen15b_wording_v2_20260905" if args.phase == "preflight-v2" else OUT
    out.mkdir(parents=True, exist_ok=True)
    state = out / (args.phase + "_PROCESS.json")
    if state.exists():
        previous = json.loads(state.read_text())
        cmdline = Path(f"/proc/{previous['pid']}/cmdline")
        if cmdline.exists() and str(ROOT).encode() in cmdline.read_bytes():
            raise RuntimeError("Phase still running")
    env = dict(os.environ, HF_HOME="/root/autodl-tmp/huggingface", HF_ENDPOINT="https://hf-mirror.com",
        HF_HUB_DISABLE_XET="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        PYTHONPATH=str(ROOT/"src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.phase == "bootstrap":
        script = "stage2_bootstrap.py"
    elif args.phase == "preflight":
        script = "stage2_preflight.py"
    elif args.phase == "preflight-v2":
        if not args.assets:
            raise ValueError("Existing model asset manifest required")
        script = "stage2_preflight_v2.py"
    else:
        script = "stage2_pipeline.py"
    command = [sys.executable, "-u", str(ROOT/"scripts"/script)]
    if args.phase in {"attacks", "search", "evaluate"}:
        command += [args.phase]
    if args.phase == "preflight-v2":
        command += ["--assets", args.assets]
    if not Path(command[2]).exists():
        raise FileNotFoundError(command[2])
    log = out / (args.phase + ".log")
    with log.open("ab") as handle:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    payload = {"pid":proc.pid, "phase":args.phase, "command":command, "log":str(log)}
    state.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
