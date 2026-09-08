"""Acceptance evidence for a completed local R1 run; no model dependencies."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_integrity.stage1_r1 import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")]),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    targets = ["tests/test_stage1_r1.py", "tests/test_features.py", "tests/test_h8_precalibration.py",
               "tests/test_h9_stage1_attack_stat_calibration.py"]
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "-o", "addopts=", "--junitxml=" + str(out / "pytest.xml")]
    test = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    (out / "pytest.log").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    paths = [out / "R1_REPORT.json", out / "prompt_endpoint_results.json", out / "legacy_comparison.json"]
    before = {p.name: digest(p) for p in paths}
    command = [sys.executable, str(ROOT / "scripts/run_stage1_r1.py"), "--input", str(args.input.resolve()),
               "--output", str(out), "--permutations", "999"]
    resume = subprocess.run(command, env=env, text=True, capture_output=True)
    (out / "resume.log").write_text(resume.stdout + resume.stderr, encoding="utf-8")
    if resume.returncode:
        raise RuntimeError(resume.stdout + resume.stderr)
    after = {p.name: digest(p) for p in paths}
    if before != after:
        raise RuntimeError("Resume changed result bytes")
    rows = json.loads((out / "prompt_endpoint_results.json").read_text(encoding="utf-8"))
    case = [r for r in rows if r["prompt_id"] == "instruction_h6_b147b35b8b1f" and r["family"] == "lora" and r["strength"] == "medium"]
    if len(rows) != 324 or len(case) != 3 or not all(r[ch]["detected"] is True for r in case for ch in ("raw", "h8")):
        raise RuntimeError("Missing units or deterministic-prompt regression")
    import numpy
    import pytest
    import xml.etree.ElementTree as ET
    suites = ET.parse(out / "pytest.xml").getroot().iter("testsuite")
    counts = [s.attrib for s in suites]
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    unchanged = all(digest(item["file"]) == item["sha256"] for item in manifest["inputs"])
    if not unchanged:
        raise RuntimeError("Historical input changed")
    save_json(out / "VERIFICATION.json", {
        "status": "PASS", "test_count": sum(int(c["tests"]) for c in counts),
        "failures": sum(int(c["failures"]) + int(c["errors"]) for c in counts),
        "test_files_sha256": {t: digest(ROOT / t) for t in targets},
        "test_log_sha256": digest(out / "pytest.log"), "junit_sha256": digest(out / "pytest.xml"),
        "resume_exit_code": resume.returncode, "resume_result_byte_identity": before == after,
        "result_sha256": after, "historical_inputs_unchanged": unchanged,
        "known_instruction_medium_case": [{"seed": r["attack_seed"], "raw_p": r["raw"]["p_value"],
              "h8_p": r["h8"]["p_value"], "task_pass_drop": r["task_pass_drop"]} for r in case],
        "new_model_responses": 0, "server_access_performed": False,
        "local_execution_environment": {"python": sys.version, "numpy": numpy.__version__,
                                         "pytest": pytest.__version__, "platform": platform.platform()},
        "verification_script_sha256": digest(Path(__file__)),
    })
    print((out / "VERIFICATION.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
