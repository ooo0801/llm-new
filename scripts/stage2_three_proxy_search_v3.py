"""Stage2 three-proxy search v3: repair singleton-batch proxy interface.

The v2 preflight and proxy calibration are reused byte-for-byte.  The failed
v2 pilot search is retained as provenance; v3 starts with an empty search
directory and a newly frozen plan that includes the repaired optimizer hash.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import stage2_three_proxy_search as base
import stage2_three_proxy_search_v2 as v2
from _bootstrap import project_path
from llm_integrity.stage1_r1 import digest, save_json


V2 = project_path("results/stage2_three_proxy_ceiling_v2_20260906")
OUT = project_path("results/stage2_three_proxy_ceiling_v3_20260906")
base.OUT = OUT
v2.OUT = OUT


REUSED = (
    "preflight_responses.jsonl",
    "PREFLIGHT_REPORT.json",
    "proxy_calibration_records.json",
    "PROXY_CALIBRATION.json",
)


def source_files() -> list[Path]:
    return [
        Path(__file__),
        project_path("scripts/stage2_three_proxy_search.py"),
        project_path("scripts/stage2_three_proxy_search_v2.py"),
        project_path("src/llm_integrity/macro_proxy.py"),
        project_path("src/llm_integrity/joint_inner_optimizer.py"),
        project_path("src/llm_integrity/discrete_joint_inner_optimizer.py"),
        project_path("src/llm_integrity/inner_micro_proxy.py"),
        project_path("src/llm_integrity/inner_variant_sampler.py"),
    ]


def freeze_plan() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = base.read(V2 / "PLAN.json")
    payload = dict(old)
    payload.update(
        {
            "schema": "stage2-three-proxy-ceiling-v3",
            "repair": (
                "Convert singleton-batch next-token logits from [1,V] to [V] "
                "at the optimizer boundary before proxy evaluation."
            ),
            "v2_plan_sha256": digest(V2 / "PLAN.json"),
            "v2_failure_status_sha256": digest(V2 / "STATUS.json"),
            "v2_supervisor_sha256": digest(V2 / "SUPERVISOR.json"),
            "reused_artifact_hashes": {
                name: digest(V2 / name) for name in REUSED
            },
            "source_hashes": {
                str(path.relative_to(project_path("."))).replace("\\", "/"): digest(path)
                for path in source_files()
            },
        }
    )
    base.freeze(OUT / "PLAN.json", payload)
    base.status("V3_PLAN_FROZEN", repair="singleton_batch_proxy_interface")


def context() -> dict:
    plan = base.read(OUT / "PLAN.json")
    if digest(V2 / "PLAN.json") != plan["v2_plan_sha256"]:
        raise RuntimeError("v2 plan changed")
    for name, expected in plan["reused_artifact_hashes"].items():
        if digest(V2 / name) != expected:
            raise RuntimeError(f"v2 reused artifact changed: {name}")
    for name, expected in plan["source_hashes"].items():
        if digest(project_path(name)) != expected:
            raise RuntimeError(f"v3 source changed after freeze: {name}")
    return plan


def reuse_v2() -> None:
    plan = context()
    copied = {}
    for name in REUSED:
        source = V2 / name
        target = OUT / name
        if target.exists() and digest(target) != digest(source):
            raise RuntimeError(f"Refusing to overwrite divergent artifact: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        copied[name] = digest(target)
    save_json(
        OUT / "REUSE_MANIFEST.json",
        {
            "schema": "stage2-v3-reuse-v1",
            "source": str(V2.relative_to(project_path("."))),
            "artifacts": copied,
            "v3_plan_sha256": digest(OUT / "PLAN.json"),
        },
    )
    base.status("V2_PREFLIGHT_AND_CALIBRATION_REUSED", artifacts=len(copied))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["freeze", "reuse", "pilot-search", "formal-search"]
    )
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze_plan()
        return
    plan = context()
    if args.phase == "reuse":
        reuse_v2()
    elif args.phase == "pilot-search":
        base.search(plan, False)
    else:
        base.search(plan, True)


if __name__ == "__main__":
    main()
