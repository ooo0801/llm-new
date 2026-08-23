#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from _bootstrap import ROOT
from llm_integrity.h8_f1b import file_sha256
from llm_integrity.h8_f1b_recovery import (
    RECOVERY_SCHEMA_VERSION,
    load_recovery_authorization,
    normalize_identity_payload,
    read_attempt_events,
    sha256_prefix_lines,
    validate_attempt_alignment,
)

import run_h8_f1b_formal_sampling as frozen


DEFAULT_AUTH = ROOT / "configs/h8_f1b_tuple_list_recovery.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured recovery path escapes repository")
    return path


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Recovery {label} SHA256 mismatch")


def verify_recovery_git_and_files(auth: Mapping[str, Any], auth_path: Path) -> str:
    implementation_commit = str(auth["git"]["recovery_implementation_commit"])
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    paths = (
        auth_path,
        resolve(auth["implementation"]["runner"]),
        resolve(auth["implementation"]["module"]),
        resolve(auth["implementation"]["protocol"]),
        resolve(auth["frozen_inputs"]["original_authorization"]),
    )
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    require_hash(
        resolve(auth["implementation"]["runner"]),
        auth["implementation"]["runner_sha256"],
        "runner",
    )
    require_hash(
        resolve(auth["implementation"]["module"]),
        auth["implementation"]["module_sha256"],
        "module",
    )
    require_hash(
        resolve(auth["implementation"]["protocol"]),
        auth["implementation"]["protocol_sha256"],
        "protocol",
    )
    require_hash(
        resolve(auth["frozen_inputs"]["original_authorization"]),
        auth["frozen_inputs"]["original_authorization_sha256"],
        "original F1-B authorization",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree must be clean during F1-B recovery")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_context(auth_path: Path) -> dict[str, Any]:
    auth = load_recovery_authorization(auth_path)
    execution_commit = verify_recovery_git_and_files(auth, auth_path)
    snapshot_path = resolve(auth["frozen_inputs"]["pause_snapshot"])
    require_hash(
        snapshot_path,
        auth["frozen_inputs"]["pause_snapshot_sha256"],
        "pause snapshot",
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (
        snapshot.get("status") != "PASS_USER_REQUESTED_PAUSE_READY_TO_RESUME"
        or int(snapshot.get("validated_success_response_count", -1)) != 12583
        or int(snapshot.get("validated_success_attempt_event_count", -1)) != 12583
        or int(snapshot.get("remaining_response_count", -1)) != 137
    ):
        raise ValueError("Recovery pause snapshot gate failed")

    original_auth_path = resolve(auth["frozen_inputs"]["original_authorization"])
    frozen_context = frozen.load_context(original_auth_path)
    frozen.verify_git_state(frozen_context["auth"], original_auth_path)
    if frozen_context["frozen_identity_sha256"] != auth["frozen_inputs"]["frozen_identity_sha256"]:
        raise ValueError("Recovery frozen identity changed")
    if frozen_context["frozen_identity_sha256"] != snapshot["frozen_identity_sha256"]:
        raise ValueError("Recovery snapshot/frozen identity mismatch")
    runtime = frozen.verify_runtime(frozen_context["runtime_auth"])
    model_snapshot = frozen.verify_snapshot(
        frozen_context["runtime_auth"], frozen_context["h8_config"]
    )
    if (
        model_snapshot["snapshot_listing_sha256"]
        != auth["frozen_inputs"]["model_snapshot_listing_sha256"]
    ):
        raise ValueError("Recovery model snapshot listing changed")
    recovery_files = [
        auth_path,
        resolve(auth["implementation"]["runner"]),
        resolve(auth["implementation"]["module"]),
        resolve(auth["implementation"]["protocol"]),
        snapshot_path,
    ]
    return {
        "auth": auth,
        "auth_path": auth_path,
        "execution_commit": execution_commit,
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "original_auth_path": original_auth_path,
        "frozen": frozen_context,
        "runtime": runtime,
        "model_snapshot": model_snapshot,
        "recovery_file_seal": frozen.build_file_metadata_seal(recovery_files),
    }


def _expected_sha(path: Path, expected: str | None, label: str) -> None:
    if expected is None:
        if path.exists():
            raise ValueError(f"Unexpected pre-recovery artifact: {path}")
        return
    require_hash(path, expected, label)


def audit_prefix(context: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = context["snapshot"]
    frozen_context = context["frozen"]
    partitions = frozen.partition_requests(frozen_context["generation"]["requests"])
    snapshot_rows = {row["partition_id"]: row for row in snapshot["partition_audit"]}
    allowed = set(context["auth"]["recovery_bounds"]["allowed_partitions"])
    all_records: list[dict[str, Any]] = []
    total_attempt_events = 0
    total_technical_failures = 0
    partition_counts: dict[str, int] = {}

    for partition_id, requests in partitions.items():
        expected = snapshot_rows[partition_id]
        record_path = frozen.partition_path(frozen_context["auth"], partition_id, "records")
        attempt_path = frozen.partition_path(frozen_context["auth"], partition_id, "attempts")
        materialization_path = frozen.partition_path(
            frozen_context["auth"], partition_id, "materialization"
        )
        provenance_path = frozen.partition_path(
            frozen_context["auth"], partition_id, "provenance"
        )

        if partition_id not in allowed:
            _expected_sha(record_path, expected["records_sha256"], "frozen record partition")
            _expected_sha(attempt_path, expected["attempts_sha256"], "frozen attempt partition")
            _expected_sha(
                materialization_path,
                expected["materialization_sha256"],
                "frozen materialization partition",
            )
            _expected_sha(
                provenance_path,
                expected["provenance_sha256"],
                "frozen provenance partition",
            )
        else:
            base_records = int(expected["records"])
            base_attempts = int(expected["attempt_events"])
            if sha256_prefix_lines(record_path, base_records) != expected["records_sha256"]:
                raise ValueError(f"Recovery immutable record prefix changed: {partition_id}")
            if sha256_prefix_lines(attempt_path, base_attempts) != expected["attempts_sha256"]:
                raise ValueError(f"Recovery immutable attempt prefix changed: {partition_id}")
            if expected["materialization_sha256"] is not None:
                _expected_sha(
                    materialization_path,
                    expected["materialization_sha256"],
                    "recovery materialization prefix",
                )
                _expected_sha(
                    provenance_path,
                    expected["provenance_sha256"],
                    "recovery provenance prefix",
                )

        identity_sha = None
        if materialization_path.is_file():
            materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
            identity_sha = materialization.get("materialization_identity_sha256")
        records = frozen.load_existing_prefix(
            record_path,
            requests,
            frozen_identity_sha256=frozen_context["frozen_identity_sha256"],
            materialization_identity_sha256=identity_sha,
        )
        attempt_events = read_attempt_events(attempt_path)
        attempt_audit = validate_attempt_alignment(records, attempt_events)
        if partition_id not in allowed and len(records) != int(expected["records"]):
            raise ValueError(f"Non-recovery partition count changed: {partition_id}")
        if partition_id in allowed and len(records) < int(expected["records"]):
            raise ValueError(f"Recovery partition lost sealed prefix records: {partition_id}")
        partition_counts[partition_id] = len(records)
        all_records.extend(records)
        total_attempt_events += attempt_audit["total_attempt_event_count"]
        total_technical_failures += attempt_audit["technical_failure_event_count"]

    ordered = sorted(all_records, key=lambda record: int(record["schedule_position"]))
    if len(ordered) < 12583 or len(ordered) > 12720:
        raise ValueError("Recovery response total is outside the authorized bounds")
    if [int(record["schedule_position"]) for record in ordered] != list(range(len(ordered))):
        raise ValueError("Recovery records are not a contiguous frozen schedule prefix")
    ids = [str(record["response_id"]) for record in ordered]
    seeds = [int(record["generation_seed"]) for record in ordered]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError("Recovery response ID or generation seed is not globally unique")
    for record in ordered[12583:]:
        if record.get("attack_instance_id") not in allowed:
            raise ValueError("Recovery generated a response outside the two authorized partitions")
        if record.get("data_role") != "final_heldout_attack_only":
            raise ValueError("Recovery changed the authorized data role")
    if any(record.get("formal_detector_statistics_computed") is not False for record in ordered):
        raise ValueError("Forbidden detector statistics found during recovery")
    frozen.verify_file_metadata_seal(context["recovery_file_seal"])
    roles = Counter(record["data_role"] for record in ordered)
    next_request = None
    if len(ordered) < 12720:
        next_request = frozen_context["generation"]["requests"][len(ordered)]
        if int(next_request["schedule_position"]) != len(ordered):
            raise ValueError("Next recovery request is not the next frozen schedule position")
    return {
        "status": "PASS",
        "validated_response_count": len(ordered),
        "remaining_response_count": 12720 - len(ordered),
        "attempt_event_count": total_attempt_events,
        "technical_failure_event_count": total_technical_failures,
        "role_counts": dict(roles),
        "partition_counts": partition_counts,
        "next_frozen_request": next_request,
        "formal_detector_statistics_computed": False,
    }


def worker(auth_path: Path, partition_id: str) -> int:
    context = load_context(auth_path)
    allowed = context["auth"]["recovery_bounds"]["allowed_partitions"]
    if partition_id not in allowed:
        raise PermissionError("Recovery worker may only run the two bounded quantization partitions")
    before = audit_prefix(context)
    partition_index = allowed.index(partition_id)
    if any(
        before["partition_counts"][earlier] != 120
        for earlier in allowed[:partition_index]
    ):
        raise PermissionError("Recovery partitions must follow frozen order")
    if before["partition_counts"][partition_id] >= 120:
        return 0

    original_load_context = frozen.load_context
    original_materialization_identity = frozen.materialization_identity
    original_verify_git_state = frozen.verify_git_state

    def sealed_load_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
        loaded = original_load_context(*args, **kwargs)
        loaded["file_metadata_seal"] = {
            **loaded["file_metadata_seal"],
            **context["recovery_file_seal"],
        }
        return loaded

    def normalized_materialization_identity(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        payload, identity_sha = original_materialization_identity(*args, **kwargs)
        return normalize_identity_payload(payload, identity_sha), identity_sha

    existing_provenance_path = frozen.partition_path(
        context["frozen"]["auth"], partition_id, "provenance"
    )
    if existing_provenance_path.is_file():
        expected_execution_commit = json.loads(
            existing_provenance_path.read_text(encoding="utf-8")
        )["execution_commit"]
    else:
        expected_execution_commit = context["execution_commit"]

    def provenance_preserving_verify_git(*args: Any, **kwargs: Any) -> str:
        original_verify_git_state(*args, **kwargs)
        return str(expected_execution_commit)

    frozen.load_context = sealed_load_context
    frozen.materialization_identity = normalized_materialization_identity
    frozen.verify_git_state = provenance_preserving_verify_git
    try:
        return frozen.worker(context["original_auth_path"], partition_id)
    finally:
        frozen.load_context = original_load_context
        frozen.materialization_identity = original_materialization_identity
        frozen.verify_git_state = original_verify_git_state


def parent(auth_path: Path) -> int:
    context = load_context(auth_path)
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("HF_HUB_CACHE") != "/root/autodl-tmp/huggingface"
    ):
        raise PermissionError("Recovery requires the frozen strict-offline cache")
    if frozen.gpu_compute_processes()["status"] != "PASS":
        raise RuntimeError("Recovery requires clean GPUs before launch")
    before = audit_prefix(context)
    partitions = frozen.partition_requests(context["frozen"]["generation"]["requests"])
    allowed = context["auth"]["recovery_bounds"]["allowed_partitions"]
    cleanup_events: list[dict[str, Any]] = []

    for partition_id in partitions:
        if partition_id not in allowed:
            cleanup_events.append(
                {
                    "partition_id": partition_id,
                    "resume_reused_validated_complete_partition": True,
                }
            )
            continue
        current = audit_prefix(context)
        if current["partition_counts"][partition_id] == len(partitions[partition_id]):
            cleanup_events.append(
                {
                    "partition_id": partition_id,
                    "resume_reused_validated_complete_partition": True,
                }
            )
            continue
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_CACHE": "/root/autodl-tmp/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_VISIBLE_DEVICES": context["frozen"]["auth"]["runtime"][
                    "cuda_visible_devices"
                ],
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--auth",
                str(auth_path),
                "--worker-partition",
                partition_id,
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        cleanup = frozen.gpu_compute_processes()
        cleanup_events.append(
            {
                "partition_id": partition_id,
                "stage": "tuple_list_normalized_recovery_generation",
                "exit_code": completed.returncode,
                "gpu_cleanup": cleanup,
            }
        )
        if completed.returncode != 0 or cleanup["status"] != "PASS":
            raise RuntimeError(f"F1-B recovery partition failed or leaked GPU workers: {partition_id}")
        after_partition = audit_prefix(context)
        if after_partition["partition_counts"][partition_id] != len(partitions[partition_id]):
            raise RuntimeError("Recovery worker exited without completing the frozen partition")

    complete = audit_prefix(context)
    if complete["validated_response_count"] != 12720:
        raise RuntimeError("Recovery did not complete exactly 12,720 formal responses")
    recovery_report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "status": "PASS_RECOVERY_WORKERS_COMPLETE_PENDING_ORIGINAL_FINAL_AUDIT",
        "authorization_sha256": file_sha256(auth_path),
        "recovery_execution_commit": context["execution_commit"],
        "original_frozen_identity_sha256": context["frozen"]["frozen_identity_sha256"],
        "pause_snapshot_sha256": context["auth"]["frozen_inputs"]["pause_snapshot_sha256"],
        "normalization_scope": context["auth"]["recovery_bounds"]["normalized_fields"],
        "normalization_semantics": "JSON data-model container normalization only; canonical identity unchanged",
        "before_response_count": before["validated_response_count"],
        "after_response_count": complete["validated_response_count"],
        "new_response_count": complete["validated_response_count"] - before["validated_response_count"],
        "remaining_response_count": 0,
        "role_counts": complete["role_counts"],
        "technical_failure_event_count": complete["technical_failure_event_count"],
        "formal_detector_statistics_computed": False,
        "final_feature_extraction_performed": False,
        "final_mmd_computation_performed": False,
        "final_score_computation_performed": False,
        "final_global_permutation_performed": False,
        "created_at_utc": now(),
    }
    report_path = resolve(context["auth"]["artifacts"]["recovery_report"])
    frozen.atomic_json(report_path, recovery_report)
    final_report = frozen.final_audit(context["frozen"], cleanup_events)
    print(
        json.dumps(
            {
                "status": "PASS",
                "formal_responses": 12720,
                "recovered_responses": recovery_report["new_response_count"],
                "formal_detector_statistics_computed": False,
                "recovery_report": str(report_path),
                "f1b_final_report": str(final_report),
            },
            ensure_ascii=False,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 F1-B bounded tuple/list recovery runner")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--worker-partition", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.audit_only:
        context = load_context(auth_path)
        print(json.dumps(audit_prefix(context), ensure_ascii=False))
        return 0
    if args.worker_partition:
        return worker(auth_path, args.worker_partition)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
