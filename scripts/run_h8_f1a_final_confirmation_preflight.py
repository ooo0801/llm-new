#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d2b0 import audit_lora_training_isolation
from llm_integrity.h8_f1a import (
    ALPHA,
    ATTACK_FAMILIES,
    ATTACK_ENDPOINTS_PER_FAMILY,
    ATTACK_RATE_SCOPE,
    ATTACK_TOTAL,
    FINAL_ATTACK_ROLE,
    FINAL_ESTIMAND,
    FINAL_INTACT_ROLE,
    FORMAL_TOTAL,
    F1A_SCHEMA_VERSION,
    INTACT_TOTAL,
    INTACT_UNIT_COUNT,
    N_REFERENCE,
    N_TARGET,
    INTERVAL_SCOPE,
    PERMUTATIONS,
    PROMPT_COUNT,
    REFERENCE_TOTAL,
    TOP_R,
    UINT32_MAX,
    build_attack_endpoints,
    build_generation_schedule,
    build_permutation_seed_manifest,
    endpoint_configuration_sha256,
    file_sha256,
    load_frozen_detector,
    save_canonical_envelope,
    validate_attack_endpoints,
)
from llm_integrity.h8_precalibration import canonical_sha256


DEFAULT_CONFIG = ROOT / "configs/h8_f1a_final_confirmation_preflight.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch")


def _objects(path: Path) -> Iterable[Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        yield yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _walk_identities(
    value: Any,
    *,
    key: str = "",
    seeds: set[int],
    response_ids: set[str],
    endpoint_ids: set[str],
) -> None:
    lowered = key.lower()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _walk_identities(
                child,
                key=str(child_key),
                seeds=seeds,
                response_ids=response_ids,
                endpoint_ids=endpoint_ids,
            )
        return
    if isinstance(value, list):
        for child in value:
            _walk_identities(
                child,
                key=key,
                seeds=seeds,
                response_ids=response_ids,
                endpoint_ids=endpoint_ids,
            )
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int) and "seed" in lowered and 0 <= value <= UINT32_MAX:
        seeds.add(value)
    elif isinstance(value, str):
        if lowered == "response_id" and value:
            response_ids.add(value)
        if lowered in {"attack_instance_id", "variant_id"} and value:
            endpoint_ids.add(value)


def collect_historical_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    candidates: list[Path] = []
    candidates.extend(resolve(value) for value in frozen["historical_seed_manifests"])
    h6_archive = resolve("reproducibility/fingerprint_h6_qwen32b_mcc_20260812")
    candidates.extend(
        path
        for path in h6_archive.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml"}
    )
    optional_server_roots = (
        resolve("results/experiment_h6_qwen32b_reconstruction_20260812"),
        resolve("results/h8_qwen32b_mmd_precalibration"),
        resolve("results/h8_qwen32b_score_calibration"),
        resolve("results/h8_qwen32b_detector_development"),
    )
    for root in optional_server_roots:
        if root.is_dir():
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml"}
                and path.stat().st_size <= 256 * 1024 * 1024
            )
    unique_paths = sorted(set(candidates))
    seeds: set[int] = set()
    response_ids: set[str] = set()
    endpoint_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        before = (len(seeds), len(response_ids), len(endpoint_ids))
        for obj in _objects(path):
            _walk_identities(
                obj,
                seeds=seeds,
                response_ids=response_ids,
                endpoint_ids=endpoint_ids,
            )
        after = (len(seeds), len(response_ids), len(endpoint_ids))
        sources.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "file_sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "new_uint32_seed_count": after[0] - before[0],
                "new_response_id_count": after[1] - before[1],
                "new_endpoint_id_count": after[2] - before[2],
            }
        )
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "source_count": len(sources),
        "sources": sources,
        "uint32_seed_count": len(seeds),
        "uint32_seed_set_sha256": canonical_sha256(sorted(seeds)),
        "response_id_count": len(response_ids),
        "response_id_set_sha256": canonical_sha256(sorted(response_ids)),
        "endpoint_id_count": len(endpoint_ids),
        "endpoint_id_set_sha256": canonical_sha256(sorted(endpoint_ids)),
        "server_ignored_results_included_when_available": True,
        "seeds": sorted(seeds),
        "response_ids": sorted(response_ids),
        "endpoint_ids": sorted(endpoint_ids),
    }


def historical_attack_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for logical in config["frozen_inputs"]["historical_h6_attack_manifests"]:
        path = resolve(logical)
        source_rows = list(_objects(path))
        rows.extend(source_rows)
        sources.append({"path": logical, "file_sha256": file_sha256(path), "row_count": len(source_rows)})
    d2_path = resolve(
        "reproducibility/h8_qwen32b_detector_development_20260820/d2a/D2A_FRESH_ATTACK_INSTANCE_MANIFEST.json"
    )
    d2_envelope = json.loads(d2_path.read_text(encoding="utf-8"))
    d2_rows = list(d2_envelope["payload"]["instances"])
    rows.extend(d2_rows)
    sources.append({"path": str(d2_path.relative_to(ROOT)).replace("\\", "/"), "file_sha256": file_sha256(d2_path), "row_count": len(d2_rows)})
    configs = {
        canonical_sha256(dict(row["configuration"]))
        for row in rows
        if isinstance(row.get("configuration"), Mapping)
    }
    ids = {
        str(row.get("attack_instance_id") or row.get("variant_id"))
        for row in rows
        if row.get("attack_instance_id") or row.get("variant_id")
    }
    seeds = {
        int(row.get("materialization_seed", row.get("seed")))
        for row in rows
        if row.get("materialization_seed", row.get("seed")) is not None
    }
    return {
        "sources": sources,
        "rows": rows,
        "configuration_sha256": configs,
        "endpoint_ids": ids,
        "materialization_seeds": seeds,
    }


def _derive_uint32(domain: str, root_seed: int, forbidden: set[int], *parts: object) -> tuple[int, int, str]:
    counter = 0
    while True:
        material = "\0".join([domain, str(root_seed), *(str(part) for part in parts), str(counter)])
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little")
        if seed not in forbidden:
            forbidden.add(seed)
            return seed, counter, hashlib.sha256(material.encode("utf-8")).hexdigest()
        counter += 1


def build_smoke_manifest(
    config: Mapping[str, Any],
    entries: list[dict[str, Any]],
    *,
    forbidden_seeds: set[int],
    forbidden_response_ids: set[str],
    formal_endpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    smoke = config["tiny_smoke"]
    endpoint_configs = list(smoke["attack_endpoints"])
    if len(endpoint_configs) != 4 or {row["family"] for row in endpoint_configs} != set(ATTACK_FAMILIES):
        raise ValueError("F1-A tiny smoke requires one endpoint per attack family")
    formal_config_hashes = {endpoint_configuration_sha256(row) for row in formal_endpoints}
    material_root = int(config["seed_roots"]["tiny_smoke_attack_materialization_uint32"])
    training_root = int(config["seed_roots"]["tiny_smoke_lora_training_uint32"])
    normalized_endpoints: list[dict[str, Any]] = []
    for index, raw in enumerate(endpoint_configs):
        row = dict(raw)
        config_sha = canonical_sha256(dict(row["configuration"]))
        if config_sha in formal_config_hashes:
            raise ValueError("Smoke attack configuration overlaps formal endpoint configuration")
        material_seed, _, _ = _derive_uint32(
            "h8-f1a-smoke-materialization-v1", material_root, forbidden_seeds, row["family"], index, config_sha
        )
        training_seed = None
        if row["family"] == "lora":
            training_seed, _, _ = _derive_uint32(
                "h8-f1a-smoke-lora-training-v1", training_root, forbidden_seeds, row["family"], index, config_sha
            )
        normalized_endpoints.append(
            {
                **row,
                "materialization_seed": material_seed,
                "training_seed": training_seed,
                "configuration_sha256": config_sha,
                "formal_endpoint": False,
                "detector_evaluation_eligible": False,
            }
        )
    roles = [
        "final_reference_smoke_only",
        "final_heldout_intact_smoke_only",
        *["final_heldout_attack_smoke_only"] * 4,
    ]
    root_seed = int(config["seed_roots"]["tiny_smoke_generation_uint32"])
    requests: list[dict[str, Any]] = []
    for position, role in enumerate(roles):
        prompt = entries[position]
        seed, counter, digest = _derive_uint32(
            "h8-f1a-tiny-smoke-generation-v1", root_seed, forbidden_seeds, role, position, prompt["prompt_id"]
        )
        response_id = f"h8-f1a-smoke-{position:02d}-s{seed:010d}"
        if response_id in forbidden_response_ids:
            raise ValueError("F1-A smoke response ID overlaps history")
        forbidden_response_ids.add(response_id)
        endpoint = None if position < 2 else normalized_endpoints[position - 2]
        requests.append(
            {
                "schedule_position": position,
                "data_role": role,
                "response_id": response_id,
                "generation_seed": seed,
                "derivation_counter": counter,
                "seed_digest_sha256": digest,
                "prompt_index": position,
                "prompt_id": prompt["prompt_id"],
                "prompt_sha256": prompt["metadata"]["prompt_sha256"],
                "attack_instance_id": None if endpoint is None else endpoint["attack_instance_id"],
                "attack_family": None if endpoint is None else endpoint["family"],
                "formal_final_confirmation_eligible": False,
                "detector_statistics_eligible": False,
                "measurement_or_score_fit_eligible": False,
            }
        )
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "artifact_status": "frozen_before_tiny_smoke",
        "formal_final_confirmation_eligible": False,
        "detector_statistics_forbidden": True,
        "exact_response_count": 6,
        "unique_generation_seed_count": 6,
        "attack_endpoints": normalized_endpoints,
        "requests": requests,
    }


def run(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "H8_F1A_FRESH_HELDOUT_FINAL_CONFIRMATION_PROTOCOL_FREEZE_PREFLIGHT":
        raise ValueError("Invalid F1-A phase")
    if config.get("formal_final_confirmation_sampling_authorized") is not False:
        raise PermissionError("F1-A must not authorize 12,720 formal responses")
    if config.get("tiny_smoke_sampling_authorized") is not True:
        raise PermissionError("F1-A tiny smoke authorization is missing")
    if not all(value is True for value in config["forbidden_operations"].values()):
        raise PermissionError("Every F1-A forbidden operation must remain enabled")
    design = config["formal_design"]
    frozen_reporting_contract = {
        "reference_bank_shared_across_units": True,
        "reference_bank_count": 1,
        "primary_estimand": FINAL_ESTIMAND,
        "interval_label": "conditional_clopper_pearson_95ci",
        "interval_interpretation": INTERVAL_SCOPE,
        "unconditional_reference_regeneration_coverage_claimed": False,
        "attack_rate_scope": ATTACK_RATE_SCOPE,
        "iid_attack_superpopulation_claimed": False,
    }
    for key, expected in frozen_reporting_contract.items():
        if design.get(key) != expected:
            raise ValueError(f"F1-A Scheme-A reporting contract changed: {key}")
    detector = load_frozen_detector(
        resolve(config["frozen_detector"]["archive"]),
        expected_manifest_sha256=config["frozen_detector"]["manifest_sha256"],
    )
    frozen = config["frozen_inputs"]
    for path_key, hash_key, label in (
        ("h8_generation_config", "h8_generation_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12"),
        ("confirmed_prompt_pool", "confirmed_prompt_pool_sha256", "H6 confirmed prompt pool"),
        ("final_lora_training_data", "final_lora_training_data_sha256", "F1 LoRA training data"),
        ("protocol", "protocol_sha256", "F1-A protocol"),
    ):
        require_file_hash(resolve(frozen[path_key]), frozen[hash_key], label)
    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    entries = list(fingerprint.get("entries", []))
    if len(entries) != PROMPT_COUNT:
        raise ValueError("Frozen MCC12 count changed")
    training_path = resolve(frozen["final_lora_training_data"])
    training_rows = list(_objects(training_path))
    confirmed_rows = list(
        _objects(
            resolve(frozen["confirmed_prompt_pool"])
        )
    )
    isolation = audit_lora_training_isolation(training_rows, fingerprint, confirmed_rows)
    if isolation.get("status") != "PASS" or isolation.get("mcc12_prompt_or_response_leakage_detected") is not False:
        raise ValueError("F1-A LoRA training data isolation gate failed")

    historical = collect_historical_identities(config)
    attack_history = historical_attack_evidence(config)
    forbidden_seeds = set(historical["seeds"])
    formal_endpoints_raw = build_attack_endpoints(
        config["attack_design"],
        materialization_root_seed=int(config["seed_roots"]["formal_attack_materialization_uint32"]),
        lora_training_root_seed=int(config["seed_roots"]["formal_lora_training_uint32"]),
        forbidden_seeds=forbidden_seeds,
    )
    endpoint_audit = validate_attack_endpoints(
        formal_endpoints_raw,
        historical_endpoint_ids=attack_history["endpoint_ids"],
        historical_materialization_seeds=attack_history["materialization_seeds"],
        historical_configuration_sha256=attack_history["configuration_sha256"],
    )
    formal_endpoints = endpoint_audit["endpoints"]
    final_attack_seeds = {
        int(row["materialization_seed"]) for row in formal_endpoints
    } | {
        int(row["training_seed"]) for row in formal_endpoints if row.get("training_seed") is not None
    }
    schedule = build_generation_schedule(
        entries,
        formal_endpoints,
        generation_root_seed=int(config["seed_roots"]["formal_generation_uint32"]),
        order_root_seed=int(config["seed_roots"]["formal_order_uint64"]),
        forbidden_generation_seeds=set(historical["seeds"]) | final_attack_seeds,
        forbidden_response_ids=historical["response_ids"],
    )
    formal_generation_seeds = {int(row["generation_seed"]) for row in schedule["requests"]}
    if formal_generation_seeds & set(historical["seeds"]) or formal_generation_seeds & final_attack_seeds:
        raise ValueError("F1-A formal generation seed isolation failed")
    units = [
        {"evaluation_unit_id": f"final_intact_unit_{index:03d}", "data_role": FINAL_INTACT_ROLE, "attack_family": None}
        for index in range(INTACT_UNIT_COUNT)
    ] + [
        {
            "evaluation_unit_id": row["attack_instance_id"],
            "data_role": FINAL_ATTACK_ROLE,
            "attack_family": row["family"],
        }
        for row in formal_endpoints
    ]
    permutation = build_permutation_seed_manifest(
        units,
        [str(entry["prompt_id"]) for entry in entries],
        permutation_root_seed=int(config["seed_roots"]["final_global_permutation_uint64"]),
    )
    smoke = build_smoke_manifest(
        config,
        entries,
        forbidden_seeds=set(historical["seeds"]) | final_attack_seeds | formal_generation_seeds,
        forbidden_response_ids=set(historical["response_ids"]) | {str(row["response_id"]) for row in schedule["requests"]},
        formal_endpoints=formal_endpoints,
    )
    smoke_seeds = {int(row["generation_seed"]) for row in smoke["requests"]}
    if smoke_seeds & formal_generation_seeds or smoke_seeds & set(historical["seeds"]):
        raise ValueError("F1-A smoke seed isolation failed")

    out = resolve(config["artifacts"]["preflight_directory"])
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    artifacts["historical_identity_index"] = save_canonical_envelope(
        out / config["artifacts"]["historical_identity_index"],
        "h8_f1a_historical_identity_index",
        historical,
    )
    artifacts["formal_attack_manifest"] = save_canonical_envelope(
        out / config["artifacts"]["formal_attack_manifest"],
        "h8_f1a_fresh_heldout_attack_endpoint_manifest",
        {
            "schema_version": F1A_SCHEMA_VERSION,
            "artifact_status": "frozen_before_formal_materialization",
            "formal_endpoint_count": 40,
            "family_counts": endpoint_audit["family_counts"],
            "endpoints": formal_endpoints,
        },
    )
    artifacts["formal_generation_manifest"] = save_canonical_envelope(
        out / config["artifacts"]["formal_generation_manifest"],
        "h8_f1a_final_generation_manifest",
        schedule,
    )
    artifacts["permutation_seed_manifest"] = save_canonical_envelope(
        out / config["artifacts"]["permutation_seed_manifest"],
        "h8_f1a_final_global_permutation_seed_manifest",
        permutation,
    )
    freshness_payload = {
        **{key: value for key, value in endpoint_audit.items() if key != "endpoints"},
        "historical_attack_sources": attack_history["sources"],
        "historical_seed_source_count": historical["source_count"],
        "historical_uint32_seed_count": historical["uint32_seed_count"],
        "formal_generation_seed_count": len(formal_generation_seeds),
        "formal_generation_historical_overlap_count": 0,
        "formal_generation_attack_seed_overlap_count": 0,
        "formal_response_id_historical_overlap_count": 0,
        "smoke_formal_generation_seed_overlap_count": 0,
        "smoke_historical_seed_overlap_count": 0,
        "lora_training_isolation": isolation,
    }
    artifacts["freshness_audit"] = save_canonical_envelope(
        out / config["artifacts"]["freshness_audit"],
        "h8_f1a_freshness_audit",
        freshness_payload,
    )
    artifacts["smoke_manifest"] = save_canonical_envelope(
        out / config["artifacts"]["smoke_manifest"],
        "h8_f1a_tiny_smoke_manifest",
        smoke,
    )
    report = {
        "schema_version": F1A_SCHEMA_VERSION,
        "phase": config["phase"],
        "status": "PASS_READY_FOR_EXACTLY_SIX_SMOKE_ONLY",
        "created_at_utc": now(),
        "frozen_detector_manifest_sha256": config["frozen_detector"]["manifest_sha256"],
        "selected_detector_payload_sha256": detector["manifest"]["selected_detector_payload_sha256"],
        "selected_configuration_id": "r60_q10_top2",
        "n_reference": N_REFERENCE,
        "n_target": N_TARGET,
        "top_r": TOP_R,
        "alpha": ALPHA,
        "global_permutations": PERMUTATIONS,
        "global_p_value_formula": "(1 + count(T_perm >= T_observed)) / 1000",
        "formal_design": {
            "reference_responses": REFERENCE_TOTAL,
            "intact_target_responses": INTACT_TOTAL,
            "attack_responses": ATTACK_TOTAL,
            "total_responses": FORMAL_TOTAL,
            "intact_evaluation_units": INTACT_UNIT_COUNT,
            "attack_endpoints_per_family": ATTACK_ENDPOINTS_PER_FAMILY,
            "attack_families": list(ATTACK_FAMILIES),
        },
        "formal_final_confirmation_responses": 0,
        "formal_reference_responses": 0,
        "formal_heldout_intact_responses": 0,
        "formal_heldout_attack_responses": 0,
        "tiny_smoke_responses": 0,
        "formal_sampling_authorized": False,
        "formal_detector_statistics_computed": False,
        "measurement_or_score_refit_performed": False,
        "detector_configuration_changed": False,
        "exact_binomial_interval": "two_sided_clopper_pearson_95_percent",
        "primary_estimand": FINAL_ESTIMAND,
        "reference_bank_count": 1,
        "reference_bank_shared_across_all_evaluation_units": True,
        "interval_label": "conditional_clopper_pearson_95ci",
        "interval_scope_note": INTERVAL_SCOPE,
        "unconditional_reference_regeneration_coverage_claimed": False,
        "attack_rate_scope": ATTACK_RATE_SCOPE,
        "iid_attack_superpopulation_claimed": False,
        "shared_reference_dependence_disclosed": True,
        "between_reference_bank_uncertainty_included": False,
        "freshness_status": freshness_payload["status"],
        "formal_endpoint_materialization_status": "not_materialized_preflight_only",
        "implementation_files": {
            logical: file_sha256(resolve(logical))
            for logical in (
                "src/llm_integrity/h8_f1a.py",
                "src/llm_integrity/modeling.py",
                "src/llm_integrity/paper_quantization.py",
                "scripts/run_h8_f1a_final_confirmation_preflight.py",
                "scripts/run_h8_f1a_tiny_smoke.py",
            )
        },
        "artifacts": artifacts,
        "next_gate": "RUN_EXACTLY_SIX_SMOKE_ONLY_THEN_STOP_FOR_SEPARATE_FORMAL_SAMPLING_AUTHORIZATION",
    }
    report_path = out / config["artifacts"]["report"]
    atomic_json(report_path, report)
    artifacts["report"] = {"file_sha256": file_sha256(report_path)}
    index = {
        "schema_version": F1A_SCHEMA_VERSION,
        "created_at_utc": now(),
        "config": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "file_sha256": file_sha256(config_path)},
        "protocol": {"path": frozen["protocol"], "file_sha256": file_sha256(resolve(frozen["protocol"]))},
        "training_data": {"path": frozen["final_lora_training_data"], "file_sha256": file_sha256(training_path)},
        "artifacts": artifacts,
    }
    index_path = out / config["artifacts"]["sha256_index"]
    atomic_json(index_path, index)
    print(json.dumps({"status": report["status"], "formal_responses": 0, "formal_schedule": FORMAL_TOTAL, "smoke_planned": 6, "output": str(out)}, ensure_ascii=False))
    return out


if __name__ == "__main__":
    run()
