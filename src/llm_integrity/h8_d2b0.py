from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .h8_d2a import ATTACK_FAMILIES, ATTACK_ENDPOINTS_PER_FAMILY
from .h8_precalibration import canonical_sha256


D2B0_SCHEMA_VERSION = "h8-d2b0-development-sampling-authorization-preflight-1.0"
SMOKE_ROLE = "detector_development_attack_smoke_only"
SMOKE_COUNT = 8
UINT32_MAX = 2**32 - 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_lora_training_isolation(
    training_rows: Sequence[Mapping[str, Any]],
    fingerprint: Mapping[str, Any],
    confirmed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not training_rows:
        raise ValueError("D2 LoRA training data is empty")
    train_ids = [_normalized_text(row.get("id")) for row in training_rows]
    train_prompts = [_normalized_text(row.get("prompt")) for row in training_rows]
    train_answers = [_normalized_text(row.get("expected_answer")) for row in training_rows]
    if any(not value for value in train_ids + train_prompts + train_answers):
        raise ValueError("D2 LoRA training rows require nonempty id/prompt/expected_answer")
    if len(set(train_ids)) != len(train_ids) or len(set(train_prompts)) != len(train_prompts):
        raise ValueError("D2 LoRA training IDs/prompts must be unique")

    entries = list(fingerprint.get("entries", []))
    if len(entries) != 12:
        raise ValueError("Expected the frozen H6 MCC12 fingerprint")
    mcc_ids = {_normalized_text(row.get("prompt_id")) for row in entries}
    mcc_prompts = {_normalized_text(row.get("prompt")) for row in entries}
    mcc_responses: set[str] = set()
    for row in entries:
        for response in row.get("reference_responses", []) or []:
            if _normalized_text(response):
                mcc_responses.add(_normalized_text(response))

    confirmed_ids: set[str] = set()
    confirmed_prompts: set[str] = set()
    for row in confirmed_rows:
        confirmed_ids.add(_normalized_text(row.get("prompt_id", row.get("id"))))
        for key in ("prompt", "initial_prompt", "optimized_prompt"):
            value = _normalized_text(row.get(key))
            if value:
                confirmed_prompts.add(value)

    exact_mcc_id_overlap = sorted(set(train_ids) & mcc_ids)
    exact_mcc_prompt_overlap = sorted(set(train_prompts) & mcc_prompts)
    exact_mcc_response_overlap = sorted(set(train_prompts + train_answers) & mcc_responses)
    exact_confirmed_id_overlap = sorted(set(train_ids) & confirmed_ids)
    exact_confirmed_prompt_overlap = sorted(set(train_prompts) & confirmed_prompts)
    substring_mcc_prompt_overlap: list[dict[str, str]] = []
    for train_prompt in train_prompts:
        for mcc_prompt in mcc_prompts:
            if min(len(train_prompt), len(mcc_prompt)) >= 12 and (
                train_prompt in mcc_prompt or mcc_prompt in train_prompt
            ):
                substring_mcc_prompt_overlap.append(
                    {"training_prompt": train_prompt, "mcc_prompt": mcc_prompt}
                )
    passed = not any(
        (
            exact_mcc_id_overlap,
            exact_mcc_prompt_overlap,
            exact_mcc_response_overlap,
            exact_confirmed_id_overlap,
            exact_confirmed_prompt_overlap,
            substring_mcc_prompt_overlap,
        )
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "training_row_count": len(training_rows),
        "training_id_unique_count": len(set(train_ids)),
        "training_prompt_unique_count": len(set(train_prompts)),
        "mcc_prompt_count": len(mcc_prompts),
        "mcc_archived_response_count": len(mcc_responses),
        "confirmed_prompt_form_count": len(confirmed_prompts),
        "exact_mcc_id_overlap": exact_mcc_id_overlap,
        "exact_mcc_prompt_overlap": exact_mcc_prompt_overlap,
        "exact_mcc_response_overlap": exact_mcc_response_overlap,
        "exact_confirmed_id_overlap": exact_confirmed_id_overlap,
        "exact_confirmed_prompt_overlap": exact_confirmed_prompt_overlap,
        "substring_mcc_prompt_overlap": substring_mcc_prompt_overlap,
        "mcc12_prompt_or_response_leakage_detected": not passed,
    }
    if not passed:
        raise ValueError("D2 LoRA training data overlaps H6/MCC12 protected content")
    return result


def _family_group(value: str) -> str:
    family = value.lower()
    if family == "gaussian_noise":
        return "gaussian"
    if family in {"unstructured_pruning", "structured_pruning"}:
        return "pruning"
    if family == "finetuning":
        return "lora"
    return family


def audit_fresh_attack_provenance(
    d2_instances: Sequence[Mapping[str, Any]],
    h6_manifest_paths: Sequence[str | Path],
) -> dict[str, Any]:
    if len(d2_instances) != 8 or Counter(str(row.get("family")) for row in d2_instances) != Counter(
        {family: ATTACK_ENDPOINTS_PER_FAMILY for family in ATTACK_FAMILIES}
    ):
        raise ValueError("D2 freshness audit requires exactly eight frozen instances")
    source_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path_value in h6_manifest_paths:
        path = Path(path_value)
        rows = _read_jsonl(path)
        sources.append({"path": str(path), "sha256": file_sha256(path), "row_count": len(rows)})
        for row in rows:
            source_rows.append({**row, "source_manifest": str(path)})
    by_id = {str(row.get("variant_id")): row for row in source_rows}
    if len(by_id) != len(source_rows) or not by_id:
        raise ValueError("H6 attack manifests lack unique explicit variant IDs")
    h6_seeds = {int(row["seed"]) for row in source_rows}
    d2_ids = {str(row["attack_instance_id"]) for row in d2_instances}
    d2_seeds = {int(row["materialization_seed"]) for row in d2_instances}
    rows_out: list[dict[str, Any]] = []
    for instance in d2_instances:
        family = str(instance["family"])
        configuration = dict(instance["configuration"])
        comparable = [row for row in source_rows if _family_group(str(row.get("family", ""))) == family]
        exact_configuration = [
            str(row["variant_id"])
            for row in comparable
            if dict(row.get("configuration", {})) == configuration
        ]
        seed_overlap = [str(row["variant_id"]) for row in comparable if int(row["seed"]) == int(instance["materialization_seed"])]
        exact_instance = [
            str(row["variant_id"])
            for row in comparable
            if dict(row.get("configuration", {})) == configuration
            and int(row["seed"]) == int(instance["materialization_seed"])
        ]
        quantization_same_deterministic_instance = family == "quantization" and bool(exact_instance)
        rows_out.append(
            {
                "attack_instance_id": str(instance["attack_instance_id"]),
                "family": family,
                "materialization_seed": int(instance["materialization_seed"]),
                "configuration_sha256": canonical_sha256(configuration),
                "comparable_h6_variant_count": len(comparable),
                "id_overlap": str(instance["attack_instance_id"]) in by_id,
                "seed_overlap_variant_ids": seed_overlap,
                "exact_configuration_overlap_variant_ids": exact_configuration,
                "exact_configuration_and_seed_overlap_variant_ids": exact_instance,
                "quantization_same_deterministic_instance": quantization_same_deterministic_instance,
                "fresh_by_configuration_seed": not exact_configuration and not seed_overlap and not exact_instance,
            }
        )
    id_overlap = sorted(d2_ids & set(by_id))
    seed_overlap_values = sorted(d2_seeds & h6_seeds)
    passed = (
        not id_overlap
        and not seed_overlap_values
        and all(row["fresh_by_configuration_seed"] for row in rows_out)
        and not any(row["quantization_same_deterministic_instance"] for row in rows_out)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "h6_explicit_id_provenance_available": True,
        "id_level_overlap_status": "verified_zero" if not id_overlap else "overlap_detected",
        "h6_explicit_variant_id_count": len(by_id),
        "h6_manifest_sources": sources,
        "d2_instance_count": len(d2_instances),
        "id_overlap": id_overlap,
        "materialization_seed_overlap": seed_overlap_values,
        "configuration_seed_comparisons": rows_out,
        "fresh_by_construction_status": "verified_at_configuration_and_seed_level" if passed else "FAIL",
        "artifact_level_status": "pending_materialization_audit",
        "historical_in_memory_or_quantized_artifact_hash_limitation": (
            "H6 persisted LoRA adapters are available on the execution server; H6 Gaussian/pruning "
            "states and deterministic quantized model states were not archived as model payloads."
        ),
    }


def _derive_uint32(domain: str, root_seed: int, forbidden: set[int], *parts: object) -> tuple[int, int, str]:
    counter = 0
    while True:
        encoded = "\0".join([domain, str(root_seed), *(str(part) for part in parts), str(counter)]).encode()
        digest = hashlib.sha256(encoded).digest()
        value = int.from_bytes(digest[:4], "little")
        if 0 <= value <= UINT32_MAX and value not in forbidden:
            forbidden.add(value)
            return value, counter, digest.hex()
        counter += 1


@dataclass(frozen=True)
class AttackSmokeRequest:
    schedule_position: int
    attack_instance_id: str
    family: str
    materialization_seed: int
    generation_seed: int
    derivation_counter: int
    seed_digest_sha256: str
    prompt_index: int
    prompt_id: str
    prompt_sha256: str

    @property
    def response_id(self) -> str:
        return f"h8-d2b0-smoke-e{self.schedule_position:02d}-s{self.generation_seed:010d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "response_id": self.response_id,
            "data_role": SMOKE_ROLE,
            "formal_development_eligible": False,
            "nested_q10_q20_eligible": False,
            "configuration_selection_eligible": False,
        }


def build_attack_smoke_manifest(
    instances: Sequence[Mapping[str, Any]],
    fingerprint_entries: Sequence[Mapping[str, Any]],
    *,
    root_seed: int,
    forbidden_seeds: Iterable[int],
    formal_development_seeds: Iterable[int],
) -> dict[str, Any]:
    if len(instances) != SMOKE_COUNT or len(fingerprint_entries) != 12:
        raise ValueError("D2-B0 smoke requires eight endpoints and MCC12")
    formal = {int(value) for value in formal_development_seeds}
    used = {int(value) for value in forbidden_seeds} | formal
    requests: list[AttackSmokeRequest] = []
    for position, instance in enumerate(instances):
        entry = fingerprint_entries[position % len(fingerprint_entries)]
        seed, counter, digest = _derive_uint32(
            "h8-d2b0-attack-smoke-response-seed-v1",
            root_seed,
            used,
            str(instance["attack_instance_id"]),
            position,
        )
        requests.append(
            AttackSmokeRequest(
                schedule_position=position,
                attack_instance_id=str(instance["attack_instance_id"]),
                family=str(instance["family"]),
                materialization_seed=int(instance["materialization_seed"]),
                generation_seed=seed,
                derivation_counter=counter,
                seed_digest_sha256=digest,
                prompt_index=position,
                prompt_id=str(entry["prompt_id"]),
                prompt_sha256=str((entry.get("metadata") or {})["prompt_sha256"]),
            )
        )
    seeds = {row.generation_seed for row in requests}
    if len(seeds) != SMOKE_COUNT or seeds & formal:
        raise ValueError("D2-B0 smoke seeds are duplicate or overlap formal development seeds")
    return {
        "schema_version": D2B0_SCHEMA_VERSION,
        "artifact_status": "frozen_before_smoke_generation",
        "data_role": SMOKE_ROLE,
        "root_seed_uint32": int(root_seed),
        "derivation": "domain_separated_sha256_first_uint32_with_collision_counter",
        "request_count": SMOKE_COUNT,
        "unique_generation_seed_count": len(seeds),
        "formal_development_seed_overlap_count": len(seeds & formal),
        "formal_development_eligible": False,
        "nested_q10_q20_eligible": False,
        "configuration_selection_eligible": False,
        "requests": [row.as_dict() for row in requests],
    }


def parameter_state_sketch(
    model: Any,
    *,
    target_scope: str,
    seed: int,
    max_tensors: int = 48,
    samples_per_tensor: int = 8,
) -> dict[str, Any]:
    import torch

    from .paper_in_memory_attacks import _scope_matches

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and _scope_matches(name, target_scope)
    ]
    if not selected:
        raise ValueError(f"No parameters matched state-sketch scope {target_scope}")
    ranked = sorted(
        selected,
        key=lambda row: hashlib.sha256(f"{seed}\0{row[0]}".encode()).digest(),
    )[:max_tensors]
    records: list[dict[str, Any]] = []
    nonfinite = 0
    for name, parameter in ranked:
        flat = parameter.detach().reshape(-1)
        indices: list[int] = []
        values: list[str] = []
        for sample in range(min(samples_per_tensor, int(flat.numel()))):
            digest = hashlib.sha256(f"{seed}\0{name}\0{sample}".encode()).digest()
            index = int.from_bytes(digest[:8], "little") % int(flat.numel())
            value = float(flat[index].float().cpu().item())
            if not math.isfinite(value):
                nonfinite += 1
            indices.append(index)
            values.append(value.hex())
        records.append({"name": name, "shape": list(parameter.shape), "indices": indices, "float_hex": values})
    payload = {
        "target_scope": target_scope,
        "seed": int(seed),
        "selected_tensor_count": len(selected),
        "selected_parameter_count": sum(int(parameter.numel()) for _, parameter in selected),
        "sketched_tensor_count": len(ranked),
        "sample_count": sum(len(row["indices"]) for row in records),
        "nonfinite_sample_count": nonfinite,
        "records": records,
    }
    return {**payload, "state_sketch_sha256": canonical_sha256(payload)}


def save_canonical_envelope(path: str | Path, artifact_type: str, payload: Mapping[str, Any]) -> dict[str, str]:
    target = Path(path)
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": artifact_type,
        "schema_version": D2B0_SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return {"payload_sha256": payload_sha, "file_sha256": file_sha256(target)}


def load_canonical_envelope(
    path: str | Path,
    *,
    expected_artifact_type: str,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    target = Path(path)
    if file_sha256(target) != expected_file_sha256:
        raise ValueError("D2-B0 artifact file SHA256 mismatch")
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if (
        envelope.get("artifact_type") != expected_artifact_type
        or envelope.get("schema_version") != D2B0_SCHEMA_VERSION
        or not isinstance(payload, dict)
    ):
        raise ValueError("D2-B0 artifact type/schema mismatch")
    actual = canonical_sha256(payload)
    if actual != envelope.get("payload_sha256") or actual != expected_payload_sha256:
        raise ValueError("D2-B0 artifact payload SHA256 mismatch")
    return payload
