from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .h8_m0ij import mmd2_unbiased_from_kernel
from .h8_precalibration import canonical_sha256
from .h8_score_calibration import SAMPLE_STRUCTURES, score_value


D2A_SCHEMA_VERSION = "h8-d2a-detector-development-1.0"
PROMPT_COUNT = 12
TOP_R_VALUES = (2, 3, 4)
DEVELOPMENT_PERMUTATIONS = 999
DEVELOPMENT_ALPHA = 0.05
REFERENCE_ROLE = "detector_development_reference_only"
INTACT_TARGET_ROLE = "detector_development_intact_target_only"
ATTACK_ROLE = "detector_development_attack_only"
ALLOWED_DEVELOPMENT_ROLES = {REFERENCE_ROLE, INTACT_TARGET_ROLE, ATTACK_ROLE}
FORBIDDEN_SELECTION_ROLE_MARKERS = ("formal", "heldout", "final", "confirmation")
ATTACK_FAMILIES = ("gaussian", "pruning", "lora", "quantization")
INTACT_DEVELOPMENT_UNITS = 5
ATTACK_ENDPOINTS_PER_FAMILY = 2
STRUCTURE_SIZES = {
    "r40_q10": (40, 10),
    "r40_q20": (40, 20),
    "r60_q10": (60, 10),
    "r60_q20": (60, 20),
}
UINT32_MAX = 2**32 - 1


def top_r_sum(scores: Sequence[float], r: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (PROMPT_COUNT,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Top-r requires exactly twelve finite nonnegative frozen scores")
    if r not in TOP_R_VALUES:
        raise ValueError("Primary aggregation only permits Top-2, Top-3, or Top-4")
    return float(np.sort(values)[-r:].sum(dtype=np.float64))


def max_score_diagnostic(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (PROMPT_COUNT,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Max diagnostic requires twelve finite nonnegative scores")
    return float(values.max())


def empirical_global_p_value(observed: float, permutations: Sequence[float]) -> float:
    values = np.asarray(permutations, dtype=np.float64)
    if not math.isfinite(float(observed)) or values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Global p-value inputs must be finite")
    return float((1 + int(np.sum(values >= float(observed)))) / (values.size + 1))


def _uint64(domain: str, root_seed: int, *parts: object) -> int:
    data = "\0".join([domain, str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(data.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class GlobalPermutationResult:
    observed_scores: tuple[float, ...]
    observed_top_r: float
    diagnostic_max: float
    permutation_top_r: tuple[float, ...]
    exceedance_count: int
    global_p_value: float
    detected: bool
    permutations: int
    alpha: float


def global_permutation_test(
    kernels_by_prompt: Mapping[str, np.ndarray],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    n_reference: int,
    n_target: int,
    top_r: int,
    permutation_root_seed: int,
    permutations: int = DEVELOPMENT_PERMUTATIONS,
    alpha: float = DEVELOPMENT_ALPHA,
) -> GlobalPermutationResult:
    prompt_ids = tuple(sorted(kernels_by_prompt))
    if len(prompt_ids) != PROMPT_COUNT or set(prompt_ids) != set(score_parameters_by_prompt):
        raise ValueError("Global permutation requires exactly twelve matched prompt kernels/parameters")
    if min(n_reference, n_target) < 2 or permutations < 1:
        raise ValueError("Invalid global permutation group sizes/count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    observed_scores: list[float] = []
    matrices: dict[str, np.ndarray] = {}
    reference = np.arange(n_reference, dtype=np.int64)
    target = np.arange(n_reference, n_reference + n_target, dtype=np.int64)
    for prompt_id in prompt_ids:
        kernel = np.asarray(kernels_by_prompt[prompt_id], dtype=np.float64)
        if kernel.shape != (n_reference + n_target, n_reference + n_target) or not np.isfinite(kernel).all():
            raise ValueError(f"Invalid pooled kernel for {prompt_id}")
        matrices[prompt_id] = kernel
        raw = mmd2_unbiased_from_kernel(kernel, reference, target)
        observed_scores.append(score_value(raw, score_parameters_by_prompt[prompt_id]))
    observed = top_r_sum(observed_scores, top_r)
    permutation_values: list[float] = []
    for permutation_id in range(permutations):
        scores: list[float] = []
        for prompt_id in prompt_ids:
            seed = _uint64(
                "h8-d2a-within-prompt-permutation-v1",
                permutation_root_seed,
                permutation_id,
                prompt_id,
            )
            order = np.random.Generator(np.random.PCG64(seed)).permutation(n_reference + n_target)
            perm_reference = order[:n_reference]
            perm_target = order[n_reference:]
            raw = mmd2_unbiased_from_kernel(matrices[prompt_id], perm_reference, perm_target)
            scores.append(score_value(raw, score_parameters_by_prompt[prompt_id]))
        permutation_values.append(top_r_sum(scores, top_r))
    p_value = empirical_global_p_value(observed, permutation_values)
    return GlobalPermutationResult(
        observed_scores=tuple(observed_scores),
        observed_top_r=observed,
        diagnostic_max=max_score_diagnostic(observed_scores),
        permutation_top_r=tuple(permutation_values),
        exceedance_count=sum(value >= observed for value in permutation_values),
        global_p_value=p_value,
        detected=p_value <= alpha,
        permutations=permutations,
        alpha=alpha,
    )


def development_global_permutation_test(
    kernels_by_prompt: Mapping[str, np.ndarray],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    structure: str,
    top_r: int,
    permutation_root_seed: int,
) -> GlobalPermutationResult:
    if structure not in STRUCTURE_SIZES:
        raise ValueError("Unknown frozen sample structure")
    n_reference, n_target = STRUCTURE_SIZES[structure]
    return global_permutation_test(
        kernels_by_prompt,
        score_parameters_by_prompt,
        n_reference=n_reference,
        n_target=n_target,
        top_r=top_r,
        permutation_root_seed=permutation_root_seed,
        permutations=DEVELOPMENT_PERMUTATIONS,
        alpha=DEVELOPMENT_ALPHA,
    )


def evaluate_development_unit(
    kernels_by_prompt: Mapping[str, np.ndarray],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    data_role: str,
    evaluation_unit_id: str,
    structure: str,
    top_r: int,
    permutation_root_seed: int,
    attack_family: str | None = None,
) -> dict[str, Any]:
    if data_role not in {INTACT_TARGET_ROLE, ATTACK_ROLE}:
        raise ValueError("Detector evaluation may read development intact or attack data only")
    if any(marker in data_role.lower() for marker in FORBIDDEN_SELECTION_ROLE_MARKERS):
        raise ValueError("Final/held-out roles are forbidden in detector development")
    if not evaluation_unit_id:
        raise ValueError("Development evaluation unit ID is required")
    if data_role == ATTACK_ROLE:
        if attack_family not in ATTACK_FAMILIES:
            raise ValueError("Development attack evaluation requires a registered family")
    elif attack_family is not None:
        raise ValueError("Intact development evaluation cannot carry an attack family")
    result = development_global_permutation_test(
        kernels_by_prompt,
        score_parameters_by_prompt,
        structure=structure,
        top_r=top_r,
        permutation_root_seed=permutation_root_seed,
    )
    return {
        "data_role": data_role,
        "evaluation_unit_id": evaluation_unit_id,
        "attack_family": attack_family,
        "configuration_id": configuration_id(structure, top_r),
        "sample_structure": structure,
        "top_r": top_r,
        "observed_top_r": result.observed_top_r,
        "diagnostic_max": result.diagnostic_max,
        "global_p_value": result.global_p_value,
        "detected": result.detected,
        "permutations": result.permutations,
        "alpha": result.alpha,
        "local_p_values_used": False,
        "measurement_or_score_refit_performed": False,
    }


def configuration_id(structure: str, top_r: int) -> str:
    if structure not in STRUCTURE_SIZES or top_r not in TOP_R_VALUES:
        raise ValueError("Invalid detector configuration")
    return f"{structure}_top{top_r}"


def configuration_manifest_payload() -> dict[str, Any]:
    rows = []
    for structure in SAMPLE_STRUCTURES:
        n_reference, n_target = STRUCTURE_SIZES[structure]
        for top_r in TOP_R_VALUES:
            rows.append(
                {
                    "configuration_id": configuration_id(structure, top_r),
                    "sample_structure": structure,
                    "n_reference": n_reference,
                    "n_target": n_target,
                    "primary_aggregation": f"top_{top_r}_sum",
                    "top_r": top_r,
                    "max_diagnostic_only": True,
                    "local_p_values_used": False,
                }
            )
    return {
        "schema_version": D2A_SCHEMA_VERSION,
        "configuration_count": 12,
        "configurations": rows,
        "permutations": DEVELOPMENT_PERMUTATIONS,
        "alpha": DEVELOPMENT_ALPHA,
        "energy_implemented": False,
        "selection_rule": {
            "primary": "minimize_false_positive_count_across_five_development_intact_units",
            "tie_1": "maximize_minimum_detected_endpoint_count_across_four_attack_families",
            "tie_2": "maximize_total_detected_count_across_eight_development_attack_endpoints",
            "tie_3": "prefer_q10_over_q20",
            "tie_4": "prefer_r40_over_r60",
            "tie_5": "fixed_top_r_order",
            "fixed_top_r_order": [2, 3, 4],
            "single_gaussian_direct_selection_forbidden": True,
            "development_counts_are_not_formal_fpr_or_tpr_estimates": True,
        },
    }


def select_development_configuration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    configurations = {row["configuration_id"]: row for row in configuration_manifest_payload()["configurations"]}
    attack_grouped: dict[str, dict[str, dict[str, bool]]] = {
        config_id: {family: {} for family in ATTACK_FAMILIES} for config_id in configurations
    }
    intact_grouped: dict[str, dict[str, bool]] = {config_id: {} for config_id in configurations}
    for record in records:
        role = str(record.get("data_role", ""))
        if role not in {ATTACK_ROLE, INTACT_TARGET_ROLE} or any(
            marker in role.lower() for marker in FORBIDDEN_SELECTION_ROLE_MARKERS
        ):
            raise ValueError("Configuration selection may read development intact/attack data only")
        config_id = str(record.get("configuration_id", ""))
        if config_id not in configurations:
            raise ValueError("Unknown detector configuration")
        detected = record.get("detected")
        if not isinstance(detected, bool):
            raise ValueError("Development detection result must be boolean")
        unit_id = str(record.get("evaluation_unit_id", "")).strip()
        if not unit_id:
            raise ValueError("Every selection record requires an evaluation_unit_id")
        if role == INTACT_TARGET_ROLE:
            if record.get("attack_family") is not None:
                raise ValueError("Development intact units cannot carry an attack family")
            if unit_id in intact_grouped[config_id]:
                raise ValueError("Duplicate intact evaluation unit for a configuration")
            intact_grouped[config_id][unit_id] = detected
        else:
            family = str(record.get("attack_family", ""))
            if family not in ATTACK_FAMILIES:
                raise ValueError("Unknown development attack family")
            if unit_id in attack_grouped[config_id][family]:
                raise ValueError("Duplicate attack endpoint for a configuration/family")
            attack_grouped[config_id][family][unit_id] = detected
    metrics: dict[str, Any] = {}
    for config_id, by_family in attack_grouped.items():
        intact = intact_grouped[config_id]
        if len(intact) != INTACT_DEVELOPMENT_UNITS:
            raise ValueError("Every configuration requires exactly five development intact units")
        if any(len(by_family[family]) != ATTACK_ENDPOINTS_PER_FAMILY for family in ATTACK_FAMILIES):
            raise ValueError("Every configuration requires exactly two fresh endpoints per family")
        family_counts = {
            family: int(sum(by_family[family].values())) for family in ATTACK_FAMILIES
        }
        metrics[config_id] = {
            "development_false_positive_count": int(sum(intact.values())),
            "development_intact_unit_count": INTACT_DEVELOPMENT_UNITS,
            "family_detected_endpoint_counts": family_counts,
            "minimum_family_detected_endpoint_count": min(family_counts.values()),
            "total_detected_attack_endpoint_count": sum(family_counts.values()),
            "development_attack_endpoint_count": len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY,
            "counts_are_not_formal_fpr_or_tpr_estimates": True,
        }
    r_order = {r: index for index, r in enumerate(TOP_R_VALUES)}

    def key(config_id: str) -> tuple[int, int, int, int, int, int]:
        config = configurations[config_id]
        metric = metrics[config_id]
        return (
            int(metric["development_false_positive_count"]),
            -int(metric["minimum_family_detected_endpoint_count"]),
            -int(metric["total_detected_attack_endpoint_count"]),
            int(config["n_target"]),
            int(config["n_reference"]),
            r_order[int(config["top_r"])],
        )

    selected = min(sorted(configurations), key=key)
    return {
        "selected_configuration_id": selected,
        "selection_metrics": metrics,
        "selection_key": list(key(selected)),
        "selection_rule": configuration_manifest_payload()["selection_rule"],
        "final_or_heldout_data_read": False,
        "formal_fpr_or_tpr_estimation_performed": False,
    }


def intact_sanity_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Intact sanity requires development records")
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if record.get("data_role") != INTACT_TARGET_ROLE:
            raise ValueError("Intact sanity may read development intact targets only")
        config_id = str(record.get("configuration_id", ""))
        if config_id not in {row["configuration_id"] for row in configuration_manifest_payload()["configurations"]}:
            raise ValueError("Unknown intact-sanity configuration")
        if not isinstance(record.get("detected"), bool):
            raise ValueError("Intact detection result must be boolean")
        grouped[config_id].append(bool(record["detected"]))
    return {
        "by_configuration": {
            config_id: {
                "replicates": len(values),
                "development_false_positive_count": int(sum(values)),
                "not_a_formal_fpr_estimate": True,
            }
            for config_id, values in sorted(grouped.items())
        },
        "used_for_configuration_selection": False,
    }


@dataclass(frozen=True)
class DevelopmentGenerationRequest:
    schedule_position: int
    bank_position: int
    replicate_id: int
    prompt_position_in_round: int
    prompt_index: int
    prompt_id: str
    prompt_sha256: str
    generation_seed: int
    derivation_counter: int
    seed_digest_sha256: str
    data_role: str
    bank_id: str
    attack_family: str | None = None
    attack_instance_id: str | None = None

    @property
    def response_id(self) -> str:
        bank = self.bank_id.replace("_", "-")
        return f"h8-d2a-{bank}-r{self.replicate_id:03d}-p{self.prompt_index:02d}-s{self.generation_seed:010d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_position": self.schedule_position,
            "bank_position": self.bank_position,
            "replicate_id": self.replicate_id,
            "prompt_position_in_round": self.prompt_position_in_round,
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "generation_seed": self.generation_seed,
            "derivation_counter": self.derivation_counter,
            "seed_digest_sha256": self.seed_digest_sha256,
            "response_id": self.response_id,
            "data_role": self.data_role,
            "bank_id": self.bank_id,
            "attack_family": self.attack_family,
            "attack_instance_id": self.attack_instance_id,
            "eligible_for_future_formal_reference": False,
            "eligible_for_future_final_heldout": False,
            "eligible_for_future_final_confirmation": False,
            "eligible_for_score_parameter_fit": False,
            "eligible_for_measurement_parameter_fit": False,
        }


def _entry_rows(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) != PROMPT_COUNT:
        raise ValueError("D2-A requires exactly twelve MCC fingerprints")
    rows = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        prompt_id = str(entry.get("prompt_id", ""))
        prompt_sha = str((entry.get("metadata") or {}).get("prompt_sha256", ""))
        if not prompt_id or prompt_id in seen or len(prompt_sha) != 64:
            raise ValueError("Invalid MCC fingerprint entry")
        seen.add(prompt_id)
        rows.append({"prompt_index": index, "prompt_id": prompt_id, "prompt_sha256": prompt_sha})
    return rows


def _derive_uint32(domain: str, root_seed: int, used: set[int], *parts: object) -> tuple[int, int, str]:
    counter = 0
    while True:
        value = "\0".join([domain, str(root_seed), *(str(part) for part in parts), str(counter)])
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little")
        if seed not in used:
            used.add(seed)
            return seed, counter, digest.hex()
        counter += 1


def _prompt_order(entries: Sequence[Mapping[str, Any]], root_seed: int, bank_id: str, replicate_id: int) -> list[int]:
    return sorted(
        range(len(entries)),
        key=lambda index: hashlib.sha256(
            f"h8-d2a-prompt-order-v1\0{root_seed}\0{bank_id}\0{replicate_id}\0{entries[index]['prompt_id']}".encode()
        ).digest(),
    )


def build_generation_schedules(
    entries: Sequence[Mapping[str, Any]],
    attack_instances: Sequence[Mapping[str, Any]],
    *,
    reference_root_seed: int,
    intact_target_root_seed: int,
    attack_root_seed: int,
    forbidden_generation_seeds: Iterable[int] = (),
) -> dict[str, list[DevelopmentGenerationRequest]]:
    prompts = _entry_rows(entries)
    attacks = [dict(row) for row in attack_instances]
    if len(attacks) != 8 or Counter(row.get("family") for row in attacks) != Counter({family: 2 for family in ATTACK_FAMILIES}):
        raise ValueError("D2-A requires exactly two fresh attack instances in each of four families")
    if len({str(row.get("attack_instance_id")) for row in attacks}) != 8:
        raise ValueError("Fresh attack instance IDs must be unique")
    used = {int(seed) for seed in forbidden_generation_seeds if 0 <= int(seed) <= UINT32_MAX}
    schedules: dict[str, list[DevelopmentGenerationRequest]] = {
        "reference": [],
        "intact_target": [],
        "attack": [],
    }
    global_position = 0

    def append_bank(
        output_name: str,
        bank_id: str,
        role: str,
        rounds: int,
        root_seed: int,
        family: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        nonlocal global_position
        for replicate_id in range(rounds):
            for prompt_position, prompt_index in enumerate(_prompt_order(prompts, root_seed, bank_id, replicate_id)):
                prompt = prompts[prompt_index]
                seed, counter, digest = _derive_uint32(
                    "h8-d2a-development-response-seed-v1",
                    root_seed,
                    used,
                    bank_id,
                    replicate_id,
                    prompt["prompt_id"],
                )
                rows = schedules[output_name]
                rows.append(
                    DevelopmentGenerationRequest(
                        schedule_position=global_position,
                        bank_position=len(rows),
                        replicate_id=replicate_id,
                        prompt_position_in_round=prompt_position,
                        prompt_index=prompt_index,
                        prompt_id=prompt["prompt_id"],
                        prompt_sha256=prompt["prompt_sha256"],
                        generation_seed=seed,
                        derivation_counter=counter,
                        seed_digest_sha256=digest,
                        data_role=role,
                        bank_id=bank_id,
                        attack_family=family,
                        attack_instance_id=instance_id,
                    )
                )
                global_position += 1

    append_bank("reference", "development_reference_intact", REFERENCE_ROLE, 60, reference_root_seed)
    append_bank("intact_target", "development_target_intact", INTACT_TARGET_ROLE, 100, intact_target_root_seed)
    for attack in attacks:
        instance_id = str(attack["attack_instance_id"])
        append_bank(
            "attack",
            instance_id,
            ATTACK_ROLE,
            20,
            attack_root_seed,
            str(attack["family"]),
            instance_id,
        )
    validate_generation_schedules(schedules)
    return schedules


def validate_generation_schedules(schedules: Mapping[str, Sequence[DevelopmentGenerationRequest]]) -> None:
    expected = {"reference": 720, "intact_target": 1200, "attack": 1920}
    if set(schedules) != set(expected) or any(len(schedules[key]) != count for key, count in expected.items()):
        raise ValueError("D2-A proposed generation counts mismatch")
    rows = [row for key in ("reference", "intact_target", "attack") for row in schedules[key]]
    if [row.schedule_position for row in rows] != list(range(3840)):
        raise ValueError("D2-A schedule positions must be contiguous")
    if len({row.response_id for row in rows}) != 3840 or len({row.generation_seed for row in rows}) != 3840:
        raise ValueError("D2-A response IDs/generation seeds must be unique")
    if any(row.data_role not in ALLOWED_DEVELOPMENT_ROLES for row in rows):
        raise ValueError("D2-A schedule contains a forbidden role")
    for key, count_per_prompt in (("reference", 60), ("intact_target", 100)):
        counts = Counter(row.prompt_id for row in schedules[key])
        if len(counts) != PROMPT_COUNT or set(counts.values()) != {count_per_prompt}:
            raise ValueError(f"D2-A {key} per-prompt count mismatch")
    attack_counts = Counter((row.attack_instance_id, row.prompt_id) for row in schedules["attack"])
    if len(attack_counts) != 8 * PROMPT_COUNT or set(attack_counts.values()) != {20}:
        raise ValueError("D2-A attack endpoint/prompt count mismatch")


def generation_manifest_payload(
    rows: Sequence[DevelopmentGenerationRequest],
    *,
    manifest_kind: str,
    frozen_identity: Mapping[str, Any],
    repository_seed_source_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": D2A_SCHEMA_VERSION,
        "manifest_kind": manifest_kind,
        "status": "proposed_not_authorized",
        "sampling_authorized": False,
        "generation_started": False,
        "new_model_responses": 0,
        "batch_size": 1,
        "response_count_if_future_authorized": len(rows),
        "unique_generation_seed_count": len({row.generation_seed for row in rows}),
        "repository_seed_source_count": repository_seed_source_count,
        "frozen_identity": dict(frozen_identity),
        "eligibility": {
            "future_formal_reference": False,
            "future_final_heldout": False,
            "future_final_confirmation": False,
            "score_or_measurement_fit": False,
        },
        "schedule": [row.as_dict() for row in rows],
    }


def _hash_rank(root_seed: int, domain: str, *parts: object) -> bytes:
    return hashlib.sha256("\0".join([domain, str(root_seed), *(str(x) for x in parts)]).encode()).digest()


def make_nested_subset_payload(
    schedules: Mapping[str, Sequence[DevelopmentGenerationRequest]],
    *,
    subset_root_seed: int,
) -> dict[str, Any]:
    validate_generation_schedules(schedules)
    reference_by_prompt: dict[str, list[DevelopmentGenerationRequest]] = defaultdict(list)
    for row in schedules["reference"]:
        reference_by_prompt[row.prompt_id].append(row)
    reference_membership: dict[str, Any] = {}
    for prompt_id, rows in sorted(reference_by_prompt.items()):
        r60 = sorted(row.response_id for row in rows)
        r40 = sorted(
            r60,
            key=lambda response_id: _hash_rank(subset_root_seed, "h8-d2a-r40-v1", prompt_id, response_id),
        )[:40]
        reference_membership[prompt_id] = {
            "R60": r60,
            "R40": sorted(r40),
            "nested": set(r40).issubset(r60),
        }
    target_units: list[dict[str, Any]] = []

    def add_target_unit(unit_id: str, rows: Sequence[DevelopmentGenerationRequest], role: str, family: str | None) -> None:
        by_prompt: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            by_prompt[row.prompt_id].append(row.response_id)
        membership: dict[str, Any] = {}
        for prompt_id, response_ids in sorted(by_prompt.items()):
            q20 = sorted(response_ids)
            if len(q20) != 20:
                raise ValueError(f"Target unit {unit_id} does not have 20 responses for {prompt_id}")
            q10 = sorted(
                q20,
                key=lambda response_id: _hash_rank(
                    subset_root_seed, "h8-d2a-q10-v1", unit_id, prompt_id, response_id
                ),
            )[:10]
            membership[prompt_id] = {"Q20": q20, "Q10": sorted(q10), "nested": set(q10).issubset(q20)}
        target_units.append(
            {
                "evaluation_unit_id": unit_id,
                "data_role": role,
                "attack_family": family,
                "members_by_prompt": membership,
            }
        )

    intact_rows = list(schedules["intact_target"])
    for block_id in range(5):
        add_target_unit(
            f"development_intact_block_{block_id:02d}",
            [row for row in intact_rows if row.replicate_id // 20 == block_id],
            INTACT_TARGET_ROLE,
            None,
        )
    attack_rows = list(schedules["attack"])
    for instance_id in sorted({str(row.attack_instance_id) for row in attack_rows}):
        unit_rows = [row for row in attack_rows if row.attack_instance_id == instance_id]
        add_target_unit(instance_id, unit_rows, ATTACK_ROLE, str(unit_rows[0].attack_family))
    payload = {
        "schema_version": D2A_SCHEMA_VERSION,
        "subset_root_seed_uint64": subset_root_seed,
        "membership_frozen_before_generation": True,
        "single_banks_serve_all_four_structures": True,
        "reference_membership_by_prompt": reference_membership,
        "target_evaluation_units": target_units,
        "intact_evaluation_unit_count": 5,
        "attack_evaluation_unit_count": 8,
        "nested_invariants": {"R40_subset_R60": True, "Q10_subset_Q20": True},
    }
    validate_nested_subset_payload(payload)
    return payload


def validate_nested_subset_payload(payload: Mapping[str, Any]) -> None:
    refs = payload.get("reference_membership_by_prompt")
    units = payload.get("target_evaluation_units")
    if not isinstance(refs, dict) or len(refs) != PROMPT_COUNT or not isinstance(units, list) or len(units) != 13:
        raise ValueError("D2-A nested subset prompt/unit counts mismatch")
    for row in refs.values():
        r40, r60 = row["R40"], row["R60"]
        if len(r40) != 40 or len(r60) != 60 or len(set(r60)) != 60 or not set(r40).issubset(r60):
            raise ValueError("R40 must be a strict membership subset of R60")
    for unit in units:
        members = unit.get("members_by_prompt")
        if not isinstance(members, dict) or len(members) != PROMPT_COUNT:
            raise ValueError("Target unit must cover all prompts")
        for row in members.values():
            q10, q20 = row["Q10"], row["Q20"]
            if len(q10) != 10 or len(q20) != 20 or len(set(q20)) != 20 or not set(q10).issubset(q20):
                raise ValueError("Q10 must be a membership subset of Q20")


def make_permutation_seed_payload(
    nested_subset_payload: Mapping[str, Any],
    *,
    permutation_root_seed: int,
) -> dict[str, Any]:
    validate_nested_subset_payload(nested_subset_payload)
    prompt_ids = sorted(nested_subset_payload["reference_membership_by_prompt"])
    streams: list[dict[str, Any]] = []
    all_seed_hash = hashlib.sha256()
    seen: set[int] = set()
    for unit in nested_subset_payload["target_evaluation_units"]:
        unit_id = str(unit["evaluation_unit_id"])
        for structure in SAMPLE_STRUCTURES:
            stream_seed = _uint64("h8-d2a-global-permutation-stream-v1", permutation_root_seed, unit_id, structure)
            stream_hash = hashlib.sha256()
            for permutation_id in range(DEVELOPMENT_PERMUTATIONS):
                for prompt_id in prompt_ids:
                    seed = _uint64(
                        "h8-d2a-within-prompt-permutation-v1",
                        stream_seed,
                        permutation_id,
                        prompt_id,
                    )
                    if seed in seen:
                        raise ValueError("D2-A permutation seed collision")
                    seen.add(seed)
                    encoded = seed.to_bytes(8, "little")
                    stream_hash.update(encoded)
                    all_seed_hash.update(encoded)
            streams.append(
                {
                    "evaluation_unit_id": unit_id,
                    "sample_structure": structure,
                    "stream_seed_uint64": stream_seed,
                    "permutations": DEVELOPMENT_PERMUTATIONS,
                    "prompt_count": PROMPT_COUNT,
                    "derived_prompt_permutation_seed_set_sha256": stream_hash.hexdigest(),
                }
            )
    return {
        "schema_version": D2A_SCHEMA_VERSION,
        "permutation_root_seed_uint64": permutation_root_seed,
        "algorithm": "domain_separated_sha256_uint64_to_numpy_pcg64",
        "within_prompt_only": True,
        "group_sizes_preserved": True,
        "stream_count": len(streams),
        "derived_seed_count": len(seen),
        "derived_seed_unique": True,
        "derived_seed_set_sha256": all_seed_hash.hexdigest(),
        "streams": streams,
    }


def save_canonical_envelope(path: str | Path, artifact_type: str, payload: Mapping[str, Any]) -> dict[str, str]:
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": artifact_type,
        "schema_version": D2A_SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return {"payload_sha256": payload_sha, "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def load_canonical_envelope(
    path: str | Path,
    *,
    expected_artifact_type: str,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    target = Path(path)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_file_sha256:
        raise ValueError("D2-A artifact file SHA256 mismatch")
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if (
        envelope.get("artifact_type") != expected_artifact_type
        or envelope.get("schema_version") != D2A_SCHEMA_VERSION
        or not isinstance(payload, dict)
    ):
        raise ValueError("D2-A artifact type/schema mismatch")
    actual = canonical_sha256(payload)
    if actual != envelope.get("payload_sha256") or actual != expected_payload_sha256:
        raise ValueError("D2-A artifact payload SHA256 mismatch")
    return payload
