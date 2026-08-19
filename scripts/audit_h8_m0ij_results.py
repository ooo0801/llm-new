from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "h8_qwen32b_mmd_precalibration" / "m0ij"
ARCHIVE = (
    ROOT
    / "reproducibility"
    / "h8_qwen32b_mmd_precalibration_20260818"
    / "mmd_measurement_frozen_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    final = json.loads((RESULT / "H8_MMD_PRECALIBRATION_FINAL_REPORT.json").read_text(encoding="utf-8"))
    numerical = json.loads((RESULT / "M0I_NUMERICAL_VALIDATION_REPORT.json").read_text(encoding="utf-8"))
    permutation = json.loads((RESULT / "M0J_PERMUTATION_SANITY_REPORT.json").read_text(encoding="utf-8"))
    pseudo_rows = jsonl(RESULT / "intact_pseudo_mmd_trials.jsonl")
    permutation_rows = jsonl(RESULT / "permutation_sanity_trials.jsonl")
    manifest_path = ARCHIVE / "MMD_FROZEN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_archive_hashes = {
        logical: {"actual": sha256(ARCHIVE / logical), "expected": expected}
        for logical, expected in manifest["files"].items()
        if not (ARCHIVE / logical).is_file() or sha256(ARCHIVE / logical) != expected
    }
    sidecar_value = (manifest_path.with_suffix(".json.sha256")).read_text(encoding="ascii").split()[0]
    prompt_structures = Counter((row["prompt_id"], row["structure"]) for row in pseudo_rows)
    negative_by_prompt_structure = {
        f"{prompt}:{structure}": sum(row["unbiased_mmd2"] < 0 for row in pseudo_rows if row["prompt_id"] == prompt and row["structure"] == structure) / count
        for (prompt, structure), count in sorted(prompt_structures.items())
    }
    gates = {
        "final_pass": final["status"] == "PASS",
        "measurement_frozen": final["measurement_layer_status"] == "frozen" == manifest["measurement_layer_status"],
        "detector_not_frozen": final["detector_status"] == "not_frozen" == manifest["detector_status"],
        "no_new_or_forbidden_responses": all(
            final[key] == 0
            for key in ("new_model_responses", "formal_reference_responses", "heldout_responses", "attack_responses")
        ),
        "sample_size_not_selected": final["sample_size_selection_performed"] is False,
        "pseudo_count_9600": len(pseudo_rows) == numerical["trial_count"] == 9600,
        "each_prompt_structure_200": len(prompt_structures) == 48 and set(prompt_structures.values()) == {200},
        "numerical_pass_finite": numerical["status"] == "PASS" and numerical["nan_or_inf_count"] == 0,
        "degenerate_invariants": numerical["degenerate_invariant_failures"] == [],
        "permutation_count_48": len(permutation_rows) == permutation["representative_trial_count"] == 48,
        "permutation_pass": permutation["status"] == "PASS" and permutation["failures"] == [],
        "permutation_reproducible": all(row["fixed_seed_reproducible"] for row in permutation_rows),
        "permutation_group_sizes": all(row["group_size_correct"] for row in permutation_rows),
        "permutation_observed_once": all(row["observed_evaluation_count"] == 1 for row in permutation_rows),
        "permutation_p_valid": all(0.0 < row["p_value"] <= 1.0 for row in permutation_rows),
        "candidate_hashes_unchanged": permutation["candidate_artifact_hashes_unchanged"] is True,
        "regressions_pass": all(value["pass"] for value in permutation["regression_gates"].values()),
        "archive_hashes_valid": not bad_archive_hashes,
        "manifest_sidecar_valid": sidecar_value == sha256(manifest_path),
    }
    output = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mmd_implementation_commit": final["mmd_implementation_commit"],
        "pseudo_trial_count": len(pseudo_rows),
        "permutation_trial_count": len(permutation_rows),
        "permutation_statistic_count": len(permutation_rows) * permutation["permutations_per_trial"],
        "permutation_p_min": min(row["p_value"] for row in permutation_rows),
        "permutation_p_max": max(row["p_value"] for row in permutation_rows),
        "negative_fraction_by_prompt_structure": negative_by_prompt_structure,
        "structured_stability_note": final["structured_stability_preexisting_observation"],
        "frozen_manifest_sha256": sha256(manifest_path),
        "frozen_file_count": len(manifest["files"]),
        "bad_archive_hashes": bad_archive_hashes,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
