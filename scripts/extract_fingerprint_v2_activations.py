from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from _bootstrap import ROOT, project_path

from llm_integrity.activations import TransformerActivationProfiler
from llm_integrity.config import load_config
from llm_integrity.global_coverage import component_counts, stable_prompt_components, union_components
from llm_integrity.io import read_jsonl, write_json, write_jsonl
from llm_integrity.modeling import load_model, next_token_logits


SPLIT_KEYS = {
    "build": "calibration_build_manifest",
    "audit": "calibration_audit_manifest",
    "candidate": "strict16_manifest",
}


def hook_count(model) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def load_checkpoint(path: Path, *, prompt_id: str, repeat: int, prompt_sha256: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(row.get("id")) != prompt_id
        or int(row.get("profile_repeat", -1)) != repeat
        or str(row.get("prompt_sha256")) != prompt_sha256
    ):
        raise ValueError(f"Stale activation checkpoint: {path}")
    return row


def split_audit(
    rows: list[dict],
    manifest: list[dict],
    *,
    repetitions: int,
    stable_frequency: float,
    minimum_jaccard: float,
) -> dict:
    stable = stable_prompt_components(
        rows,
        repetitions=repetitions,
        stable_frequency=stable_frequency,
    )
    expected_ids = [str(row["id"]) for row in manifest]
    if set(stable) != set(expected_ids):
        raise ValueError("Activation profiles and manifest prompt IDs differ")
    grouped: dict[str, list[set[str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(set(row["components"]))
    per_prompt = {}
    for prompt_id in expected_ids:
        values = grouped[prompt_id]
        pairwise = [
            jaccard(values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        per_prompt[prompt_id] = min(pairwise) if pairwise else 1.0
    component_sizes = [len(values) for values in stable.values()]
    universe = union_components(stable)
    minimum = min(per_prompt.values())
    return {
        "prompts": len(manifest),
        "repetitions": repetitions,
        "rows": len(rows),
        "minimum_jaccard": minimum,
        "mean_jaccard": mean(per_prompt.values()),
        "per_prompt_jaccard": per_prompt,
        "stable_component_count_min": min(component_sizes),
        "stable_component_count_max": max(component_sizes),
        "stable_component_count_mean": mean(component_sizes),
        "stable_union_components": len(universe),
        "stable_union_counts_by_type": component_counts(universe),
        "passed": minimum >= minimum_jaccard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract resumable V2 calibration and candidate activations")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_KEYS),
        default=list(SPLIT_KEYS),
    )
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["activations"]
    repetitions = int(settings["repetitions"])
    stable_frequency = float(settings["stable_component_frequency"])
    minimum_jaccard = float(settings["minimum_jaccard"])
    output_dir = project_path(config["output_dir"])
    profiler = TransformerActivationProfiler(
        attention_entropy_fraction=float(settings["attention_entropy_fraction"]),
        ffn_quantile=float(settings["ffn_quantile"]),
        residual_threshold=float(settings["residual_threshold"]),
        max_ffn_units_per_layer=int(settings["max_ffn_units_per_layer"]),
    )
    manifests = {
        split: read_jsonl(project_path(config["data"][SPLIT_KEYS[split]]))
        for split in args.splits
    }
    if args.max_prompts is not None:
        manifests = {
            split: rows[: args.max_prompts]
            for split, rows in manifests.items()
        }

    bundle = load_model(config["model"])
    initial_hooks = hook_count(bundle.model)
    invariance = None
    completed: dict[str, list[dict]] = {}
    try:
        first_split = args.splits[0]
        first_row = manifests[first_split][0]
        before = next_token_logits(
            bundle,
            [str(first_row["prompt"])],
            int(settings["max_length"]),
            config["generation"].get("system_prompt"),
        )
        for split in args.splits:
            manifest = manifests[split]
            checkpoint_dir = output_dir / "activation_checkpoints" / split
            rows: list[dict] = []
            for prompt_index, row in enumerate(manifest, start=1):
                prompt_id = str(row["id"])
                prompt_sha256 = str(row["prompt_sha256"])
                for repeat in range(repetitions):
                    checkpoint = checkpoint_dir / f"{prompt_id}__r{repeat}.json"
                    if checkpoint.exists():
                        profile = load_checkpoint(
                            checkpoint,
                            prompt_id=prompt_id,
                            repeat=repeat,
                            prompt_sha256=prompt_sha256,
                        )
                        status = "reused"
                    else:
                        profile = profiler.profile(
                            bundle,
                            prompt_id,
                            str(row["prompt"]),
                            int(settings["max_length"]),
                            config["generation"].get("system_prompt"),
                        ).as_dict()
                        if hook_count(bundle.model) != initial_hooks:
                            raise RuntimeError("Activation profiler leaked forward hooks")
                        profile.update(
                            {
                                "profile_repeat": repeat,
                                "category": row.get("category"),
                                "language": row.get("language"),
                                "length_bucket": row.get("length_bucket"),
                                "calibration_split": row.get("calibration_split", split),
                                "prompt_sha256": prompt_sha256,
                            }
                        )
                        write_json(checkpoint, profile)
                        status = "computed"
                    rows.append(profile)
                    print(
                        json.dumps(
                            {
                                "stage": "v2_activation",
                                "split": split,
                                "status": status,
                                "repeat": repeat,
                                "index": prompt_index,
                                "total": len(manifest),
                                "id": prompt_id,
                                "components": len(profile["components"]),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            write_jsonl(output_dir / "activations" / f"{split}_profiles.jsonl", rows)
            completed[split] = rows
        after = next_token_logits(
            bundle,
            [str(first_row["prompt"])],
            int(settings["max_length"]),
            config["generation"].get("system_prompt"),
        )
        import torch

        invariance = {
            "allclose": bool(torch.allclose(before, after, rtol=0.0, atol=0.0)),
            "max_abs_difference": float((before - after).abs().max().item()),
            "same_rendered_system_prompt": True,
        }
        if not invariance["allclose"]:
            raise RuntimeError("V2 profiling changed reference model logits")
    finally:
        bundle.close()

    audits = {
        split: split_audit(
            completed[split],
            manifests[split],
            repetitions=repetitions,
            stable_frequency=stable_frequency,
            minimum_jaccard=minimum_jaccard,
        )
        for split in args.splits
    }
    combined = {
        "schema_version": "fingerprint_v2_activation_audit_1.0",
        "component_id_schema": "activation_observable_v1",
        "splits": audits,
        "logit_invariance": invariance,
        "passed": bool(
            invariance
            and invariance["allclose"]
            and all(value["passed"] for value in audits.values())
        ),
    }
    write_json(output_dir / "activation_audit.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
