from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.paper_macro_sampling import sample_macro_sensitivity


PRIMARY_DISTANCE_TO_METRIC = {
    "squared_l2_logits": "macro_l2_raw",
    "squared_l2_probabilities": "macro_probability_l2",
    "js_probabilities": "macro_js",
}


def resolve_primary_metric(
    config: Mapping[str, Any],
    command_line_value: str | None,
) -> str:
    """Resolve the CLI override or the proposal-aligned config distance."""
    if command_line_value is not None:
        return command_line_value

    configured_distance = str(
        config.get("sensitivity", {})
        .get("macro", {})
        .get("primary_distance", "")
    ).strip()

    try:
        return PRIMARY_DISTANCE_TO_METRIC[configured_distance]
    except KeyError as exc:
        supported = ", ".join(sorted(PRIMARY_DISTANCE_TO_METRIC))
        raise ValueError(
            "Unsupported sensitivity.macro.primary_distance "
            f"{configured_distance!r}; expected one of: {supported}"
        ) from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = project_path(path)
    rows: list[dict[str, Any]] = []

    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(
                    f"{source}:{line_number} is not a JSON object"
                )
            rows.append(row)

    return rows


def read_adapter_registry(
    path: str | Path | None,
) -> dict[str, str]:
    if path is None:
        return {}

    source = project_path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise TypeError("LoRA adapter registry must be a JSON object")

    return {
        str(variant_id): str(adapter_path)
        for variant_id, adapter_path in payload.items()
    }


def family_of(manifest: Mapping[str, Any]) -> str:
    configuration = manifest.get("configuration", {})

    return str(
        manifest.get(
            "family",
            configuration.get("type", ""),
        )
    ).lower()


def parse_families(value: str | None) -> set[str] | None:
    if value is None:
        return None

    families = {
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    }

    return families or None


def filter_manifests(
    manifests: list[dict[str, Any]],
    *,
    split: str,
    families: set[str] | None,
    max_variants: int | None,
) -> list[dict[str, Any]]:
    selected = [
        manifest
        for manifest in manifests
        if str(manifest.get("split", "")).lower() == split.lower()
    ]

    if families is not None:
        selected = [
            manifest
            for manifest in selected
            if family_of(manifest) in families
        ]

    selected.sort(
        key=lambda manifest: (
            family_of(manifest),
            str(manifest.get("variant_id", "")),
        )
    )

    if max_variants is not None:
        selected = selected[:max_variants]

    return selected


def summarize_plan(
    prompts: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    adapters: Mapping[str, str],
    *,
    split: str,
) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    missing_adapters: list[str] = []

    for manifest in manifests:
        family = family_of(manifest)
        family_counts[family] = family_counts.get(family, 0) + 1

        if family == "finetuning":
            variant_id = str(manifest.get("variant_id", "unknown"))
            if variant_id not in adapters:
                missing_adapters.append(variant_id)

    return {
        "split": split,
        "prompts": len(prompts),
        "variants": len(manifests),
        "observations": len(prompts) * len(manifests),
        "family_counts": family_counts,
        "registered_lora_adapters": len(adapters),
        "missing_lora_adapters": missing_adapters,
        "ready_for_full_run": len(missing_adapters) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate application-aligned macro sensitivity over "
            "isolated sampled model modifications"
        )
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b.yaml"),
    )
    parser.add_argument(
        "--prompts",
        default="data/candidate.jsonl",
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_aligned_qwen_7b/macro",
    )
    parser.add_argument(
        "--adapter-registry",
        default=None,
    )
    parser.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-separated family names. "
            "Example: unstructured_pruning,quantization"
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--primary-metric",
        choices=[
            "macro_l2_raw",
            "macro_l2_per_dimension",
            "macro_probability_l2",
            "macro_js",
        ],
        default=None,
        help=(
            "Override sensitivity.macro.primary_distance from the config. "
            "When omitted, the configured primary distance is used."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the plan without loading any model",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    primary_metric = resolve_primary_metric(
        config,
        args.primary_metric,
    )
    prompts = read_jsonl(args.prompts)
    manifests = read_jsonl(args.manifest)
    adapters = read_adapter_registry(args.adapter_registry)

    if args.max_prompts is not None:
        if args.max_prompts <= 0:
            raise ValueError("--max-prompts must be positive")
        prompts = prompts[: args.max_prompts]

    families = parse_families(args.families)
    manifests = filter_manifests(
        manifests,
        split=args.split,
        families=families,
        max_variants=args.max_variants,
    )

    plan = summarize_plan(
        prompts,
        manifests,
        adapters,
        split=args.split,
    )

    print(
        json.dumps(
            {
                "stage": "macro_sampling_plan",
                "primary_metric": primary_metric,
                **plan,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not prompts:
        raise ValueError("No prompts were selected")

    if not manifests:
        raise ValueError("No attack variants were selected")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "dry_run_complete",
                    "models_loaded": 0,
                    "attacks_executed": 0,
                },
                ensure_ascii=False,
            )
        )
        return

    if plan["missing_lora_adapters"]:
        raise RuntimeError(
            "The selected manifests contain finetuning variants without "
            "registered LoRA adapters. Train those adapters first or exclude "
            "the finetuning family during a debug run."
        )

    report = sample_macro_sensitivity(
        base_model_config=config["model"],
        prompt_rows=prompts,
        manifests=manifests,
        output_dir=project_path(args.output_dir),
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        system_prompt=config.get("generation", {}).get("system_prompt"),
        primary_metric=primary_metric,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        adapter_paths=adapters,
    )

    print(
        json.dumps(
            {
                "stage": "macro_sampling_complete",
                **asdict(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
