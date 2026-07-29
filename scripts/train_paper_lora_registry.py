from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT

from llm_integrity.config import load_config
from llm_integrity.paper_finetuning import train_lora_manifest_variant


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            rows.append(row)

    return rows


def load_registry(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(
            f"Adapter registry must be a JSON object: {path}"
        )

    return {
        str(key): str(value)
        for key, value in payload.items()
    }


def save_registry(
    path: Path,
    registry: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_finetuning_variants(
    rows: list[dict[str, Any]],
    split: str,
    max_variants: int | None,
    variant_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("family", "")).lower() == "finetuning"
        and str(row.get("split", "")).lower() == split.lower()
    ]

    if variant_ids:
        by_id = {
            str(row.get("variant_id")): row for row in selected
        }
        missing = [
            variant_id
            for variant_id in variant_ids
            if variant_id not in by_id
        ]
        if missing:
            raise ValueError(
                f"Requested finetuning variants are missing: {missing}"
            )
        selected = [by_id[variant_id] for variant_id in variant_ids]
    else:
        selected.sort(
            key=lambda row: str(row.get("variant_id", ""))
        )

    if max_variants is not None:
        selected = selected[:max_variants]

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train isolated LoRA attack variants and build an "
            "adapter registry for paper-aligned macro sensitivity."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_aligned_qwen_7b.yaml",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "results/paper_aligned_qwen_7b/"
            "manifests_truthful_quant/"
            "attack_manifest_validation.jsonl"
        ),
    )
    parser.add_argument(
        "--split",
        default="validation",
    )
    parser.add_argument(
        "--data",
        default="data/attack_train_lora.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "results/paper_aligned_qwen_7b/"
            "finetuning_adapters"
        ),
    )
    parser.add_argument(
        "--registry-output",
        default=(
            "results/paper_aligned_qwen_7b/"
            "finetuning_adapters/"
            "adapter_registry_validation.json"
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        default=None,
        help=(
            "Train only the requested manifest variant. Repeat the "
            "option to preserve an explicit deterministic order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain adapters even if adapter_config.json exists.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    manifest_path = resolve_path(args.manifest)
    data_path = resolve_path(args.data)
    output_root = resolve_path(args.output_root)
    registry_path = resolve_path(args.registry_output)

    config = load_config(str(config_path))
    model_config = config["model"]

    manifest_rows = read_jsonl(manifest_path)
    variants = select_finetuning_variants(
        manifest_rows,
        split=args.split,
        max_variants=args.max_variants,
        variant_ids=args.variant_id,
    )

    if not variants:
        raise ValueError(
            f"No finetuning variants found for split {args.split!r}"
        )

    print(
        json.dumps(
            {
                "stage": "lora_training_plan",
                "config": str(config_path),
                "manifest": str(manifest_path),
                "split": args.split,
                "training_data": str(data_path),
                "variants": len(variants),
                "output_root": str(output_root),
                "registry_output": str(registry_path),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )

    for index, variant in enumerate(variants, start=1):
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(variants),
                    "variant_id": variant["variant_id"],
                    "configuration": variant["configuration"],
                },
                ensure_ascii=False,
            )
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "dry_run_complete",
                    "adapters_trained": 0,
                },
                ensure_ascii=False,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    reports: list[dict[str, Any]] = []

    for index, variant in enumerate(variants, start=1):
        variant_id = str(variant["variant_id"])
        expected_adapter = output_root / variant_id
        adapter_config = expected_adapter / "adapter_config.json"

        print(
            json.dumps(
                {
                    "stage": "variant_start",
                    "index": index,
                    "total": len(variants),
                    "variant_id": variant_id,
                },
                ensure_ascii=False,
            )
        )

        if adapter_config.exists() and not args.force:
            registry[variant_id] = str(expected_adapter)
            save_registry(registry_path, registry)

            print(
                json.dumps(
                    {
                        "stage": "variant_reused",
                        "variant_id": variant_id,
                        "adapter_path": str(expected_adapter),
                    },
                    ensure_ascii=False,
                )
            )
            continue

        report = train_lora_manifest_variant(
            model_config=model_config,
            variant=variant,
            data_path=data_path,
            output_root=output_root,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
        )

        report_payload = asdict(report)
        reports.append(report_payload)

        adapter_path = Path(str(report.adapter_path))
        registry[variant_id] = str(adapter_path)

        # 每训练完一个适配器立即保存注册表。
        # 即使后续训练中断，前面已经完成的结果仍然保留。
        save_registry(registry_path, registry)

        print(
            json.dumps(
                {
                    "stage": "variant_complete",
                    "variant_id": variant_id,
                    "adapter_path": str(adapter_path),
                    "training_loss": report.training_loss,
                    "completed_steps": report.completed_steps,
                    "trainable_parameters": (
                        report.trainable_parameters
                    ),
                    "trainable_ratio": report.trainable_ratio,
                },
                ensure_ascii=False,
            )
        )

    report_path = output_root / (
        f"training_reports_{args.split}.json"
    )
    existing_reports: list[dict[str, Any]] = []
    if report_path.exists():
        existing_payload = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        if not isinstance(existing_payload, list):
            raise ValueError(
                f"Training report must be a JSON list: {report_path}"
            )
        existing_reports = [
            dict(item) for item in existing_payload
        ]
    reports_by_id = {
        str(item["variant_id"]): item
        for item in existing_reports
    }
    reports_by_id.update(
        {
            str(item["variant_id"]): item
            for item in reports
        }
    )
    merged_reports = [
        reports_by_id[variant_id]
        for variant_id in sorted(reports_by_id)
    ]
    temporary_report = report_path.with_suffix(
        report_path.suffix + ".tmp"
    )
    temporary_report.write_text(
        json.dumps(
            merged_reports,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)

    print(
        json.dumps(
            {
                "stage": "all_complete",
                "variants": len(variants),
                "registry_entries": len(registry),
                "registry_output": str(registry_path),
                "training_report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
