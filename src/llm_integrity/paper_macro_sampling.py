from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .modeling import load_model
from .paper_macro import (
    compute_macro_distances,
    summarize_stratified_macro,
)
from .paper_outputs import differentiable_next_token_logits
from .paper_variant_executor import load_manifest_variant


@dataclass(frozen=True)
class PromptMacroRecord:
    """Macro-sensitivity observation for one prompt and one variant."""

    prompt_id: str
    category: str
    variant_id: str
    family: str
    split: str
    seed: int
    family_weight: float
    macro_l2_raw: float
    macro_l2_per_dimension: float
    macro_probability_l2: float
    macro_js: float
    output_dimension: int
    realized_method: str
    execution_mode: str


@dataclass(frozen=True)
class MacroSamplingReport:
    """Traceable report for one complete macro-sensitivity sampling run."""

    split: str
    prompt_count: int
    variant_count: int
    observation_count: int
    reference_model: str
    primary_metric: str
    family_weights: dict[str, float]
    records_path: str
    summary_path: str


def _manifest_family(manifest: Mapping[str, Any]) -> str:
    configuration = manifest.get("configuration", {})
    return str(
        manifest.get(
            "family",
            configuration.get("type", ""),
        )
    ).lower()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
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


def _write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _prompt_id(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get(
            "prompt_id",
            row.get(
                "id",
                f"prompt_{index:06d}",
            ),
        )
    )


def _validate_prompts(
    prompt_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not prompt_rows:
        raise ValueError("No prompt rows were provided")

    seen: set[str] = set()

    for index, row in enumerate(prompt_rows):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(
                f"Prompt row {index} has no non-empty 'prompt' field"
            )

        identifier = _prompt_id(row, index)
        if identifier in seen:
            raise ValueError(f"Duplicate prompt id: {identifier}")
        seen.add(identifier)


def _validate_manifests(
    manifests: Sequence[Mapping[str, Any]],
    split: str,
) -> None:
    if not manifests:
        raise ValueError("No attack manifests were provided")

    variant_ids: set[str] = set()

    for index, manifest in enumerate(manifests):
        variant_id = str(
            manifest.get(
                "variant_id",
                f"variant_{index:06d}",
            )
        )

        if variant_id in variant_ids:
            raise ValueError(f"Duplicate variant id: {variant_id}")
        variant_ids.add(variant_id)

        manifest_split = str(manifest.get("split", "")).lower()
        if manifest_split != split.lower():
            raise ValueError(
                f"Variant {variant_id!r} belongs to split "
                f"{manifest_split!r}, not requested split {split!r}"
            )

        family = _manifest_family(manifest)
        if not family:
            raise ValueError(
                f"Variant {variant_id!r} has no attack family"
            )


def _family_weights(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}

    for manifest in manifests:
        family = _manifest_family(manifest)
        weight = float(manifest.get("weight", 1.0))
        values.setdefault(family, []).append(weight)

    weights = {
        family: sum(family_values) / len(family_values)
        for family, family_values in values.items()
    }

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("The sum of attack-family weights must be positive")

    return {
        family: weight / total
        for family, weight in weights.items()
    }


def _collect_logits(
    bundle,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    system_prompt: str | None,
):
    """
    Collect full next-token logit vectors on CPU.

    Logits remain full-vocabulary vectors. No top-k truncation is applied.
    """

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks: list[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])

        with torch.inference_mode():
            logits = differentiable_next_token_logits(
                bundle,
                batch,
                max_length=max_length,
                system_prompt=system_prompt,
                cast_float32=True,
            )

        chunks.append(
            logits.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)


def _resolve_adapter_path(
    manifest: Mapping[str, Any],
    adapter_paths: Mapping[str, str | Path] | None,
) -> str | Path | None:
    if adapter_paths is None:
        return None

    variant_id = str(manifest.get("variant_id", "unknown"))
    return adapter_paths.get(variant_id)


def sample_macro_sensitivity(
    *,
    base_model_config: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    split: str,
    batch_size: int = 1,
    max_length: int = 512,
    system_prompt: str | None = None,
    primary_metric: str = "macro_l2_raw",
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    adapter_paths: Mapping[str, str | Path] | None = None,
) -> MacroSamplingReport:
    """
    Estimate macro sensitivity over a sampled modification distribution.

    For every prompt x and sampled modification theta':

        D(f_theta(x), f_theta'(x))

    is calculated from full next-token logits. Every variant is isolated:
    the original checkpoint is freshly loaded before each modification.
    """

    import torch

    split = str(split).lower()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prompts_data = [dict(row) for row in prompt_rows]
    manifest_data = [dict(row) for row in manifests]

    _validate_prompts(prompts_data)
    _validate_manifests(manifest_data, split)

    prompts = [
        str(row["prompt"])
        for row in prompts_data
    ]
    weights = _family_weights(manifest_data)

    print(
        json.dumps(
            {
                "stage": "reference_logits",
                "prompts": len(prompts),
                "variants": len(manifest_data),
                "split": split,
            },
            ensure_ascii=False,
        )
    )

    reference_bundle = load_model(dict(base_model_config))

    try:
        reference_logits = _collect_logits(
            reference_bundle,
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            system_prompt=system_prompt,
        )
        reference_model_name = str(reference_bundle.name)
    finally:
        reference_bundle.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    records: list[dict[str, Any]] = []

    for variant_index, manifest in enumerate(manifest_data, 1):
        variant_id = str(
            manifest.get(
                "variant_id",
                f"variant_{variant_index:06d}",
            )
        )
        family = _manifest_family(manifest)
        adapter_path = _resolve_adapter_path(
            manifest,
            adapter_paths,
        )

        print(
            json.dumps(
                {
                    "stage": "variant",
                    "index": variant_index,
                    "total": len(manifest_data),
                    "variant_id": variant_id,
                    "family": family,
                },
                ensure_ascii=False,
            )
        )

        with load_manifest_variant(
            base_model_config,
            manifest,
            adapter_path=adapter_path,
        ) as loaded:
            variant_logits = _collect_logits(
                loaded.bundle,
                prompts,
                batch_size=batch_size,
                max_length=max_length,
                system_prompt=system_prompt,
            )

            execution_report = loaded.report

        if variant_logits.shape != reference_logits.shape:
            raise RuntimeError(
                f"Reference logits shape {tuple(reference_logits.shape)} "
                f"does not match variant logits shape "
                f"{tuple(variant_logits.shape)} for {variant_id}"
            )

        for prompt_index, prompt_row in enumerate(prompts_data):
            distances = compute_macro_distances(
                reference_logits[prompt_index],
                variant_logits[prompt_index],
            )

            record = PromptMacroRecord(
                prompt_id=_prompt_id(
                    prompt_row,
                    prompt_index,
                ),
                category=str(
                    prompt_row.get(
                        "category",
                        "unknown",
                    )
                ),
                variant_id=variant_id,
                family=family,
                split=split,
                seed=int(manifest.get("seed", seed)),
                family_weight=float(
                    manifest.get(
                        "weight",
                        weights[family],
                    )
                ),
                macro_l2_raw=float(distances.macro_l2_raw),
                macro_l2_per_dimension=float(
                    distances.macro_l2_per_dimension
                ),
                macro_probability_l2=float(
                    distances.macro_probability_l2
                ),
                macro_js=float(distances.macro_js),
                output_dimension=int(
                    distances.output_dimension
                ),
                realized_method=execution_report.realized_method,
                execution_mode=execution_report.execution_mode,
            )

            records.append(asdict(record))

        del variant_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    records_path = output_root / f"macro_records_{split}.jsonl"
    _write_jsonl(records_path, records)

    summary = summarize_stratified_macro(
        records,
        family_weights=weights,
        metric=primary_metric,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )

    summary_payload = {
        "split": split,
        "reference_model": reference_model_name,
        "prompt_count": len(prompts_data),
        "variant_count": len(manifest_data),
        "observation_count": len(records),
        "primary_metric": primary_metric,
        "family_weights": weights,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "seed": seed,
        "summary": asdict(summary),
    }

    summary_path = output_root / f"macro_summary_{split}.json"
    summary_path.write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    del reference_logits
    gc.collect()

    return MacroSamplingReport(
        split=split,
        prompt_count=len(prompts_data),
        variant_count=len(manifest_data),
        observation_count=len(records),
        reference_model=reference_model_name,
        primary_metric=primary_metric,
        family_weights=weights,
        records_path=str(records_path),
        summary_path=str(summary_path),
    )


def sample_macro_sensitivity_from_files(
    *,
    base_model_config: Mapping[str, Any],
    prompts_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    split: str,
    batch_size: int = 1,
    max_length: int = 512,
    system_prompt: str | None = None,
    primary_metric: str = "macro_l2_raw",
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    adapter_paths: Mapping[str, str | Path] | None = None,
) -> MacroSamplingReport:
    """File-based wrapper around sample_macro_sensitivity()."""

    return sample_macro_sensitivity(
        base_model_config=base_model_config,
        prompt_rows=_read_jsonl(prompts_path),
        manifests=_read_jsonl(manifest_path),
        output_dir=output_dir,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        system_prompt=system_prompt,
        primary_metric=primary_metric,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
        adapter_paths=adapter_paths,
    )
