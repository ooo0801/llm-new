from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_integrity.paper_attack_sampling import (
    build_attack_manifests,
    write_attack_manifests,
)


def unstructured_pruning_space() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    for ratio, method, target_scope in product(
        [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        [
            "global_magnitude",
            "layerwise_magnitude",
            "random",
        ],
        [
            "attention",
            "ffn",
            "attention_ffn",
        ],
    ):
        configurations.append(
            {
                "type": "unstructured_pruning",
                "ratio": ratio,
                "method": method,
                "target_scope": target_scope,
            }
        )

    return configurations


def structured_pruning_space() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    for ratio, structure, selection, layer_scope in product(
        [0.05, 0.10, 0.20, 0.30],
        [
            "attention_heads",
            "ffn_channels",
        ],
        [
            "magnitude",
            "random",
        ],
        [
            "all_layers",
            "random_layer_subset",
        ],
    ):
        configurations.append(
            {
                "type": "structured_pruning",
                "ratio": ratio,
                "structure": structure,
                "selection": selection,
                "layer_scope": layer_scope,
                "implementation": "mask",
            }
        )

    return configurations


def quantization_space() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    for method in ["int8"]:
        for compute_dtype, target_scope in product(
            ["bfloat16", "float16"],
            ["full_model", "attention", "ffn"],
        ):
            configurations.append(
                {
                    "type": "quantization",
                    "method": method,
                    "compute_dtype": compute_dtype,
                    "double_quant": False,
                    "target_scope": target_scope,
                }
            )


    # bitsandbytes 的 4 位加载真实支持 NF4 和 FP4。
    # 不再把 NF4 错误地命名为严格整数 INT4。
    for method in ["nf4", "fp4"]:
        for compute_dtype, double_quant, target_scope in product(
            ["bfloat16", "float16"],
            [False, True],
            ["full_model", "attention", "ffn"],
        ):
            configurations.append(
                {
                    "type": "quantization",
                    "method": method,
                    "compute_dtype": compute_dtype,
                    "double_quant": double_quant,
                    "target_scope": target_scope,
                }
            )


    return configurations


def gaussian_noise_space() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    for std_ratio, target_scope in product(
        [0.0001, 0.0005, 0.001, 0.005, 0.01],
        [
            "attention",
            "ffn",
            "partial_layers",
            "all_transformer_layers",
        ],
    ):
        configurations.append(
            {
                "type": "gaussian_noise",
                "std_ratio": std_ratio,
                "target_scope": target_scope,
                "scale_rule": "parameter_tensor_std",
            }
        )

    return configurations


def finetuning_space() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    for rank, learning_rate, steps, target_scope in product(
        [4, 8, 16],
        [1e-5, 5e-5],
        [25, 50],
        [
            "attention",
            "attention_ffn",
        ],
    ):
        configurations.append(
            {
                "type": "finetuning",
                "method": "lora",
                "rank": rank,
                "alpha": 2 * rank,
                "dropout": 0.05,
                "learning_rate": learning_rate,
                "steps": steps,
                "target_scope": target_scope,
                "data_source": "isolated_attack_training_data",
            }
        )

    return configurations


def build_family_spaces() -> dict[
    str,
    list[dict[str, Any]],
]:
    return {
        "unstructured_pruning": (
            unstructured_pruning_space()
        ),
        "structured_pruning": (
            structured_pruning_space()
        ),
        "quantization": quantization_space(),
        "gaussian_noise": gaussian_noise_space(),
        "finetuning": finetuning_space(),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build proposal-aligned train, validation "
            "and test attack manifests."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/paper_aligned_qwen_7b/manifests"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    family_spaces = build_family_spaces()

    family_weights = {
        "unstructured_pruning": 0.2,
        "structured_pruning": 0.2,
        "quantization": 0.2,
        "gaussian_noise": 0.2,
        "finetuning": 0.2,
    }

    # 三个集合使用完全不同的随机种子。
    split_seeds = {
        "train": [42, 123],
        "validation": [2025],
        "test": [3407, 777],
    }

    manifests = build_attack_manifests(
        family_spaces=family_spaces,
        family_weights=family_weights,
        split_seeds=split_seeds,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=arguments.seed,
    )

    outputs = write_attack_manifests(
        manifests,
        output_directory=arguments.output_dir,
    )

    summary: dict[str, Any] = {
        "generation_seed": arguments.seed,
        "family_weights": family_weights,
        "split_seeds": split_seeds,
        "configuration_counts": {
            family: len(configurations)
            for family, configurations
            in family_spaces.items()
        },
        "manifest_counts": {},
        "outputs": outputs,
    }

    for split, variants in manifests.items():
        family_counts: dict[str, int] = {}

        for variant in variants:
            family_counts[variant.family] = (
                family_counts.get(variant.family, 0)
                + 1
            )

        summary["manifest_counts"][split] = {
            "total": len(variants),
            "by_family": family_counts,
        }

    output_directory = Path(arguments.output_dir)
    summary_path = (
        output_directory / "attack_manifest_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
