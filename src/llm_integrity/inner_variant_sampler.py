from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FAMILIES = (
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
)


@dataclass(frozen=True)
class VariantSample:
    family: str
    variant_id: str
    seed: int
    manifest: dict[str, Any]
    q_family: float
    q_variant_given_family: float
    q_joint: float
    target_family_weight: float
    target_variant_weight: float
    importance_correction: float
    adapter_path: str | None

    def trace(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "q_family": self.q_family,
            "q_variant_given_family": self.q_variant_given_family,
            "q_joint": self.q_joint,
            "target_family_weight": self.target_family_weight,
            "target_variant_weight": self.target_variant_weight,
            "importance_correction": self.importance_correction,
            "adapter_path": self.adapter_path,
            "configuration": self.manifest.get("configuration", {}),
            "split": self.manifest.get("split"),
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_registry(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload.items()}


class StratifiedVariantSampler:
    def __init__(
        self,
        manifests: list[dict[str, Any]],
        *,
        family_weights: Mapping[str, float],
        adapter_registry: Mapping[str, str] | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.family_weights = {
            family: float(family_weights[family])
            for family in FAMILIES
        }
        total_weight = sum(self.family_weights.values())
        if total_weight <= 0:
            raise ValueError("Family weights must sum to a positive value")
        self.family_weights = {
            family: value / total_weight
            for family, value in self.family_weights.items()
        }
        self.adapter_registry = dict(adapter_registry or {})
        grouped: dict[str, list[dict[str, Any]]] = {
            family: [] for family in FAMILIES
        }
        for manifest in manifests:
            family = str(manifest.get("family", "")).lower()
            if family not in grouped:
                continue
            configuration = manifest.get("configuration", {})
            if (
                family == "quantization"
                and (
                    str(configuration.get("target_scope", "")).lower()
                    != "full_model"
                    or str(configuration.get("compute_dtype", "")).lower()
                    != "bfloat16"
                )
            ):
                # bitsandbytes applies supported quantization during model
                # loading and cannot faithfully realize partial scopes.
                # float16 variants produced finite logits but non-finite
                # input gradients in the joint inner-loop smoke test.
                continue
            if family == "finetuning":
                variant_id = str(manifest.get("variant_id"))
                if variant_id not in self.adapter_registry:
                    continue
            grouped[family].append(dict(manifest))
        for family, rows in grouped.items():
            rows.sort(key=lambda row: str(row.get("variant_id")))
            if not rows:
                raise ValueError(f"No executable training variants for {family}")
        self.grouped = grouped

    def _build_sample(
        self,
        family: str,
        manifest: dict[str, Any],
    ) -> VariantSample:
        rows = self.grouped[family]
        q_family = self.family_weights[family]
        conditional = 1.0 / len(rows)
        q_joint = q_family * conditional
        target_variant = self.family_weights[family] / len(rows)
        correction = target_variant / q_joint
        variant_id = str(manifest["variant_id"])
        return VariantSample(
            family=family,
            variant_id=variant_id,
            seed=int(manifest.get("seed", 0)),
            manifest=manifest,
            q_family=q_family,
            q_variant_given_family=conditional,
            q_joint=q_joint,
            target_family_weight=self.family_weights[family],
            target_variant_weight=target_variant,
            importance_correction=correction,
            adapter_path=self.adapter_registry.get(variant_id),
        )

    def family_order(self, prompt_id: str) -> list[str]:
        digest = hashlib.sha256(
            f"{self.seed}:{prompt_id}:families".encode("utf-8")
        ).digest()
        local_seed = int.from_bytes(digest[:8], "big")
        order = list(FAMILIES)
        random.Random(local_seed).shuffle(order)
        return order

    def sample_family_variant(
        self,
        prompt_id: str,
        family: str,
        *,
        cycle: int = 0,
    ) -> VariantSample:
        rows = self.grouped[family]
        digest = hashlib.sha256(
            f"{self.seed}:{prompt_id}:{family}:{cycle}".encode("utf-8")
        ).digest()
        index = int.from_bytes(digest[:8], "big") % len(rows)
        manifest = rows[index]
        return self._build_sample(family, manifest)

    def sample_family_variants(
        self,
        prompt_id: str,
        family: str,
        *,
        count: int,
        cycle: int = 0,
    ) -> list[VariantSample]:
        """Select a deterministic without-replacement family mini-batch.

        The per-prompt permutation is fixed, while ``cycle`` advances through
        non-overlapping windows until the family pool wraps. This reduces the
        single-variant variance of the inner family objective without using
        any held-out validation variant.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        rows = self.grouped[family]
        if count > len(rows):
            raise ValueError(
                f"Requested {count} variants for {family}, "
                f"but only {len(rows)} are executable"
            )
        digest = hashlib.sha256(
            (
                f"{self.seed}:{prompt_id}:{family}:"
                "without_replacement_order"
            ).encode("utf-8")
        ).digest()
        order = list(range(len(rows)))
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
        start = (int(cycle) * count) % len(order)
        indices = [
            order[(start + offset) % len(order)]
            for offset in range(count)
        ]
        return [
            self._build_sample(family, rows[index])
            for index in indices
        ]

    def sample_disjoint_family_variants(
        self,
        prompt_id: str,
        family: str,
        *,
        search_count: int,
        anchor_count: int,
        cycle: int = 0,
    ) -> tuple[list[VariantSample], list[VariantSample]]:
        """Return rotating search variants and fixed disjoint anchors.

        Anchors remain fixed across rounds for one prompt. Search variants
        rotate only through the rest of the family pool, so candidate
        reranking is performed on variants that did not produce the gradient.
        """
        if search_count <= 0:
            raise ValueError("search_count must be positive")
        if anchor_count <= 0:
            raise ValueError("anchor_count must be positive")
        rows = self.grouped[family]
        required = search_count + anchor_count
        if required > len(rows):
            raise ValueError(
                f"Requested {search_count} search and {anchor_count} anchor "
                f"variants for {family}, but only {len(rows)} are executable"
            )
        digest = hashlib.sha256(
            (
                f"{self.seed}:{prompt_id}:{family}:"
                "disjoint_search_anchor_order"
            ).encode("utf-8")
        ).digest()
        order = list(range(len(rows)))
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
        anchor_indices = order[:anchor_count]
        search_order = order[anchor_count:]
        start = (int(cycle) * search_count) % len(search_order)
        search_indices = [
            search_order[(start + offset) % len(search_order)]
            for offset in range(search_count)
        ]
        search = [
            self._build_sample(family, rows[index])
            for index in search_indices
        ]
        anchors = [
            self._build_sample(family, rows[index])
            for index in anchor_indices
        ]
        return search, anchors

    def balanced_schedule(
        self,
        prompt_id: str,
        *,
        block_types: tuple[str, ...] = (
            "q_proj",
            "v_proj",
            "down_proj",
        ),
        cycle: int = 0,
    ) -> list[dict[str, Any]]:
        schedule: list[dict[str, Any]] = []
        for family in self.family_order(prompt_id):
            sample = self.sample_family_variant(
                prompt_id,
                family,
                cycle=cycle,
            )
            for block_type in block_types:
                schedule.append(
                    {
                        "step": len(schedule) + 1,
                        "block_type": block_type,
                        "sample": sample,
                    }
                )
        return schedule

    def representative_samples(self, prompt_id: str = "calibration") -> list[VariantSample]:
        return [
            self.sample_family_variant(prompt_id, family)
            for family in FAMILIES
        ]
