from __future__ import annotations

import gc
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .inner_micro_proxy import (
    discover_micro_blocks,
    differentiable_block_micro_proxy,
    representative_layers,
    score_block_micro_proxy,
    select_micro_block,
)
from .inner_variant_sampler import (
    FAMILIES,
    StratifiedVariantSampler,
    VariantSample,
)
from .joint_inner_optimizer import JointInnerOptimizer
from .modeling import ModelBundle, generate_texts, load_model
from .paper_variant_executor import load_manifest_variant
from .sequential_macro import squared_l2_logit_vjp
from .macro_proxy import SEARCH_PROXIES, differentiable_macro_proxy
from .task_validation import evaluate_task


@dataclass(frozen=True)
class HotFlipCandidate:
    position: int
    source_token_id: int
    candidate_token_id: int
    linear_gain: float


@dataclass
class DiscreteJointOptimizationResult:
    prompt_id: str
    initial_prompt: str
    optimized_prompt: str
    accepted: bool
    acceptance_stage: str
    requires_hard_validation: bool
    initial_ppl: float
    final_ppl: float
    ppl_ratio: float
    edit_ratio: float
    edit_count: int
    rounds_requested: int
    rounds_completed: int
    committed_rounds: int
    proxy_objective_gain: float
    active_user_tokens: int
    preserved_overflow_tokens: int
    optimization_max_length: int
    micro_weight: float
    macro_weight: float
    initial_task_passed: bool | None
    final_task_passed: bool | None
    history: list[dict[str, Any]]
    failure: dict[str, Any] | None

    def payload(self) -> dict[str, Any]:
        return dict(self.__dict__)


def resolve_final_task_status(
    *,
    require_task_preservation: bool,
    initial_task_passed: bool,
    accepted: bool,
    history: list[dict[str, Any]],
) -> bool | None:
    """Report the task status of the state that is actually returned.

    Rejected searches are rolled back to the initial prompt, so their final
    task status must be the initial status rather than ``False``.
    """
    if not require_task_preservation:
        return None
    if not accepted:
        return bool(initial_task_passed)
    return bool(
        initial_task_passed
        and all(
            bool(
                trace.get(
                    "committed_task_validation",
                    {},
                ).get("task_passed", False)
            )
            for trace in history
            if trace.get("committed")
        )
    )


def hotflip_top_candidates(
    embedding_matrix: torch.Tensor,
    gradient: torch.Tensor,
    current_ids: list[int],
    editable_positions: Iterable[int],
    *,
    candidates_per_position: int,
    forbidden_token_ids: set[int] | None = None,
    chunk_size: int = 4096,
) -> list[HotFlipCandidate]:
    """Return top first-order token replacements without copying the vocabulary."""
    if gradient.ndim != 2:
        raise ValueError("gradient must have shape [tokens, hidden]")
    if len(current_ids) != gradient.shape[0]:
        raise ValueError("current_ids and gradient length differ")
    if candidates_per_position <= 0:
        raise ValueError("candidates_per_position must be positive")

    forbidden = set(forbidden_token_ids or ())
    output: list[HotFlipCandidate] = []
    matrix = embedding_matrix.detach()
    for position in editable_positions:
        source_id = int(current_ids[position])
        source_score = float(
            torch.dot(
                matrix[source_id].float(),
                gradient[position].float(),
            ).item()
        )
        best: list[tuple[float, int]] = []
        for start in range(0, matrix.shape[0], chunk_size):
            stop = min(start + chunk_size, matrix.shape[0])
            scores = (
                matrix[start:stop].float()
                @ gradient[position].float()
            )
            local_k = min(
                scores.numel(),
                max(candidates_per_position * 2, 8),
            )
            values, indices = torch.topk(scores, k=local_k)
            for value, index in zip(
                values.tolist(),
                indices.tolist(),
                strict=True,
            ):
                token_id = int(start + index)
                if token_id == source_id or token_id in forbidden:
                    continue
                best.append((float(value - source_score), token_id))

        best.sort(key=lambda item: item[0], reverse=True)
        seen: set[int] = set()
        kept = 0
        for gain, token_id in best:
            if token_id in seen:
                continue
            seen.add(token_id)
            output.append(
                HotFlipCandidate(
                    position=int(position),
                    source_token_id=source_id,
                    candidate_token_id=token_id,
                    linear_gain=float(gain),
                )
            )
            kept += 1
            if kept >= candidates_per_position:
                break
    output.sort(key=lambda item: item.linear_gain, reverse=True)
    return output


def select_position_diverse_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep the strongest feasible candidate from each position first."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int]] = set()
    used_positions: set[int] = set()
    for item in candidates:
        position = int(item["position"])
        if position in used_positions:
            continue
        selected.append(dict(item))
        used_positions.add(position)
        selected_ids.add(
            (position, int(item["candidate_token_id"]))
        )
        if len(selected) >= limit:
            return selected
    for item in candidates:
        identity = (
            int(item["position"]),
            int(item["candidate_token_id"]),
        )
        if identity in selected_ids:
            continue
        selected.append(dict(item))
        selected_ids.add(identity)
        if len(selected) >= limit:
            break
    return selected


def merge_restart_candidate_pools(
    pools: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge candidates from independent gradient directions.

    A candidate can be proposed by more than one search restart.  Exact
    reranking only needs to score that discrete token sequence once, while the
    trace retains every restart and its first-order gain.
    """
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for restart_index, pool in enumerate(pools, start=1):
        for candidate in pool:
            identity = tuple(int(value) for value in candidate["token_ids"])
            gain = float(candidate["linear_gain"])
            if identity not in merged:
                item = dict(candidate)
                item["search_restarts"] = [restart_index]
                item["restart_linear_gains"] = {str(restart_index): gain}
                merged[identity] = item
                continue
            item = merged[identity]
            item["search_restarts"].append(restart_index)
            item["restart_linear_gains"][str(restart_index)] = gain
            if gain > float(item["linear_gain"]):
                provenance = {
                    "search_restarts": item["search_restarts"],
                    "restart_linear_gains": item["restart_linear_gains"],
                }
                item.update(candidate)
                item.update(provenance)
    output = list(merged.values())
    output.sort(key=lambda item: float(item["linear_gain"]), reverse=True)
    return output


def token_surface_class(text: str) -> str:
    """Classify token text after ignoring tokenizer-introduced spacing."""
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return "whitespace"
    has_cjk = any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in characters
    )
    has_latin = any(
        character.isascii() and character.isalpha()
        for character in characters
    )
    has_digit = any(character.isdigit() for character in characters)
    if has_digit:
        return "digit_mixed"
    if has_cjk and has_latin:
        return "mixed_cjk_latin"
    if has_cjk:
        return "cjk"
    if has_latin:
        return "latin"
    if all(
        unicodedata.category(character)[0] in {"P", "S"}
        for character in characters
    ):
        return "punctuation_symbol"
    return "other"


def token_surfaces_compatible(source: str, candidate: str) -> bool:
    """Reject cross-script replacements that pass global PPL by dilution."""
    source_class = token_surface_class(source)
    candidate_class = token_surface_class(candidate)
    if source_class == "digit_mixed":
        return False
    return source_class == candidate_class


def family_nondegradation_checks(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
    *,
    relative_tolerance: float,
) -> tuple[int, dict[str, dict[str, Any]]]:
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")
    if set(baseline_scores) != set(FAMILIES):
        raise ValueError("baseline_scores must contain all five families")
    if set(candidate_scores) != set(FAMILIES):
        raise ValueError("candidate_scores must contain all five families")
    checks: dict[str, dict[str, Any]] = {}
    nondegraded_families = 0
    for family in FAMILIES:
        baseline = float(baseline_scores[family])
        candidate = float(candidate_scores[family])
        tolerance = relative_tolerance * max(abs(baseline), 1e-12)
        gain = candidate - baseline
        nondegraded = bool(gain >= -tolerance)
        nondegraded_families += int(nondegraded)
        checks[family] = {
            "baseline_normalized": baseline,
            "candidate_normalized": candidate,
            "gain_normalized": gain,
            "nondegraded": nondegraded,
            "relative_tolerance": relative_tolerance,
        }
    return nondegraded_families, checks


def aggregate_family_trace_scores(
    traces: list[dict[str, Any]],
    *,
    score_key: str,
) -> dict[str, float]:
    """Average variant scores inside each family."""
    grouped: dict[str, list[float]] = {
        family: [] for family in FAMILIES
    }
    for trace in traces:
        family = str(trace["family"])
        if family not in grouped:
            raise ValueError(f"Unknown family in trace: {family}")
        grouped[family].append(float(trace[score_key]))
    missing = [
        family for family, scores in grouped.items() if not scores
    ]
    if missing:
        raise ValueError(f"Missing family scores: {missing}")
    return {
        family: sum(scores) / len(scores)
        for family, scores in grouped.items()
    }


def family_anchor_nondegradation_checks(
    baseline_variant_scores: dict[str, list[dict[str, Any]]],
    candidate_variant_scores: dict[str, list[dict[str, Any]]],
    *,
    relative_tolerance: float,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Require every fixed inner-development anchor in a family to hold."""
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")
    if set(baseline_variant_scores) != set(FAMILIES):
        raise ValueError(
            "baseline_variant_scores must contain all five families"
        )
    if set(candidate_variant_scores) != set(FAMILIES):
        raise ValueError(
            "candidate_variant_scores must contain all five families"
        )

    checks: dict[str, dict[str, Any]] = {}
    nondegraded_families = 0
    for family in FAMILIES:
        baseline_by_id = {
            str(item["variant_id"]): float(item["normalized"])
            for item in baseline_variant_scores[family]
        }
        candidate_by_id = {
            str(item["variant_id"]): float(item["normalized"])
            for item in candidate_variant_scores[family]
        }
        if not baseline_by_id:
            raise ValueError(f"No baseline anchor variants for {family}")
        if set(baseline_by_id) != set(candidate_by_id):
            raise ValueError(
                f"Baseline and candidate anchor variants differ for {family}"
            )
        variant_checks: list[dict[str, Any]] = []
        for variant_id in sorted(baseline_by_id):
            baseline = baseline_by_id[variant_id]
            candidate = candidate_by_id[variant_id]
            tolerance = relative_tolerance * max(abs(baseline), 1e-12)
            gain = candidate - baseline
            variant_checks.append(
                {
                    "variant_id": variant_id,
                    "baseline_normalized": baseline,
                    "candidate_normalized": candidate,
                    "gain_normalized": gain,
                    "nondegraded": bool(gain >= -tolerance),
                }
            )
        family_nondegraded = all(
            item["nondegraded"] for item in variant_checks
        )
        nondegraded_families += int(family_nondegraded)
        baseline_mean = sum(baseline_by_id.values()) / len(baseline_by_id)
        candidate_mean = sum(candidate_by_id.values()) / len(candidate_by_id)
        checks[family] = {
            "baseline_normalized": baseline_mean,
            "candidate_normalized": candidate_mean,
            "gain_normalized": candidate_mean - baseline_mean,
            "nondegraded": family_nondegraded,
            "relative_tolerance": relative_tolerance,
            "aggregation": "all_anchor_variants",
            "variants": variant_checks,
        }
    return nondegraded_families, checks


class DiscreteJointInnerOptimizer(JointInnerOptimizer):
    """Projected coordinate-ascent optimizer for deployable prompt tokens."""

    def __init__(
        self,
        *,
        reference: ModelBundle,
        sampler: StratifiedVariantSampler,
        model_config: dict[str, Any],
        micro_scales: dict[str, float],
        macro_scales: dict[str, float],
        micro_component_clips: dict[str, float | None] | None = None,
        macro_component_clips: dict[str, float | None] | None = None,
        micro_weight: float = 1.0,
        macro_weight: float = 1.0,
        rounds: int = 3,
        candidate_positions: int = 4,
        candidates_per_position: int = 8,
        rerank_candidates: int = 4,
        minimum_proxy_gain: float = 0.0,
        probes: int = 1,
        seed: int = 42,
        max_length: int = 64,
        max_edit_ratio: float = 0.25,
        ppl_ratio_limit: float = 2.0,
        minimum_nondegraded_families: int = 0,
        family_relative_tolerance: float = 0.01,
        variants_per_family: int = 1,
        anchor_variants_per_family: int = 0,
        gradient_restarts: int = 1,
        family_gate_aggregation: str = "mean",
        require_task_preservation: bool = False,
        task_max_input_tokens: int = 512,
        task_max_new_tokens: int = 128,
        sequential_model_execution: bool = False,
        block_types: tuple[str, ...] = ("q_proj", "v_proj", "down_proj"),
        representative_layer_count: int = 4,
        block_schedule: str = "legacy",
        task_validation_mode: str = "legacy",
        macro_proxy: str = "raw_logit_l2",
        macro_top_k: int = 10,
        enforce_surface_compatibility: bool = True,
        enforce_perplexity: bool = True,
    ) -> None:
        super().__init__(
            reference=reference,
            sampler=sampler,
            model_config=model_config,
            micro_scale=1.0,
            macro_scale=1.0,
            micro_weight=micro_weight,
            macro_weight=macro_weight,
            probes=probes,
            seed=seed,
            max_length=max_length,
            max_edit_ratio=max_edit_ratio,
            ppl_ratio_limit=ppl_ratio_limit,
            block_types=block_types,
            representative_layer_count=representative_layer_count,
        )
        if set(micro_scales) != set(self.block_types):
            raise ValueError(
                "micro_scales must match the selected block_types"
            )
        if set(macro_scales) != set(FAMILIES):
            raise ValueError("macro_scales must contain all five families")
        if any(float(value) <= 0 for value in micro_scales.values()):
            raise ValueError("All micro scales must be positive")
        if any(float(value) <= 0 for value in macro_scales.values()):
            raise ValueError("All macro scales must be positive")
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        if candidate_positions <= 0:
            raise ValueError("candidate_positions must be positive")
        if candidates_per_position <= 0:
            raise ValueError("candidates_per_position must be positive")
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        if not 0 <= minimum_nondegraded_families <= len(FAMILIES):
            raise ValueError(
                "minimum_nondegraded_families must be between 0 and 5"
            )
        if family_relative_tolerance < 0:
            raise ValueError(
                "family_relative_tolerance must be nonnegative"
            )
        if variants_per_family <= 0:
            raise ValueError("variants_per_family must be positive")
        if anchor_variants_per_family < 0:
            raise ValueError(
                "anchor_variants_per_family must be nonnegative"
            )
        if gradient_restarts <= 0:
            raise ValueError("gradient_restarts must be positive")
        if family_gate_aggregation not in {"mean", "all_anchor_variants"}:
            raise ValueError(
                "family_gate_aggregation must be mean or "
                "all_anchor_variants"
            )
        required_per_family = (
            variants_per_family + anchor_variants_per_family
            if anchor_variants_per_family
            else variants_per_family
        )
        undersized = {
            family: len(self.sampler.grouped[family])
            for family in FAMILIES
            if len(self.sampler.grouped[family]) < required_per_family
        }
        if undersized:
            raise ValueError(
                "Not enough executable variants for family mini-batch: "
                f"{undersized}"
            )
        if task_max_input_tokens <= 0 or task_max_new_tokens <= 0:
            raise ValueError("Task generation lengths must be positive")
        if macro_proxy not in SEARCH_PROXIES:
            raise ValueError(f"Unknown macro proxy: {macro_proxy}")
        if macro_top_k <= 0:
            raise ValueError("macro_top_k must be positive")
        if sequential_model_execution and macro_proxy != "raw_logit_l2":
            raise ValueError(
                "Sequential exact VJP currently supports raw_logit_l2 only"
            )

        self.micro_scales = {
            key: float(value) for key, value in micro_scales.items()
        }
        self.macro_scales = {
            key: float(value) for key, value in macro_scales.items()
        }
        self.micro_component_clips = dict(micro_component_clips or {})
        self.macro_component_clips = dict(macro_component_clips or {})
        self.rounds = int(rounds)
        self.candidate_positions = int(candidate_positions)
        self.candidates_per_position = int(candidates_per_position)
        self.rerank_candidates = int(rerank_candidates)
        self.minimum_proxy_gain = float(minimum_proxy_gain)
        self.minimum_nondegraded_families = int(
            minimum_nondegraded_families
        )
        self.family_relative_tolerance = float(
            family_relative_tolerance
        )
        self.variants_per_family = int(variants_per_family)
        self.anchor_variants_per_family = int(
            anchor_variants_per_family
        )
        self.gradient_restarts = int(gradient_restarts)
        self.family_gate_aggregation = str(family_gate_aggregation)
        self.require_task_preservation = bool(
            require_task_preservation
        )
        self.sequential_model_execution = bool(
            sequential_model_execution
        )
        if block_schedule not in {"legacy", "balanced"}:
            raise ValueError("Unknown block schedule")
        self.block_schedule = block_schedule
        if task_validation_mode not in {"legacy", "strict_r1"}:
            raise ValueError("Unknown task validation mode")
        self.task_validation_mode = task_validation_mode
        self.macro_proxy = str(macro_proxy)
        self.macro_top_k = int(macro_top_k)
        self.enforce_surface_compatibility = bool(
            enforce_surface_compatibility
        )
        self.enforce_perplexity = bool(enforce_perplexity)
        self.task_generation = {
            "max_input_tokens": int(task_max_input_tokens),
            "max_new_tokens": int(task_max_new_tokens),
            "do_sample": False,
            "system_prompt": None,
        }

    def _release_reference(self) -> None:
        self.reference.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reload_reference(self) -> None:
        self.reference = load_model(self.model_config)
        self._freeze(self.reference)
        self.blocks = discover_micro_blocks(self.reference.model, self.block_types)
        self.layers = representative_layers(self.blocks, self.representative_layer_count)

    def _task_validation(
        self,
        row: dict[str, Any],
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        if not self.require_task_preservation:
            return [
                {
                    "task_passed": True,
                    "evaluation_rule": "disabled",
                    "generated_text": None,
                }
                for _ in prompts
            ]
        generated = generate_texts(
            self.reference,
            prompts,
            self.task_generation,
            return_metadata=self.task_validation_mode == "strict_r1",
        )
        outputs: list[dict[str, Any]] = []
        for item in generated:
            metadata = {}
            if self.task_validation_mode == "strict_r1":
                from .stage1_r1 import evaluate_task_r1
                text = item["text"]
                checked = evaluate_task_r1(text, row)
                passed = checked["passed"] is True and not item["truncated"]
                rule = "strict_r1:" + checked["reason"]
                metadata = {"truncated": item["truncated"], "generated_tokens": item["token_count"],
                            "task_check": checked}
            else:
                text = item
                passed, rule = evaluate_task(row, text)
            outputs.append(
                {
                    "task_passed": bool(passed),
                    "evaluation_rule": rule,
                    "generated_text": text,
                    **metadata,
                }
            )
        return outputs

    def _chat_parts(
        self,
        prompt: str,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        tokenizer = self.reference.tokenizer
        full_user_ids = list(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if not full_user_ids:
            raise ValueError("Prompt tokenization is empty")
        if tokenizer.chat_template:
            full_ids = list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        else:
            full_ids = list(full_user_ids)
        start = self._find_subsequence(full_ids, full_user_ids)
        if start < 0:
            raise ValueError("Could not locate user tokens in chat template")
        end = start + len(full_user_ids)
        prefix = full_ids[:start]
        suffix = full_ids[end:]
        allowed_user = max(1, self.max_length - len(prefix) - len(suffix))
        return (
            prefix,
            full_user_ids[:allowed_user],
            suffix,
            full_user_ids[allowed_user:],
        )

    def _embed_user(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.reference.device,
        )
        with torch.no_grad():
            return (
                self.reference.model.get_input_embeddings()(ids)
                .detach()
                .float()
            )

    def _decode_user(self, token_ids: list[int]) -> str:
        try:
            return self.reference.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.reference.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

    def _valid_discrete_text(
        self,
        *,
        initial_prompt: str,
        active_ids: list[int],
        overflow_ids: list[int],
    ) -> tuple[bool, str, str | None]:
        complete = active_ids + overflow_ids
        decoded = self._decode_user(complete)
        if not decoded.strip():
            return False, decoded, "empty_decoded_prompt"
        if "\ufffd" in decoded:
            return False, decoded, "replacement_character"
        if any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in decoded
        ):
            return False, decoded, "control_character"
        roundtrip = list(
            self.reference.tokenizer(
                decoded,
                add_special_tokens=False,
            )["input_ids"]
        )
        if roundtrip != complete:
            return False, decoded, "tokenizer_roundtrip_mismatch"
        if re.findall(r"\d+", decoded) != re.findall(r"\d+", initial_prompt):
            return False, decoded, "digit_sequence_changed"
        return True, decoded, None

    def _micro_gradient(
        self,
        *,
        prefix_ids: list[int],
        suffix_ids: list[int],
        current_embeddings: torch.Tensor,
        block_type: str,
        layer_id: int,
        probe_seed: int,
    ) -> tuple[torch.Tensor, float, dict[str, Any], Any]:
        full, mask, user_slice = self._compose(
            prefix_ids,
            current_embeddings,
            suffix_ids,
        )
        block = select_micro_block(
            self.blocks,
            block_type=block_type,
            layer_id=layer_id,
        )
        result = differentiable_block_micro_proxy(
            self.reference,
            full,
            mask,
            block=block,
            probes=self.probes,
            seed=probe_seed,
        )
        scale = self.micro_scales[block_type]
        normalized = result.embedding_gradient[0, user_slice, :] / scale
        normalized, norm_before, clip = self._clip_component(
            normalized,
            self.micro_component_clips.get(block_type),
        )
        trace = {
            "block_type": block_type,
            "layer_id": layer_id,
            "probe_seed": probe_seed,
            "probe_count": self.probes,
            "micro_score_raw": result.raw_micro_score,
            "micro_score_normalized": result.raw_micro_score / scale,
            "micro_gradient_norm_normalized": norm_before,
            "micro_clip_coefficient": clip,
        }
        del full, mask, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return normalized, float(trace["micro_score_normalized"]), trace, block

    def _macro_gradient(
        self,
        *,
        prompt_id: str,
        round_index: int,
        prefix_ids: list[int],
        suffix_ids: list[int],
        current_embeddings: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        float,
        list[dict[str, Any]],
        list[VariantSample],
        list[VariantSample],
    ]:
        if self.sequential_model_execution:
            return self._macro_gradient_sequential(
                prompt_id=prompt_id,
                round_index=round_index,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                current_embeddings=current_embeddings,
            )
        accumulated = torch.zeros_like(current_embeddings)
        objective = 0.0
        traces: list[dict[str, Any]] = []
        samples: list[VariantSample] = []
        anchor_samples: list[VariantSample] = []
        for family in FAMILIES:
            if self.anchor_variants_per_family:
                family_samples, family_anchors = (
                    self.sampler.sample_disjoint_family_variants(
                        prompt_id,
                        family,
                        search_count=self.variants_per_family,
                        anchor_count=self.anchor_variants_per_family,
                        cycle=round_index,
                    )
                )
                anchor_samples.extend(family_anchors)
            else:
                family_samples = self.sampler.sample_family_variants(
                    prompt_id,
                    family,
                    count=self.variants_per_family,
                    cycle=round_index,
                )
            samples.extend(family_samples)
            family_weight = self.sampler.family_weights[family]
            within_family_weight = 1.0 / len(family_samples)
            effective_weight = family_weight * within_family_weight
            for sample in family_samples:
                loaded = None
                try:
                    loaded = load_manifest_variant(
                        self.model_config,
                        sample.manifest,
                        adapter_path=sample.adapter_path,
                    )
                    self._freeze(loaded.bundle)
                    full, mask, user_slice = self._compose(
                        prefix_ids,
                        current_embeddings,
                        suffix_ids,
                    )
                    reference_logits = self._next_logits(
                        self.reference,
                        full,
                        mask,
                    )
                    variant_logits = self._next_logits(
                        loaded.bundle,
                        full,
                        mask,
                    )
                    raw_score = differentiable_macro_proxy(
                        reference_logits,
                        variant_logits,
                        proxy=self.macro_proxy,
                        top_k=self.macro_top_k,
                    )
                    full_gradient = torch.autograd.grad(
                        raw_score,
                        full,
                    )[0]
                    scale = self.macro_scales[family]
                    normalized = (
                        full_gradient[0, user_slice, :].detach().float()
                        / scale
                        * sample.importance_correction
                    )
                    normalized, norm_before, clip = (
                        self._clip_component(
                            normalized,
                            self.macro_component_clips.get(family),
                        )
                    )
                    accumulated += effective_weight * normalized
                    normalized_score = (
                        float(raw_score.detach().item())
                        / scale
                        * sample.importance_correction
                    )
                    objective += effective_weight * normalized_score
                    traces.append(
                        {
                            **sample.trace(),
                            "macro_score_raw": float(
                                raw_score.detach().item()
                            ),
                            "macro_proxy": self.macro_proxy,
                            "macro_score_normalized": normalized_score,
                            "macro_gradient_norm_normalized": norm_before,
                            "macro_clip_coefficient": clip,
                            "family_weight": family_weight,
                            "within_family_weight": (
                                within_family_weight
                            ),
                            "effective_weight": effective_weight,
                            "attack_effect": {
                                "execution_mode": (
                                    loaded.report.execution_mode
                                ),
                                "realized_method": (
                                    loaded.report.realized_method
                                ),
                                "changed_parameters": (
                                    loaded.report.details.get(
                                        "changed_parameters"
                                    )
                                ),
                                "total_parameters": (
                                    loaded.report.details.get(
                                        "total_parameters"
                                    )
                                ),
                            },
                        }
                    )
                    del (
                        full,
                        mask,
                        reference_logits,
                        variant_logits,
                        raw_score,
                        full_gradient,
                        normalized,
                    )
                finally:
                    if loaded is not None:
                        loaded.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if not anchor_samples:
            anchor_samples = list(samples)
        return (
            accumulated,
            float(objective),
            traces,
            samples,
            anchor_samples,
        )

    def _macro_gradient_sequential(
        self,
        *,
        prompt_id: str,
        round_index: int,
        prefix_ids: list[int],
        suffix_ids: list[int],
        current_embeddings: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        float,
        list[dict[str, Any]],
        list[VariantSample],
        list[VariantSample],
    ]:
        """Evaluate the exact macro gradient with one 32B model resident.

        The squared-distance gradient is split into variant and reference
        VJPs.  The two terms are algebraically identical to differentiating
        with both model graphs live, but never require both weight sets in
        GPU memory at once.
        """
        samples: list[VariantSample] = []
        anchor_samples: list[VariantSample] = []
        for family in FAMILIES:
            if self.anchor_variants_per_family:
                family_samples, family_anchors = (
                    self.sampler.sample_disjoint_family_variants(
                        prompt_id,
                        family,
                        search_count=self.variants_per_family,
                        anchor_count=self.anchor_variants_per_family,
                        cycle=round_index,
                    )
                )
                anchor_samples.extend(family_anchors)
            else:
                family_samples = self.sampler.sample_family_variants(
                    prompt_id,
                    family,
                    count=self.variants_per_family,
                    cycle=round_index,
                )
            samples.extend(family_samples)
        if not anchor_samples:
            anchor_samples = list(samples)

        full, mask, user_slice = self._compose(
            prefix_ids,
            current_embeddings,
            suffix_ids,
        )
        with torch.inference_mode():
            reference_logits_cpu = self._next_logits(
                self.reference,
                full.detach(),
                mask,
            ).detach().cpu()
        full_cpu = full.detach().cpu()
        mask_cpu = mask.detach().cpu()
        del full, mask

        variant_terms: list[dict[str, Any]] = []
        self._release_reference()
        try:
            for sample in samples:
                loaded = None
                try:
                    loaded = load_manifest_variant(
                        self.model_config,
                        sample.manifest,
                        adapter_path=sample.adapter_path,
                    )
                    self._freeze(loaded.bundle)
                    variant_full = (
                        full_cpu.to(loaded.bundle.device)
                        .detach()
                        .requires_grad_(True)
                    )
                    variant_mask = mask_cpu.to(loaded.bundle.device)
                    variant_logits = self._next_logits(
                        loaded.bundle,
                        variant_full,
                        variant_mask,
                    )
                    delta = (
                        variant_logits.detach()
                        - reference_logits_cpu.to(variant_logits.device)
                    )
                    raw_score = float(delta.square().sum().item())
                    delta_cpu = delta.cpu()
                    variant_gradient_cpu = squared_l2_logit_vjp(
                        variant_logits,
                        variant_full,
                        delta_cpu,
                        sign=1.0,
                    ).detach().float().cpu()
                    variant_terms.append(
                        {
                            "sample": sample,
                            "delta": delta_cpu,
                            "raw_score": raw_score,
                            "variant_gradient": variant_gradient_cpu,
                            "attack_effect": {
                                "execution_mode": (
                                    loaded.report.execution_mode
                                ),
                                "realized_method": (
                                    loaded.report.realized_method
                                ),
                                "changed_parameters": (
                                    loaded.report.details.get(
                                        "changed_parameters"
                                    )
                                ),
                                "total_parameters": (
                                    loaded.report.details.get(
                                        "total_parameters"
                                    )
                                ),
                            },
                        }
                    )
                    del variant_full, variant_mask, variant_logits, delta
                finally:
                    if loaded is not None:
                        loaded.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            self._reload_reference()

        accumulated = torch.zeros_like(current_embeddings)
        objective = 0.0
        traces: list[dict[str, Any]] = []
        for term in variant_terms:
            sample = term["sample"]
            family = sample.family
            reference_full = (
                full_cpu.to(self.reference.device)
                .detach()
                .requires_grad_(True)
            )
            reference_mask = mask_cpu.to(self.reference.device)
            reference_logits = self._next_logits(
                self.reference,
                reference_full,
                reference_mask,
            )
            reference_gradient_cpu = squared_l2_logit_vjp(
                reference_logits,
                reference_full,
                term["delta"],
                sign=-1.0,
            ).detach().float().cpu()
            full_gradient_cpu = (
                term["variant_gradient"] + reference_gradient_cpu
            )
            scale = self.macro_scales[family]
            normalized = (
                full_gradient_cpu[0, user_slice, :]
                / scale
                * sample.importance_correction
            ).to(accumulated.device)
            normalized, norm_before, clip = self._clip_component(
                normalized,
                self.macro_component_clips.get(family),
            )
            family_weight = self.sampler.family_weights[family]
            within_family_weight = 1.0 / self.variants_per_family
            effective_weight = family_weight * within_family_weight
            accumulated += effective_weight * normalized
            normalized_score = (
                float(term["raw_score"])
                / scale
                * sample.importance_correction
            )
            objective += effective_weight * normalized_score
            traces.append(
                {
                    **sample.trace(),
                    "macro_score_raw": float(term["raw_score"]),
                    "macro_score_normalized": normalized_score,
                    "macro_gradient_norm_normalized": norm_before,
                    "macro_clip_coefficient": clip,
                    "family_weight": family_weight,
                    "within_family_weight": within_family_weight,
                    "effective_weight": effective_weight,
                    "attack_effect": term["attack_effect"],
                    "gradient_execution": "sequential_exact_vjp",
                }
            )
            del (
                reference_full,
                reference_mask,
                reference_logits,
                reference_gradient_cpu,
                full_gradient_cpu,
                normalized,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return accumulated, objective, traces, samples, anchor_samples

    def _candidate_pool(
        self,
        *,
        initial_prompt: str,
        initial_active_ids: list[int],
        current_ids: list[int],
        overflow_ids: list[int],
        combined_gradient: torch.Tensor,
        initial_ppl: float,
    ) -> tuple[
        list[dict[str, Any]],
        list[int],
        dict[str, int],
    ]:
        filter_stats = {
            "hotflip_raw": 0,
            "nonpositive_linear_gain": 0,
            "duplicate": 0,
            "edit_constraint": 0,
            "text_constraint": 0,
            "surface_constraint": 0,
            "ppl_constraint": 0,
            "feasible": 0,
        }
        total_length = len(initial_active_ids) + len(overflow_ids)
        maximum_changes = math.floor(total_length * self.max_edit_ratio)
        current_changes = sum(
            old != new
            for old, new in zip(initial_active_ids, current_ids, strict=True)
        )
        if current_changes >= maximum_changes:
            filter_stats["edit_constraint"] = 1
            return [], [], filter_stats

        protected = self._protected_positions(initial_active_ids)
        editable = [
            index
            for index in range(len(current_ids))
            if index not in protected
        ]
        editable.sort(
            key=lambda index: float(
                combined_gradient[index].float().norm().item()
            ),
            reverse=True,
        )
        selected_positions = editable[: self.candidate_positions]
        raw = hotflip_top_candidates(
            self.reference.model.get_input_embeddings().weight,
            combined_gradient,
            current_ids,
            selected_positions,
            candidates_per_position=self.candidates_per_position,
            forbidden_token_ids=set(
                self.reference.tokenizer.all_special_ids
            ),
        )
        filter_stats["hotflip_raw"] = len(raw)

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, ...]] = set()
        for item in raw:
            if item.linear_gain <= 0:
                filter_stats["nonpositive_linear_gain"] += 1
                continue
            source_token = self.reference.tokenizer.decode(
                [item.source_token_id],
                skip_special_tokens=False,
            )
            candidate_token = self.reference.tokenizer.decode(
                [item.candidate_token_id],
                skip_special_tokens=False,
            )
            if self.enforce_surface_compatibility and not token_surfaces_compatible(
                source_token,
                candidate_token,
            ):
                filter_stats["surface_constraint"] += 1
                continue
            ids = list(current_ids)
            ids[item.position] = item.candidate_token_id
            key = tuple(ids)
            if key in seen:
                filter_stats["duplicate"] += 1
                continue
            seen.add(key)
            changes = sum(
                old != new
                for old, new in zip(
                    initial_active_ids,
                    ids,
                    strict=True,
                )
            )
            edit_ratio = changes / max(1, total_length)
            if changes > maximum_changes or edit_ratio > self.max_edit_ratio:
                filter_stats["edit_constraint"] += 1
                continue
            valid, decoded, reason = self._valid_discrete_text(
                initial_prompt=initial_prompt,
                active_ids=ids,
                overflow_ids=overflow_ids,
            )
            if not valid:
                filter_stats["text_constraint"] += 1
                continue
            ppl = (
                self._perplexity(ids + overflow_ids)
                if self.enforce_perplexity
                else initial_ppl
            )
            ppl_ratio = ppl / max(initial_ppl, 1e-12)
            if (
                not math.isfinite(ppl)
                or ppl_ratio > self.ppl_ratio_limit
            ):
                filter_stats["ppl_constraint"] += 1
                continue
            candidates.append(
                {
                    "token_ids": ids,
                    "position": item.position,
                    "source_token_id": item.source_token_id,
                    "candidate_token_id": item.candidate_token_id,
                    "source_token": source_token,
                    "candidate_token": candidate_token,
                    "linear_gain": item.linear_gain,
                    "decoded_prompt": decoded,
                    "edit_ratio": edit_ratio,
                    "ppl": ppl,
                    "ppl_ratio": ppl_ratio,
                    "filter_reason": reason,
                }
            )
        candidates.sort(
            key=lambda item: float(item["linear_gain"]),
            reverse=True,
        )
        filter_stats["feasible"] = len(candidates)
        return candidates, selected_positions, filter_stats

    def _rerank_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        samples: list[VariantSample],
        current_embeddings: torch.Tensor,
        prefix_ids: list[int],
        suffix_ids: list[int],
        overflow_ids: list[int],
        block: Any,
        block_type: str,
        probe_seed: int,
        baseline_micro_normalized: float,
        initial_ppl: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected = select_position_diverse_candidates(
            candidates,
            limit=self.rerank_candidates,
        )
        if not selected:
            return [], {
                "micro_score_normalized": baseline_micro_normalized,
                "macro_scores": {},
                "macro_variant_scores": {},
                "proxy_objective": math.nan,
            }

        candidate_embeddings = [
            self._embed_user(item["token_ids"]) for item in selected
        ]
        for item, embeddings in zip(
            selected,
            candidate_embeddings,
            strict=True,
        ):
            full, mask, _ = self._compose(
                prefix_ids,
                embeddings,
                suffix_ids,
            )
            micro = score_block_micro_proxy(
                self.reference,
                full.detach(),
                mask,
                block=block,
                probes=self.probes,
                seed=probe_seed,
            ) if self.micro_weight != 0 else None
            micro_normalized = (
                micro.raw_micro_score / self.micro_scales[block_type] if micro is not None else 0.0
            )
            item["micro_score_raw"] = micro.raw_micro_score if micro is not None else 0.0
            item["micro_score_normalized"] = micro_normalized
            item["proxy_objective"] = (
                self.micro_weight * micro_normalized
            )
            item["macro_scores"] = {}
            item["macro_variant_scores"] = {
                family: [] for family in FAMILIES
            }
            del full, mask, micro

        evaluation_embeddings = [
            current_embeddings.detach(),
            *candidate_embeddings,
        ]
        reference_logits: list[torch.Tensor] = []
        evaluation_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for embeddings in evaluation_embeddings:
            full, mask, _ = self._compose(
                prefix_ids,
                embeddings,
                suffix_ids,
            )
            with torch.inference_mode():
                logits = self._next_logits(
                        self.reference,
                        full.detach(),
                        mask,
                    ).detach()
                reference_logits.append(
                    logits.cpu()
                    if self.sequential_model_execution
                    else logits
                )
            if self.sequential_model_execution:
                evaluation_inputs.append(
                    (full.detach().cpu(), mask.detach().cpu())
                )
            del full, mask

        baseline_macro_variant_scores: dict[
            str,
            list[dict[str, Any]],
        ] = {
            family: [] for family in FAMILIES
        }
        if self.sequential_model_execution:
            self._release_reference()
        try:
            for sample in samples:
                family = sample.family
                loaded = None
                try:
                    loaded = load_manifest_variant(
                        self.model_config,
                        sample.manifest,
                        adapter_path=sample.adapter_path,
                    )
                    self._freeze(loaded.bundle)
                    for index, embeddings in enumerate(evaluation_embeddings):
                        if self.sequential_model_execution:
                            cached_full, cached_mask = evaluation_inputs[index]
                            full = cached_full.to(loaded.bundle.device)
                            mask = cached_mask.to(loaded.bundle.device)
                        else:
                            full, mask, _ = self._compose(
                                prefix_ids,
                                embeddings,
                                suffix_ids,
                            )
                        with torch.inference_mode():
                            variant_logits = self._next_logits(
                                loaded.bundle,
                                full.detach(),
                                mask,
                            )
                            raw_score = float(
                                differentiable_macro_proxy(
                                    reference_logits[index].to(
                                        variant_logits.device
                                    ),
                                    variant_logits,
                                    proxy=self.macro_proxy,
                                    top_k=self.macro_top_k,
                                ).item()
                            )
                            normalized = (
                                raw_score
                                / self.macro_scales[family]
                                * sample.importance_correction
                            )
                            score = {
                                "raw": raw_score,
                                "normalized": normalized,
                                "variant_id": sample.variant_id,
                                "proxy": self.macro_proxy,
                            }
                            if index == 0:
                                baseline_macro_variant_scores[family].append(
                                    score
                                )
                            else:
                                selected[index - 1][
                                    "macro_variant_scores"
                                ][family].append(score)
                        del full, mask, variant_logits
                finally:
                    if loaded is not None:
                        loaded.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if self.sequential_model_execution:
                self._reload_reference()

        baseline_macro_scores: dict[str, dict[str, Any]] = {}
        baseline_objective = (
            self.micro_weight * baseline_micro_normalized
        )
        for family in FAMILIES:
            variant_scores = baseline_macro_variant_scores[family]
            if not variant_scores:
                raise RuntimeError(
                    f"Missing baseline anchor scores for family {family}"
                )
            mean_raw = sum(
                float(score["raw"]) for score in variant_scores
            ) / len(variant_scores)
            mean_normalized = sum(
                float(score["normalized"]) for score in variant_scores
            ) / len(variant_scores)
            baseline_macro_scores[family] = {
                "raw": mean_raw,
                "normalized": mean_normalized,
                "aggregation": "mean",
                "variant_count": len(variant_scores),
                "variant_ids": [
                    str(score["variant_id"]) for score in variant_scores
                ],
                "variants": variant_scores,
            }
            baseline_objective += (
                self.macro_weight
                * self.sampler.family_weights[family]
                * mean_normalized
            )

        baseline_trace = {
            "micro_score_normalized": baseline_micro_normalized,
            "macro_scores": baseline_macro_scores,
            "macro_variant_scores": baseline_macro_variant_scores,
            "proxy_objective": float(baseline_objective),
            "family_gate_aggregation": self.family_gate_aggregation,
        }

        for item in selected:
            for family in FAMILIES:
                variant_scores = item["macro_variant_scores"][family]
                if not variant_scores:
                    raise RuntimeError(
                        f"Missing candidate scores for family {family}"
                    )
                mean_raw = sum(
                    float(score["raw"]) for score in variant_scores
                ) / len(variant_scores)
                mean_normalized = sum(
                    float(score["normalized"])
                    for score in variant_scores
                ) / len(variant_scores)
                item["macro_scores"][family] = {
                    "raw": mean_raw,
                    "normalized": mean_normalized,
                    "aggregation": "mean",
                    "variant_count": len(variant_scores),
                    "variant_ids": [
                        str(score["variant_id"])
                        for score in variant_scores
                    ],
                    "variants": variant_scores,
                }
                item["proxy_objective"] += (
                    self.macro_weight
                    * self.sampler.family_weights[family]
                    * mean_normalized
                )
            ppl = float(item["ppl"])
            if self.family_gate_aggregation == "all_anchor_variants":
                nondegraded_families, family_checks = (
                    family_anchor_nondegradation_checks(
                        baseline_macro_variant_scores,
                        item["macro_variant_scores"],
                        relative_tolerance=(
                            self.family_relative_tolerance
                        ),
                    )
                )
            else:
                nondegraded_families, family_checks = (
                    family_nondegradation_checks(
                        {
                            family: float(
                                baseline_macro_scores[family][
                                    "normalized"
                                ]
                            )
                            for family in FAMILIES
                        },
                        {
                            family: float(
                                item["macro_scores"][family][
                                    "normalized"
                                ]
                            )
                            for family in FAMILIES
                        },
                        relative_tolerance=(
                            self.family_relative_tolerance
                        ),
                    )
                )
            item["training_family_checks"] = family_checks
            item["training_nondegraded_families"] = (
                nondegraded_families
            )
            item["proxy_gain"] = (
                float(item["proxy_objective"])
                - float(baseline_objective)
            )
            item["constraints_passed"] = bool(
                math.isfinite(item["proxy_objective"])
                and math.isfinite(ppl)
                and item["ppl_ratio"] <= self.ppl_ratio_limit
                and item["edit_ratio"] <= self.max_edit_ratio
                and nondegraded_families
                >= self.minimum_nondegraded_families
            )

        selected.sort(
            key=lambda item: float(item["proxy_gain"]),
            reverse=True,
        )
        del (
            reference_logits,
            candidate_embeddings,
            evaluation_embeddings,
            evaluation_inputs,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return selected, baseline_trace

    def optimize(
        self,
        row: dict[str, Any],
    ) -> DiscreteJointOptimizationResult:
        prompt_id = str(row.get("id", row.get("prompt_id")))
        prompt = str(row["prompt"])
        (
            prefix_ids,
            initial_active_ids,
            suffix_ids,
            overflow_ids,
        ) = self._chat_parts(prompt)
        current_ids = list(initial_active_ids)
        current_embeddings = self._embed_user(current_ids)
        initial_ppl = (
            self._perplexity(current_ids + overflow_ids)
            if self.enforce_perplexity
            else 1.0
        )
        initial_task = self._task_validation(row, [prompt])[0]
        if (
            self.require_task_preservation
            and not initial_task["task_passed"]
        ):
            return DiscreteJointOptimizationResult(
                prompt_id=prompt_id,
                initial_prompt=prompt,
                optimized_prompt=prompt,
                accepted=False,
                acceptance_stage="initial_task_validation_failed",
                requires_hard_validation=False,
                initial_ppl=initial_ppl,
                final_ppl=initial_ppl,
                ppl_ratio=1.0,
                edit_ratio=0.0,
                edit_count=0,
                rounds_requested=self.rounds,
                rounds_completed=0,
                committed_rounds=0,
                proxy_objective_gain=0.0,
                active_user_tokens=len(initial_active_ids),
                preserved_overflow_tokens=len(overflow_ids),
                optimization_max_length=self.max_length,
                micro_weight=self.micro_weight,
                macro_weight=self.macro_weight,
                initial_task_passed=False,
                final_task_passed=False,
                history=[
                    {
                        "stage": "initial_task_validation",
                        **initial_task,
                    }
                ],
                failure=None,
            )
        history: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        committed_rounds = 0
        total_proxy_gain = 0.0

        for round_index in range(self.rounds):
            block_type = self.block_types[
                round_index % len(self.block_types)
            ]
            layer_id = self.layers[round_index % len(self.layers)]
            if self.block_schedule == "balanced":
                from .stage2_contract import balanced_block_schedule
                block_type, layer_id = balanced_block_schedule(
                    self.block_types, self.layers, self.rounds, prompt_id
                )[round_index]
            probe_seed = self.seed + round_index * 1000
            round_trace: dict[str, Any] = {
                "round": round_index + 1,
                "block_type": block_type,
                "layer_id": layer_id,
                "probe_seed": probe_seed,
                "gradient_variants_per_family": (
                    self.variants_per_family
                ),
                "anchor_variants_per_family": (
                    self.anchor_variants_per_family
                ),
                "gradient_restarts": self.gradient_restarts,
                "family_gate_aggregation": (
                    self.family_gate_aggregation
                ),
                "state_token_ids_before": list(current_ids),
                "committed": False,
            }
            try:
                (
                    micro_gradient,
                    micro_objective,
                    micro_trace,
                    block,
                ) = self._micro_gradient(
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    current_embeddings=current_embeddings,
                    block_type=block_type,
                    layer_id=layer_id,
                    probe_seed=probe_seed,
                )
                candidate_pools: list[list[dict[str, Any]]] = []
                macro_traces: list[dict[str, Any]] = []
                gradient_samples: list[VariantSample] = []
                anchor_samples: list[VariantSample] = []
                anchor_variant_ids: list[str] | None = None
                restart_details: list[dict[str, Any]] = []
                filter_stats_by_restart: list[dict[str, int]] = []
                selected_positions_by_restart: list[list[int]] = []
                gradient_baseline_objectives: list[float] = []
                for restart_index in range(self.gradient_restarts):
                    variant_cycle = (
                        round_index + restart_index * self.rounds
                    )
                    (
                        macro_gradient,
                        macro_objective,
                        restart_macro_traces,
                        restart_gradient_samples,
                        restart_anchor_samples,
                    ) = self._macro_gradient(
                        prompt_id=prompt_id,
                        round_index=variant_cycle,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        current_embeddings=current_embeddings,
                    )
                    restart_anchor_ids = [
                        sample.variant_id
                        for sample in restart_anchor_samples
                    ]
                    if anchor_variant_ids is None:
                        anchor_variant_ids = restart_anchor_ids
                        anchor_samples = restart_anchor_samples
                    elif restart_anchor_ids != anchor_variant_ids:
                        raise RuntimeError(
                            "Anchor variants changed across gradient restarts"
                        )
                    for trace in restart_macro_traces:
                        trace["gradient_restart"] = restart_index + 1
                        trace["variant_cycle"] = variant_cycle
                    macro_traces.extend(restart_macro_traces)
                    gradient_samples.extend(restart_gradient_samples)
                    combined = (
                        self.micro_weight * micro_gradient
                        + self.macro_weight * macro_gradient
                    )
                    gradient_baseline_objective = (
                        self.micro_weight * micro_objective
                        + self.macro_weight * macro_objective
                    )
                    (
                        restart_candidates,
                        restart_positions,
                        restart_filter_stats,
                    ) = self._candidate_pool(
                        initial_prompt=prompt,
                        initial_active_ids=initial_active_ids,
                        current_ids=current_ids,
                        overflow_ids=overflow_ids,
                        combined_gradient=combined,
                        initial_ppl=initial_ppl,
                    )
                    candidate_pools.append(restart_candidates)
                    selected_positions_by_restart.append(restart_positions)
                    filter_stats_by_restart.append(restart_filter_stats)
                    gradient_baseline_objectives.append(
                        gradient_baseline_objective
                    )
                    restart_details.append(
                        {
                            "restart": restart_index + 1,
                            "variant_cycle": variant_cycle,
                            "gradient_variant_ids": [
                                sample.variant_id
                                for sample in restart_gradient_samples
                            ],
                            "combined_gradient_norm": float(
                                combined.float().norm().item()
                            ),
                            "gradient_baseline_proxy_objective": (
                                gradient_baseline_objective
                            ),
                            "selected_positions": restart_positions,
                            "generated_candidates": len(
                                restart_candidates
                            ),
                            "candidate_filter_stats": (
                                restart_filter_stats
                            ),
                        }
                    )
                    del macro_gradient, combined
                candidates = merge_restart_candidate_pools(candidate_pools)
                selected_positions = sorted(
                    {
                        position
                        for positions in selected_positions_by_restart
                        for position in positions
                    }
                )
                candidate_filter_stats = {
                    key: sum(stats.get(key, 0) for stats in filter_stats_by_restart)
                    for key in {
                        key
                        for stats in filter_stats_by_restart
                        for key in stats
                    }
                }
                candidate_filter_stats["unique_feasible"] = len(candidates)
                gradient_baseline_objective = sum(
                    gradient_baseline_objectives
                ) / len(gradient_baseline_objectives)
                reranked, rerank_baseline = self._rerank_candidates(
                    candidates=candidates,
                    samples=anchor_samples,
                    current_embeddings=current_embeddings,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    overflow_ids=overflow_ids,
                    block=block,
                    block_type=block_type,
                    probe_seed=probe_seed,
                    baseline_micro_normalized=micro_objective,
                    initial_ppl=initial_ppl,
                )
                eligible_for_task = [
                    item
                    for item in reranked
                    if item["constraints_passed"]
                    and item["proxy_gain"]
                    > self.minimum_proxy_gain
                ]
                if eligible_for_task:
                    task_results = self._task_validation(
                        row,
                        [
                            str(item["decoded_prompt"])
                            for item in eligible_for_task
                        ],
                    )
                    for item, task_result in zip(
                        eligible_for_task,
                        task_results,
                        strict=True,
                    ):
                        item["task_validation"] = task_result
                        item["constraints_passed"] = bool(
                            item["constraints_passed"]
                            and task_result["task_passed"]
                        )
                accepted_candidate = next(
                    (
                        item
                        for item in reranked
                        if item["constraints_passed"]
                        and item["proxy_gain"]
                        > self.minimum_proxy_gain
                    ),
                    None,
                )
                round_trace.update(
                    {
                        "micro": micro_trace,
                        "macro": macro_traces,
                        "gradient_variant_ids": [
                            sample.variant_id
                            for sample in gradient_samples
                        ],
                        "anchor_variant_ids": [
                            sample.variant_id
                            for sample in anchor_samples
                        ],
                        "rerank_baseline": rerank_baseline,
                        "micro_weight": self.micro_weight,
                        "macro_weight": self.macro_weight,
                        "gradient_baseline_proxy_objective": (
                            gradient_baseline_objective
                        ),
                        "gradient_restart_details": restart_details,
                        "baseline_proxy_objective": (
                            rerank_baseline["proxy_objective"]
                        ),
                        "combined_gradient_norm": sum(
                            item["combined_gradient_norm"]
                            for item in restart_details
                        ) / len(restart_details),
                        "selected_positions": selected_positions,
                        "generated_candidates": len(candidates),
                        "candidate_filter_stats": (
                            candidate_filter_stats
                        ),
                        "reranked_candidates": [
                            {
                                key: value
                                for key, value in item.items()
                                if key != "token_ids"
                            }
                            for item in reranked
                        ],
                    }
                )
                if accepted_candidate is not None:
                    current_ids = list(
                        accepted_candidate["token_ids"]
                    )
                    # This is the discrete commit missing from the old
                    # diagnostic projection path.
                    current_embeddings = self._embed_user(current_ids)
                    committed_rounds += 1
                    total_proxy_gain += float(
                        accepted_candidate["proxy_gain"]
                    )
                    round_trace.update(
                        {
                            "committed": True,
                            "committed_position": (
                                accepted_candidate["position"]
                            ),
                            "committed_source_token_id": (
                                accepted_candidate["source_token_id"]
                            ),
                            "committed_candidate_token_id": (
                                accepted_candidate[
                                    "candidate_token_id"
                                ]
                            ),
                            "committed_source_token": (
                                accepted_candidate["source_token"]
                            ),
                            "committed_candidate_token": (
                                accepted_candidate["candidate_token"]
                            ),
                            "committed_proxy_gain": (
                                accepted_candidate["proxy_gain"]
                            ),
                            "committed_task_validation": (
                                accepted_candidate.get(
                                    "task_validation"
                                )
                            ),
                            "state_token_ids_after": list(current_ids),
                        }
                    )
                else:
                    round_trace["state_token_ids_after"] = list(
                        current_ids
                    )
                history.append(round_trace)
                del (
                    micro_gradient,
                    candidate_pools,
                    candidates,
                    reranked,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                failure = {
                    "round": round_index + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                round_trace["failure"] = failure
                history.append(round_trace)
                break

        edit_count = sum(
            old != new
            for old, new in zip(
                initial_active_ids,
                current_ids,
                strict=True,
            )
        )
        edit_ratio = edit_count / max(
            1,
            len(initial_active_ids) + len(overflow_ids),
        )
        final_ppl = (
            self._perplexity(current_ids + overflow_ids)
            if self.enforce_perplexity
            else initial_ppl
        )
        ppl_ratio = final_ppl / max(initial_ppl, 1e-12)
        valid_text, decoded, _ = self._valid_discrete_text(
            initial_prompt=prompt,
            active_ids=current_ids,
            overflow_ids=overflow_ids,
        )
        accepted = bool(
            failure is None
            and committed_rounds > 0
            and edit_count > 0
            and total_proxy_gain > self.minimum_proxy_gain
            and valid_text
            and edit_ratio <= self.max_edit_ratio
            and ppl_ratio <= self.ppl_ratio_limit
        )
        final_task_passed = resolve_final_task_status(
            require_task_preservation=self.require_task_preservation,
            initial_task_passed=bool(initial_task["task_passed"]),
            accepted=accepted,
            history=history,
        )
        optimized_prompt = decoded if accepted else prompt
        if not accepted:
            final_ppl = initial_ppl
            ppl_ratio = 1.0
            edit_ratio = 0.0
            edit_count = 0

        return DiscreteJointOptimizationResult(
            prompt_id=prompt_id,
            initial_prompt=prompt,
            optimized_prompt=optimized_prompt,
            accepted=accepted,
            acceptance_stage="proxy_search_pending_hard_validation",
            requires_hard_validation=True,
            initial_ppl=initial_ppl,
            final_ppl=final_ppl,
            ppl_ratio=ppl_ratio,
            edit_ratio=edit_ratio,
            edit_count=edit_count,
            rounds_requested=self.rounds,
            rounds_completed=len(history),
            committed_rounds=committed_rounds if accepted else 0,
            proxy_objective_gain=(
                total_proxy_gain if accepted else 0.0
            ),
            active_user_tokens=len(initial_active_ids),
            preserved_overflow_tokens=len(overflow_ids),
            optimization_max_length=self.max_length,
            micro_weight=self.micro_weight,
            macro_weight=self.macro_weight,
            initial_task_passed=bool(initial_task["task_passed"]),
            final_task_passed=final_task_passed,
            history=history,
            failure=failure,
        )
