from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .inner_micro_proxy import (
    differentiable_block_micro_proxy,
    discover_micro_blocks,
    representative_layers,
    select_micro_block,
)
from .inner_variant_sampler import StratifiedVariantSampler
from .modeling import ModelBundle
from .paper_variant_executor import load_manifest_variant


@dataclass
class JointOptimizationResult:
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
    steps_requested: int
    steps_completed: int
    active_user_tokens: int
    preserved_overflow_tokens: int
    optimization_max_length: int
    learning_rate: float
    global_clip_norm: float
    maximum_relative_update: float
    clip_mode: str
    top_token_updates: int | None
    projection_interval: int
    micro_weight: float
    micro_scale: float
    macro_scale: float
    history: list[dict[str, Any]]
    failure: dict[str, Any] | None

    def payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }


class JointInnerOptimizer:
    def __init__(
        self,
        *,
        reference: ModelBundle,
        sampler: StratifiedVariantSampler,
        model_config: dict[str, Any],
        micro_scale: float,
        macro_scale: float,
        micro_weight: float,
        macro_weight: float = 1.0,
        micro_component_clip: float | None = None,
        macro_component_clip: float | None = None,
        global_clip_norm: float = 1.0,
        learning_rate: float = 0.01,
        maximum_relative_update: float = 0.001,
        clip_mode: str = "global",
        top_token_updates: int | None = None,
        projection_interval: int = 15,
        probes: int = 1,
        seed: int = 42,
        max_length: int = 64,
        max_edit_ratio: float = 0.25,
        ppl_ratio_limit: float = 2.0,
        block_types: tuple[str, ...] = ("q_proj", "v_proj", "down_proj"),
        representative_layer_count: int = 4,
    ) -> None:
        self.reference = reference
        self.sampler = sampler
        self.model_config = dict(model_config)
        self.micro_scale = float(micro_scale)
        self.macro_scale = float(macro_scale)
        self.micro_weight = float(micro_weight)
        self.macro_weight = float(macro_weight)
        self.micro_component_clip = micro_component_clip
        self.macro_component_clip = macro_component_clip
        self.global_clip_norm = float(global_clip_norm)
        self.learning_rate = float(learning_rate)
        self.maximum_relative_update = float(maximum_relative_update)
        if clip_mode not in {"global", "tokenwise"}:
            raise ValueError("clip_mode must be 'global' or 'tokenwise'")
        self.clip_mode = clip_mode
        self.top_token_updates = (
            None if top_token_updates is None else int(top_token_updates)
        )
        if self.top_token_updates is not None and self.top_token_updates <= 0:
            raise ValueError("top_token_updates must be positive")
        self.projection_interval = int(projection_interval)
        if self.projection_interval <= 0:
            raise ValueError("projection_interval must be positive")
        self.probes = int(probes)
        self.seed = int(seed)
        self.max_length = int(max_length)
        self.max_edit_ratio = float(max_edit_ratio)
        self.ppl_ratio_limit = float(ppl_ratio_limit)
        self.block_types = tuple(block_types)
        self.representative_layer_count = int(representative_layer_count)
        self.blocks = discover_micro_blocks(reference.model, self.block_types)
        self.layers = representative_layers(self.blocks, self.representative_layer_count)
        self._freeze(reference)

    @staticmethod
    def _freeze(bundle: ModelBundle) -> None:
        bundle.model.eval()
        for parameter in bundle.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

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
            full_ids = list(user_ids)
        start = self._find_subsequence(full_ids, full_user_ids)
        if start < 0:
            raise ValueError("Could not locate user tokens in chat template")
        end = start + len(full_user_ids)
        prefix = full_ids[:start]
        suffix = full_ids[end:]
        allowed_user = max(1, self.max_length - len(prefix) - len(suffix))
        active_user_ids = full_user_ids[:allowed_user]
        overflow_user_ids = full_user_ids[allowed_user:]
        return prefix, active_user_ids, suffix, overflow_user_ids

    @staticmethod
    def _find_subsequence(sequence: list[int], target: list[int]) -> int:
        for index in range(max(0, len(sequence) - len(target) + 1)):
            if sequence[index : index + len(target)] == target:
                return index
        return -1

    def _compose(
        self,
        prefix_ids: list[int],
        user_embeddings: torch.Tensor,
        suffix_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, slice]:
        layer = self.reference.model.get_input_embeddings()
        parts: list[torch.Tensor] = []
        if prefix_ids:
            ids = torch.tensor(
                [prefix_ids],
                dtype=torch.long,
                device=self.reference.device,
            )
            parts.append(layer(ids).detach().float())
        user_start = sum(part.shape[1] for part in parts)
        parts.append(user_embeddings.unsqueeze(0))
        user_stop = user_start + user_embeddings.shape[0]
        if suffix_ids:
            ids = torch.tensor(
                [suffix_ids],
                dtype=torch.long,
                device=self.reference.device,
            )
            parts.append(layer(ids).detach().float())
        full = torch.cat(parts, dim=1).detach().requires_grad_(True)
        mask = torch.ones(
            full.shape[:2],
            dtype=torch.long,
            device=full.device,
        )
        return full, mask, slice(user_start, user_stop)

    @staticmethod
    def _next_logits(bundle, embeddings, mask):
        dtype = bundle.model.get_input_embeddings().weight.dtype
        output = bundle.model(
            inputs_embeds=embeddings.to(dtype),
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits[:, -1, :].float()
        if logits.shape[0] != 1:
            raise ValueError("Prompt optimization expects batch size one")
        return logits[0]

    @staticmethod
    def _clip_component(
        gradient: torch.Tensor,
        threshold: float | None,
    ) -> tuple[torch.Tensor, float, float]:
        norm = float(gradient.float().norm().item())
        if threshold is None or threshold <= 0 or norm <= threshold:
            return gradient, norm, 1.0
        coefficient = float(threshold / (norm + 1e-12))
        return gradient * coefficient, norm, coefficient

    def _perplexity(self, user_ids: list[int]) -> float:
        if len(user_ids) < 2:
            return float("inf")
        ids = torch.tensor(
            [user_ids],
            dtype=torch.long,
            device=self.reference.device,
        )
        with torch.inference_mode():
            loss = self.reference.model(
                input_ids=ids,
                labels=ids,
                use_cache=False,
            ).loss
        return float(math.exp(min(float(loss.item()), 20.0)))

    def _project(
        self,
        original_ids: list[int],
        initial_embeddings: torch.Tensor,
        optimized_embeddings: torch.Tensor,
        *,
        edit_denominator: int,
    ) -> tuple[list[int], float]:
        matrix = self.reference.model.get_input_embeddings().weight.detach()
        optimized = F.normalize(optimized_embeddings.float(), dim=-1)
        best_scores = torch.full(
            (optimized.shape[0],),
            -float("inf"),
            device=optimized.device,
        )
        best_ids = torch.tensor(
            original_ids,
            dtype=torch.long,
            device=optimized.device,
        )
        special_ids = set(self.reference.tokenizer.all_special_ids)
        for start in range(0, matrix.shape[0], 4096):
            stop = min(start + 4096, matrix.shape[0])
            candidates = F.normalize(matrix[start:stop].float(), dim=-1)
            scores = (optimized @ candidates.T)
            for special_id in special_ids:
                if start <= special_id < stop:
                    scores[:, special_id - start] = -float("inf")
            chunk_scores, chunk_positions = scores.max(dim=1)
            replace = chunk_scores > best_scores
            best_scores = torch.where(replace, chunk_scores, best_scores)
            best_ids = torch.where(
                replace,
                chunk_positions + start,
                best_ids,
            )
        proposed = best_ids.tolist()
        protected = {
            index
            for index, token_id in enumerate(original_ids)
            if any(
                character.isdigit()
                for character in self.reference.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
            )
        }
        movement = (
            optimized_embeddings.float() - initial_embeddings.float()
        ).norm(dim=-1)
        changed = [
            index
            for index, (old, new) in enumerate(zip(original_ids, proposed))
            if old != new and index not in protected
        ]
        maximum_changes = max(
            0,
            math.floor(len(original_ids) * self.max_edit_ratio),
        )
        keep_changes = set(
            sorted(
                changed,
                key=lambda index: float(movement[index].item()),
                reverse=True,
            )[:maximum_changes]
        )
        final_ids = [
            proposed[index]
            if index in keep_changes
            else original_ids[index]
            for index in range(len(original_ids))
        ]
        edit_ratio = sum(
            old != new for old, new in zip(original_ids, final_ids)
        ) / max(1, edit_denominator)
        return final_ids, float(edit_ratio)

    def _protected_positions(self, token_ids: list[int]) -> set[int]:
        return {
            index
            for index, token_id in enumerate(token_ids)
            if any(
                character.isdigit()
                for character in self.reference.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
            )
        }

    def _token_boundary_distances(
        self,
        token_ids: list[int],
    ) -> torch.Tensor:
        """Return the exact raw distance to each cosine Voronoi boundary."""
        matrix = self.reference.model.get_input_embeddings().weight.detach()
        normalized = F.normalize(matrix.float(), dim=-1)
        special_ids = set(self.reference.tokenizer.all_special_ids)
        unique: dict[int, float] = {}
        for token_id in sorted(set(token_ids)):
            source = matrix[token_id].float()
            source_normalized = normalized[token_id]
            best_other = -float("inf")
            for start in range(0, matrix.shape[0], 4096):
                stop = min(start + 4096, matrix.shape[0])
                scores = normalized[start:stop] @ source_normalized
                if start <= token_id < stop:
                    scores[token_id - start] = -float("inf")
                for special_id in special_ids:
                    if start <= special_id < stop:
                        scores[special_id - start] = -float("inf")
                best_other = max(best_other, float(scores.max().item()))
            unique[token_id] = float(
                source.norm().item()
                * math.sqrt(max(0.0, (1.0 - best_other) / 2.0))
            )
        return torch.tensor(
            [unique[token_id] for token_id in token_ids],
            dtype=torch.float32,
            device=self.reference.device,
        )

    def optimize(self, row: dict[str, Any]) -> JointOptimizationResult:
        prompt_id = str(row.get("id", row.get("prompt_id")))
        prompt = str(row["prompt"])
        (
            prefix_ids,
            user_ids,
            suffix_ids,
            overflow_user_ids,
        ) = self._chat_parts(prompt)
        embedding_layer = self.reference.model.get_input_embeddings()
        ids_tensor = torch.tensor(
            user_ids,
            dtype=torch.long,
            device=self.reference.device,
        )
        with torch.no_grad():
            initial_user = embedding_layer(ids_tensor).detach().float()
        current_user = initial_user.clone()
        boundary_distances = self._token_boundary_distances(user_ids)
        protected_positions = self._protected_positions(user_ids)
        initial_ppl = self._perplexity(user_ids)
        history: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        schedule = self.sampler.balanced_schedule(prompt_id)

        for family_index, family in enumerate(
            self.sampler.family_order(prompt_id)
        ):
            family_steps = [
                item for item in schedule
                if item["sample"].family == family
            ]
            sample = family_steps[0]["sample"]
            loaded = None
            try:
                loaded = load_manifest_variant(
                    self.model_config,
                    sample.manifest,
                    adapter_path=sample.adapter_path,
                )
                self._freeze(loaded.bundle)
                for local_index, item in enumerate(family_steps):
                    step_index = len(history) + 1
                    full, mask, user_slice = self._compose(
                        prefix_ids,
                        current_user,
                        suffix_ids,
                    )
                    layer_id = self.layers[
                        (step_index - 1) % len(self.layers)
                    ]
                    block = select_micro_block(
                        self.blocks,
                        block_type=item["block_type"],
                        layer_id=layer_id,
                    )
                    if self.micro_weight > 0:
                        micro = differentiable_block_micro_proxy(
                            self.reference,
                            full,
                            mask,
                            block=block,
                            probes=self.probes,
                            seed=self.seed + step_index,
                        )
                        micro_gradient = (
                            micro.embedding_gradient[0, user_slice, :]
                            / self.micro_scale
                        )
                        micro_gradient, micro_norm_before_clip, micro_clip = (
                            self._clip_component(
                                micro_gradient,
                                self.micro_component_clip,
                            )
                        )
                        micro_score_raw = micro.raw_micro_score
                    else:
                        micro = None
                        micro_gradient = torch.zeros_like(current_user)
                        micro_norm_before_clip = 0.0
                        micro_clip = 1.0
                        micro_score_raw = None
                    del full, mask
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    full, mask, user_slice = self._compose(
                        prefix_ids,
                        current_user,
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
                    macro_score = (
                        variant_logits - reference_logits
                    ).square().sum()
                    macro_full_gradient = torch.autograd.grad(
                        macro_score,
                        full,
                    )[0].detach().float()
                    macro_gradient = (
                        macro_full_gradient[0, user_slice, :]
                        / self.macro_scale
                        * sample.importance_correction
                    )
                    macro_gradient, macro_norm_before_clip, macro_clip = (
                        self._clip_component(
                            macro_gradient,
                            self.macro_component_clip,
                        )
                    )
                    weighted_micro = self.micro_weight * micro_gradient
                    weighted_macro = self.macro_weight * macro_gradient
                    combined = weighted_micro + weighted_macro
                    if not torch.isfinite(weighted_micro).all():
                        raise FloatingPointError(
                            "Non-finite normalized micro gradient"
                        )
                    if not torch.isfinite(weighted_macro).all():
                        raise FloatingPointError(
                            "Non-finite normalized macro gradient"
                        )
                    if not torch.isfinite(combined).all():
                        raise FloatingPointError(
                            "Non-finite combined gradient"
                        )
                    combined_norm_before = float(combined.norm().item())
                    per_token_gradient_norms = combined.float().norm(dim=-1)
                    selected_token_indices: list[int]
                    token_clip_min_coefficient = 1.0
                    if self.clip_mode == "global":
                        global_coefficient = min(
                            1.0,
                            self.global_clip_norm
                            / (combined_norm_before + 1e-12),
                        )
                        final_gradient = combined * global_coefficient
                        selected_token_indices = list(
                            range(combined.shape[0])
                        )
                    else:
                        editable = [
                            index
                            for index in range(combined.shape[0])
                            if index not in protected_positions
                        ]
                        editable.sort(
                            key=lambda index: float(
                                per_token_gradient_norms[index].item()
                            ),
                            reverse=True,
                        )
                        if self.top_token_updates is not None:
                            editable = editable[: self.top_token_updates]
                        selected_token_indices = editable
                        final_gradient = torch.zeros_like(combined)
                        coefficients: list[float] = []
                        for index in editable:
                            norm = float(
                                per_token_gradient_norms[index].item()
                            )
                            coefficient = min(
                                1.0,
                                self.global_clip_norm / (norm + 1e-12),
                            )
                            coefficients.append(coefficient)
                            final_gradient[index] = (
                                combined[index] * coefficient
                            )
                        token_clip_min_coefficient = (
                            min(coefficients) if coefficients else 1.0
                        )
                        global_coefficient = 1.0
                    proposed_delta = self.learning_rate * final_gradient
                    relative_delta = float(
                        proposed_delta.norm().item()
                        / (current_user.norm().item() + 1e-12)
                    )
                    trust_coefficient = min(
                        1.0,
                        self.maximum_relative_update
                        / (relative_delta + 1e-12),
                    )
                    delta = proposed_delta * trust_coefficient
                    current_user = (
                        current_user + delta
                    ).detach().float()
                    cumulative_token_movement = (
                        current_user - initial_user
                    ).float().norm(dim=-1)
                    boundary_ratio = cumulative_token_movement / (
                        boundary_distances + 1e-12
                    )
                    maximum_boundary_ratio, maximum_boundary_position = (
                        boundary_ratio.max(dim=0)
                    )
                    projection_checked = bool(
                        step_index % self.projection_interval == 0
                        or step_index == 15
                    )
                    projection_changed_tokens = 0
                    projection_edit_ratio = 0.0
                    if projection_checked:
                        projected_ids, projection_edit_ratio = self._project(
                            user_ids,
                            initial_user,
                            current_user,
                            edit_denominator=(
                                len(user_ids) + len(overflow_user_ids)
                            ),
                        )
                        projection_changed_tokens = sum(
                            old != new
                            for old, new in zip(user_ids, projected_ids)
                        )
                    cosine = float(
                        F.cosine_similarity(
                            weighted_micro.reshape(1, -1),
                            weighted_macro.reshape(1, -1),
                        ).item()
                    )
                    history.append(
                        {
                            "step": step_index,
                            **sample.trace(),
                            "block_type": item["block_type"],
                            "layer_id": layer_id,
                            "micro_score_raw": micro_score_raw,
                            "macro_score_raw": float(
                                macro_score.detach().item()
                            ),
                            "micro_gradient_norm_normalized": (
                                micro_norm_before_clip
                            ),
                            "macro_gradient_norm_normalized": (
                                macro_norm_before_clip
                            ),
                            "micro_component_clip_coefficient": micro_clip,
                            "macro_component_clip_coefficient": macro_clip,
                            "weighted_gradient_cosine": cosine,
                            "combined_gradient_norm_before_clip": (
                                combined_norm_before
                            ),
                            "clip_mode": self.clip_mode,
                            "global_clip_coefficient": global_coefficient,
                            "token_clip_min_coefficient": (
                                token_clip_min_coefficient
                            ),
                            "selected_token_indices": (
                                selected_token_indices
                            ),
                            "per_token_gradient_norm_max": float(
                                per_token_gradient_norms.max().item()
                            ),
                            "relative_update_before_trust_region": (
                                relative_delta
                            ),
                            "trust_region_coefficient": trust_coefficient,
                            "actual_embedding_delta_norm": float(
                                delta.norm().item()
                            ),
                            "actual_token_delta_norm_max": float(
                                delta.float().norm(dim=-1).max().item()
                            ),
                            "cumulative_net_embedding_delta_norm": float(
                                (current_user - initial_user).norm().item()
                            ),
                            "cumulative_token_movement_max": float(
                                cumulative_token_movement.max().item()
                            ),
                            "maximum_boundary_ratio": float(
                                maximum_boundary_ratio.item()
                            ),
                            "maximum_boundary_position": int(
                                maximum_boundary_position.item()
                            ),
                            "minimum_boundary_distance": float(
                                boundary_distances.min().item()
                            ),
                            "projection_checked": projection_checked,
                            "projection_changed_tokens": (
                                projection_changed_tokens
                            ),
                            "projection_edit_ratio": (
                                projection_edit_ratio
                            ),
                            "gradient_finite": bool(
                                torch.isfinite(final_gradient).all().item()
                            ),
                        }
                    )
                    del (
                        full,
                        mask,
                        reference_logits,
                        variant_logits,
                        macro_score,
                        macro_full_gradient,
                        micro_gradient,
                        macro_gradient,
                        weighted_micro,
                        weighted_macro,
                        combined,
                        final_gradient,
                        proposed_delta,
                        delta,
                    )
                    if micro is not None:
                        del micro
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:
                failure = {
                    "family": family,
                    "step": len(history) + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                break
            finally:
                if loaded is not None:
                    loaded.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        optimized_ids, edit_ratio = self._project(
            user_ids,
            initial_user,
            current_user,
            edit_denominator=len(user_ids) + len(overflow_user_ids),
        )
        complete_optimized_ids = optimized_ids + overflow_user_ids
        if optimized_ids == user_ids:
            # Preserve the exact source string when projection made no token
            # change.  This prevents tokenizer decode normalization and, more
            # importantly, guarantees that a memory-bound optimized prefix
            # never deletes the untouched prompt tail.
            optimized_prompt = prompt
        else:
            optimized_prompt = self.reference.tokenizer.decode(
                complete_optimized_ids,
                skip_special_tokens=True,
            )
        final_ppl = self._perplexity(optimized_ids)
        ppl_ratio = final_ppl / max(initial_ppl, 1e-12)
        accepted = bool(
            failure is None
            and len(history) == 15
            and all(item["gradient_finite"] for item in history)
            and edit_ratio <= self.max_edit_ratio
            and ppl_ratio <= self.ppl_ratio_limit
        )
        if not accepted:
            optimized_prompt = prompt
            final_ppl = initial_ppl
            ppl_ratio = 1.0
            edit_ratio = 0.0
        return JointOptimizationResult(
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
            steps_requested=15,
            steps_completed=len(history),
            active_user_tokens=len(user_ids),
            preserved_overflow_tokens=len(overflow_user_ids),
            optimization_max_length=self.max_length,
            learning_rate=self.learning_rate,
            global_clip_norm=self.global_clip_norm,
            maximum_relative_update=self.maximum_relative_update,
            clip_mode=self.clip_mode,
            top_token_updates=self.top_token_updates,
            projection_interval=self.projection_interval,
            micro_weight=self.micro_weight,
            micro_scale=self.micro_scale,
            macro_scale=self.macro_scale,
            history=history,
            failure=failure,
        )
