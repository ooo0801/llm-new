from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_integrity.distances import js_divergence_logits
from llm_integrity.modeling import ModelBundle


@dataclass
class ContinuousOptimizationResult:
    initial_prompt: str
    optimized_prompt: str
    accepted: bool
    initial_score: float
    final_score: float
    score_gain: float
    initial_ppl: float
    final_ppl: float
    ppl_ratio: float
    edit_ratio: float
    steps_run: int
    history: list[dict[str, float]]
    proxy_accepted: bool
    acceptance_stage: str
    requires_hard_validation: bool
    proxy_objective: str


class ContinuousPromptOptimizer:
    """Resource-aware implementation of proposal Eqs. (13)-(17).

    Only user-content tokens are optimized. Chat-template tokens remain fixed.
    The inner-loop objective is a differentiable proxy: next-token JS
    divergence between a reference model and one sampled modified model.
    It proposes candidates only.  Final acceptance belongs to the separate
    full micro + five-family macro hard-validation stage.
    """

    def __init__(
        self,
        reference: ModelBundle,
        modified: ModelBundle,
        candidate_tokens: int = 16,
        learning_rate: float = 0.05,
        steps: int = 10,
        initial_temperature: float = 1.0,
        anneal: float = 0.95,
        epsilon: float = 1.0,
        semantic_weight: float = 0.1,
        ppl_ratio_limit: float = 2.0,
        max_edit_ratio: float = 0.25,
        convergence_tolerance: float = 1e-4,
        vocab_chunk_size: int = 4096,
    ) -> None:
        self.reference = reference
        self.modified = modified
        self.candidate_tokens = max(2, int(candidate_tokens))
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.initial_temperature = float(initial_temperature)
        self.anneal = float(anneal)
        self.epsilon = float(epsilon)
        self.semantic_weight = float(semantic_weight)
        self.ppl_ratio_limit = float(ppl_ratio_limit)
        self.max_edit_ratio = float(max_edit_ratio)
        self.convergence_tolerance = float(convergence_tolerance)
        self.vocab_chunk_size = int(vocab_chunk_size)

        if reference.name != modified.name:
            raise ValueError("Continuous optimization requires the same base architecture")
        if reference.model.config.vocab_size != modified.model.config.vocab_size:
            raise ValueError("Reference and modified vocabularies differ")
        if str(reference.device) != str(modified.device):
            raise ValueError("Reference and modified models must use the same input device")

        for bundle in (reference, modified):
            bundle.model.eval()
            for parameter in bundle.model.parameters():
                parameter.requires_grad_(False)

    def _chat_parts(self, prompt: str) -> tuple[list[int], list[int], list[int]]:
        tokenizer = self.reference.tokenizer
        user_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not user_ids:
            raise ValueError("Prompt tokenization is empty")

        if tokenizer.chat_template:
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            full_ids = list(user_ids)

        start = self._find_subsequence(full_ids, user_ids)
        if start < 0:
            raise ValueError("Could not locate user tokens inside the chat template")
        end = start + len(user_ids)
        return list(full_ids[:start]), list(user_ids), list(full_ids[end:])

    @staticmethod
    def _find_subsequence(sequence: list[int], target: list[int]) -> int:
        limit = len(sequence) - len(target) + 1
        for index in range(max(0, limit)):
            if sequence[index : index + len(target)] == target:
                return index
        return -1

    def _nearest_candidates(self, token_ids, initial_embeddings):
        import torch
        import torch.nn.functional as F

        matrix = self.reference.model.get_input_embeddings().weight.detach()
        query = F.normalize(initial_embeddings.float(), dim=-1)
        special_ids = set(self.reference.tokenizer.all_special_ids)
        keep = self.candidate_tokens - 1
        best_scores = torch.full(
            (query.shape[0], keep),
            -float("inf"),
            device=query.device,
        )
        best_ids = torch.zeros(
            (query.shape[0], keep),
            dtype=torch.long,
            device=query.device,
        )

        for start in range(0, matrix.shape[0], self.vocab_chunk_size):
            stop = min(start + self.vocab_chunk_size, matrix.shape[0])
            chunk = F.normalize(matrix[start:stop].float(), dim=-1)
            scores = query @ chunk.T
            for special_id in special_ids:
                if start <= special_id < stop:
                    scores[:, special_id - start] = -float("inf")

            # The original token is added explicitly as candidate column 0
            # below. Exclude it here so the remaining columns contain genuine
            # replacement tokens rather than a duplicate of the original.
            original_in_chunk = (token_ids >= start) & (token_ids < stop)
            if original_in_chunk.any():
                row_ids = torch.nonzero(
                    original_in_chunk,
                    as_tuple=False,
                ).squeeze(1)
                column_ids = token_ids[row_ids] - start
                scores[row_ids, column_ids] = -float("inf")

            chunk_scores, chunk_positions = scores.topk(
                min(keep, stop - start), dim=-1
            )
            chunk_ids = chunk_positions + start
            merged_scores = torch.cat((best_scores, chunk_scores), dim=-1)
            merged_ids = torch.cat((best_ids, chunk_ids), dim=-1)
            best_scores, positions = merged_scores.topk(keep, dim=-1)
            best_ids = merged_ids.gather(1, positions)

        original = token_ids.reshape(-1, 1)
        candidate_ids = torch.cat((original, best_ids), dim=-1)
        candidate_vectors = matrix[candidate_ids].detach().float()
        return candidate_ids, candidate_vectors

    def _compose_embeddings(self, prefix_ids, user_embeddings, suffix_ids):
        import torch

        layer = self.reference.model.get_input_embeddings()
        parts = []
        if prefix_ids:
            prefix = torch.tensor(prefix_ids, device=self.reference.device).unsqueeze(0)
            parts.append(layer(prefix).detach().float())
        parts.append(user_embeddings.unsqueeze(0))
        if suffix_ids:
            suffix = torch.tensor(suffix_ids, device=self.reference.device).unsqueeze(0)
            parts.append(layer(suffix).detach().float())
        return torch.cat(parts, dim=1)

    @staticmethod
    def _model_dtype(bundle: ModelBundle):
        return bundle.model.get_input_embeddings().weight.dtype

    def _soft_divergence(self, full_embeddings):
        import torch

        attention_mask = torch.ones(
            full_embeddings.shape[:2],
            dtype=torch.long,
            device=self.reference.device,
        )
        ref_logits = self.reference.model(
            inputs_embeds=full_embeddings.to(self._model_dtype(self.reference)),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1].float()
        mod_logits = self.modified.model(
            inputs_embeds=full_embeddings.to(self._model_dtype(self.modified)),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1].float()
        return js_divergence_logits(ref_logits, mod_logits).mean()

    def _hard_divergence(self, full_ids: list[int]) -> float:
        import torch

        ids = torch.tensor(full_ids, device=self.reference.device).unsqueeze(0)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            ref = self.reference.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits[:, -1].float()
            mod = self.modified.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits[:, -1].float()
            value = js_divergence_logits(ref, mod).mean()
        return float(value.item())

    def _perplexity(self, user_ids: list[int]) -> float:
        import math
        import torch

        if len(user_ids) < 2:
            return float("inf")
        ids = torch.tensor(user_ids, device=self.reference.device).unsqueeze(0)
        with torch.inference_mode():
            loss = self.reference.model(
                input_ids=ids,
                labels=ids,
                use_cache=False,
            ).loss
        return float(math.exp(min(float(loss.item()), 20.0)))

    @staticmethod
    def _project_to_tokens(soft_embeddings, candidate_ids, candidate_vectors):
        import torch.nn.functional as F

        soft = F.normalize(soft_embeddings.float(), dim=-1)
        candidates = F.normalize(candidate_vectors.float(), dim=-1)
        similarities = (candidates * soft.unsqueeze(1)).sum(dim=-1)
        choices = similarities.argmax(dim=-1, keepdim=True)
        return candidate_ids.gather(1, choices).squeeze(1)

    def optimize(self, prompt: str) -> ContinuousOptimizationResult:
        import torch
        import torch.nn.functional as F

        prefix_ids, user_ids, suffix_ids = self._chat_parts(prompt)

        # Preserve numeric constraints, identifiers, dates, and quantities.
        # PPL alone measures fluency and cannot detect a semantic change such
        # as "25" -> "26", so digit-bearing tokens are immutable.
        protected_positions = {
            position
            for position, token_id in enumerate(user_ids)
            if any(
                character.isdigit()
                for character in self.reference.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
            )
        }
        protected_index = (
            torch.tensor(
                sorted(protected_positions),
                dtype=torch.long,
                device=self.reference.device,
            )
            if protected_positions
            else None
        )

        user_tensor = torch.tensor(user_ids, device=self.reference.device)
        embedding_layer = self.reference.model.get_input_embeddings()
        initial_embeddings = embedding_layer(user_tensor).detach().float()
        candidate_ids, candidate_vectors = self._nearest_candidates(
            user_tensor, initial_embeddings
        )

        choice_logits = torch.zeros(
            candidate_ids.shape,
            dtype=torch.float32,
            device=self.reference.device,
            requires_grad=True,
        )
        with torch.no_grad():
            choice_logits[:, 0] = 1.0
        optimizer = torch.optim.Adam([choice_logits], lr=self.learning_rate)

        initial_score = self._hard_divergence(prefix_ids + user_ids + suffix_ids)
        initial_ppl = self._perplexity(user_ids)
        temperature = self.initial_temperature
        history: list[dict[str, float]] = []
        previous = None
        best_value = -float("inf")
        best_soft = initial_embeddings.clone()

        for step in range(self.steps):
            weights = F.softmax(choice_logits / max(temperature, 1e-4), dim=-1)
            soft = (weights.unsqueeze(-1) * candidate_vectors).sum(dim=1)
            delta = soft - initial_embeddings
            norm = delta.norm()
            scale = torch.clamp(self.epsilon / norm.clamp_min(1e-12), max=1.0)
            soft = initial_embeddings + delta * scale
            if protected_index is not None:
                soft = soft.clone()
                soft[protected_index] = initial_embeddings[protected_index]
            full = self._compose_embeddings(prefix_ids, soft, suffix_ids)
            divergence = self._soft_divergence(full)
            penalty = (soft - initial_embeddings).pow(2).mean()
            objective = divergence - self.semantic_weight * penalty

            optimizer.zero_grad(set_to_none=True)
            (-objective).backward()
            optimizer.step()

            value = float(divergence.detach().item())
            if value > best_value:
                best_value = value
                best_soft = soft.detach().clone()
            history.append(
                {
                    "step": float(step + 1),
                    "divergence": value,
                    "penalty": float(penalty.detach().item()),
                    "temperature": float(temperature),
                }
            )
            final_soft = soft.detach()
            if previous is not None and abs(value - previous) < self.convergence_tolerance:
                break
            previous = value
            temperature *= self.anneal

        # Constraint-aware discrete projection. The best continuous embedding
        # guides both the replacement token and the order of edited positions.
        final_soft = best_soft
        soft_unit = F.normalize(final_soft.float(), dim=-1)
        candidate_unit = F.normalize(candidate_vectors.float(), dim=-1)
        similarities = (candidate_unit * soft_unit.unsqueeze(1)).sum(dim=-1)
        alternative_scores, alternative_offsets = similarities[:, 1:].max(dim=1)
        movement_priority = alternative_scores - similarities[:, 0]
        if protected_index is not None:
            movement_priority[protected_index] = -float("inf")

        editable_count = len(user_ids) - len(protected_positions)
        max_changes = min(
            editable_count,
            max(1, int(len(user_ids) * self.max_edit_ratio)),
        )
        position_order = torch.argsort(
            movement_priority,
            descending=True,
        )[:max_changes].tolist()

        optimized_ids = list(user_ids)
        best_discrete_score = initial_score

        for position in position_order:
            candidate_column = int(alternative_offsets[position].item()) + 1
            replacement_id = int(
                candidate_ids[position, candidate_column].item()
            )
            if replacement_id == optimized_ids[position]:
                continue

            # Start from the best accepted prompt. A rejected edit is discarded
            # instead of contaminating all later trials.
            trial_ids = list(optimized_ids)
            trial_ids[position] = replacement_id
            candidate_score = self._hard_divergence(
                prefix_ids + trial_ids + suffix_ids
            )
            candidate_ppl = self._perplexity(trial_ids)
            candidate_ppl_ratio = candidate_ppl / max(initial_ppl, 1e-12)
            score_improved = candidate_score > best_discrete_score
            ppl_passed = candidate_ppl_ratio <= self.ppl_ratio_limit
            replacement_text = self.reference.tokenizer.decode(
                [replacement_id],
                skip_special_tokens=True,
            )
            print(
                "[projection] "
                f"position={position} "
                f"token={replacement_text!r} "
                f"score={candidate_score:.8f} "
                f"ppl_ratio={candidate_ppl_ratio:.4f} "
                f"score_improved={score_improved} "
                f"ppl_passed={ppl_passed}",
                flush=True,
            )
            if score_improved and ppl_passed:
                best_discrete_score = candidate_score
                optimized_ids = trial_ids
        optimized_prompt = self.reference.tokenizer.decode(
            optimized_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        final_score = self._hard_divergence(
            prefix_ids + optimized_ids + suffix_ids
        )
        final_ppl = self._perplexity(optimized_ids)
        ppl_ratio = final_ppl / max(initial_ppl, 1e-12)
        changed = sum(a != b for a, b in zip(user_ids, optimized_ids))
        edit_ratio = changed / max(len(user_ids), 1)
        accepted = (
            final_score > initial_score
            and ppl_ratio <= self.ppl_ratio_limit
            and edit_ratio <= self.max_edit_ratio
        )

        return ContinuousOptimizationResult(
            initial_prompt=prompt,
            optimized_prompt=optimized_prompt,
            accepted=accepted,
            initial_score=initial_score,
            final_score=final_score,
            score_gain=final_score - initial_score,
            initial_ppl=initial_ppl,
            final_ppl=final_ppl,
            ppl_ratio=ppl_ratio,
            edit_ratio=edit_ratio,
            steps_run=len(history),
            history=history,
            proxy_accepted=accepted,
            acceptance_stage="proxy_search_pending_hard_validation",
            requires_hard_validation=True,
            proxy_objective="next_token_js_single_training_variant",
        )
