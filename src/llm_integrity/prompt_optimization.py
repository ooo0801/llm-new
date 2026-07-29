from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distances import js_divergence_logits
from .modeling import ModelBundle, render_prompt


@dataclass
class OptimizationResult:
    initial_prompt: str
    optimized_prompt: str
    initial_score: float
    final_score: float
    perplexity: float
    history: list[dict[str, float]]


class SoftPromptOptimizer:
    """Constrained soft-token optimizer implementing Eqs. (13)-(17) approximately.

    To avoid a sequence_length x vocabulary matrix, each position is optimized over
    its nearest candidate tokens. It is intended for same-architecture model variants.
    """

    def __init__(
        self,
        reference: ModelBundle,
        modified: ModelBundle,
        candidate_tokens: int = 64,
        learning_rate: float = 0.1,
        steps: int = 30,
        initial_temperature: float = 1.0,
        anneal: float = 0.95,
        epsilon: float = 2.0,
        semantic_weight: float = 0.1,
    ) -> None:
        self.reference = reference
        self.modified = modified
        self.candidate_tokens = candidate_tokens
        self.learning_rate = learning_rate
        self.steps = steps
        self.initial_temperature = initial_temperature
        self.anneal = anneal
        self.epsilon = epsilon
        self.semantic_weight = semantic_weight
        for parameter in self.reference.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.modified.model.parameters():
            parameter.requires_grad_(False)

    def optimize(self, prompt: str) -> OptimizationResult:
        import torch
        import torch.nn.functional as F

        if self.reference.model.config.hidden_size != self.modified.model.config.hidden_size:
            raise ValueError("Soft optimization requires the same embedding dimension")
        tokenizer = self.reference.tokenizer
        text = render_prompt(tokenizer, prompt)
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        token_ids = token_ids.to(self.reference.device)
        embedding_layer = self.reference.model.get_input_embeddings()
        embedding_matrix = embedding_layer.weight.detach()
        initial_embeddings = embedding_layer(token_ids).detach()
        normalized_vocab = F.normalize(embedding_matrix.float(), dim=-1)
        normalized_initial = F.normalize(initial_embeddings[0].float(), dim=-1)
        similarities = normalized_initial @ normalized_vocab.T
        candidate_ids = similarities.topk(self.candidate_tokens, dim=-1).indices
        candidate_vectors = embedding_matrix[candidate_ids].detach()
        choice_logits = torch.zeros(
            candidate_ids.shape,
            device=self.reference.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([choice_logits], lr=self.learning_rate)
        history: list[dict[str, float]] = []
        initial_score = 0.0
        temperature = self.initial_temperature
        for step in range(self.steps):
            weights = F.softmax(choice_logits / max(temperature, 1e-4), dim=-1)
            soft_embeddings = (weights.unsqueeze(-1) * candidate_vectors.float()).sum(dim=-2).unsqueeze(0)
            delta = soft_embeddings - initial_embeddings.float()
            delta_norm = delta.norm()
            if delta_norm > self.epsilon:
                soft_embeddings = initial_embeddings.float() + delta * (self.epsilon / delta_norm)
            ref = self.reference.model(inputs_embeds=soft_embeddings, use_cache=False).logits[:, -1].float()
            mod = self.modified.model(inputs_embeds=soft_embeddings, use_cache=False).logits[:, -1].float()
            divergence = js_divergence_logits(ref, mod).mean()
            semantic_penalty = (soft_embeddings - initial_embeddings.float()).pow(2).mean()
            objective = divergence - self.semantic_weight * semantic_penalty
            optimizer.zero_grad(set_to_none=True)
            (-objective).backward()
            optimizer.step()
            if step == 0:
                initial_score = float(divergence.detach().item())
            history.append(
                {
                    "step": float(step),
                    "divergence": float(divergence.detach().item()),
                    "penalty": float(semantic_penalty.detach().item()),
                    "temperature": float(temperature),
                }
            )
            temperature *= self.anneal
        discrete_choice = choice_logits.detach().argmax(dim=-1)
        position = torch.arange(candidate_ids.shape[0], device=candidate_ids.device)
        optimized_ids = candidate_ids[position, discrete_choice].unsqueeze(0)
        optimized_prompt = tokenizer.decode(optimized_ids[0], skip_special_tokens=True)
        final_score = history[-1]["divergence"]
        perplexity = self._perplexity(optimized_ids)
        return OptimizationResult(prompt, optimized_prompt, initial_score, final_score, perplexity, history)

    def _perplexity(self, token_ids) -> float:
        import math
        import torch

        if token_ids.shape[1] < 2:
            return float("inf")
        with torch.inference_mode():
            outputs = self.reference.model(input_ids=token_ids, labels=token_ids, use_cache=False)
        return float(math.exp(min(float(outputs.loss.item()), 20.0)))
