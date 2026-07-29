from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.paper_parameter_groups import (
    build_parameter_groups,
    classify_parameter_name,
    summarize_parameter_groups,
)


class ToyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)


class ToyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class ToyDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()
        self.input_layernorm = torch.nn.LayerNorm(4)
        self.post_attention_layernorm = torch.nn.LayerNorm(4)


class ToyInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(20, 4)
        self.layers = torch.nn.ModuleList(
            [
                ToyDecoderLayer(),
                ToyDecoderLayer(),
            ]
        )
        self.norm = torch.nn.LayerNorm(4)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ToyInnerModel()
        self.lm_head = torch.nn.Linear(4, 20, bias=False)


def test_classify_qwen_parameter_names() -> None:
    assert classify_parameter_name(
        "model.embed_tokens.weight"
    ) == (
        "token_embeddings",
        "embedding",
        None,
    )

    assert classify_parameter_name(
        "model.layers.3.self_attn.q_proj.weight"
    ) == (
        "layer_03_attention",
        "attention",
        3,
    )

    assert classify_parameter_name(
        "model.layers.3.mlp.down_proj.weight"
    ) == (
        "layer_03_ffn",
        "ffn",
        3,
    )

    assert classify_parameter_name(
        "model.layers.3.input_layernorm.weight"
    ) == (
        "layer_03_normalization",
        "normalization",
        3,
    )

    assert classify_parameter_name(
        "model.norm.weight"
    ) == (
        "final_normalization",
        "normalization",
        None,
    )

    assert classify_parameter_name(
        "lm_head.weight"
    ) == (
        "lm_head",
        "output",
        None,
    )


def test_every_parameter_is_grouped_exactly_once() -> None:
    model = ToyModel()

    groups = build_parameter_groups(model)

    original_parameters = list(
        model.named_parameters()
    )

    grouped_parameters = [
        item
        for group in groups
        for item in group.named_parameters
    ]

    assert len(grouped_parameters) == len(
        original_parameters
    )

    assert {
        id(parameter)
        for _, parameter in grouped_parameters
    } == {
        id(parameter)
        for _, parameter in original_parameters
    }

    assert sum(
        group.parameter_count
        for group in groups
    ) == sum(
        parameter.numel()
        for _, parameter in original_parameters
    )


def test_expected_component_groups_are_created() -> None:
    model = ToyModel()

    groups = build_parameter_groups(model)
    group_names = {
        group.name
        for group in groups
    }

    assert "token_embeddings" in group_names
    assert "layer_00_attention" in group_names
    assert "layer_00_ffn" in group_names
    assert "layer_00_normalization" in group_names
    assert "layer_01_attention" in group_names
    assert "layer_01_ffn" in group_names
    assert "layer_01_normalization" in group_names
    assert "final_normalization" in group_names
    assert "lm_head" in group_names


def test_summary_parameter_count_matches_model() -> None:
    model = ToyModel()
    groups = build_parameter_groups(model)

    summary = summarize_parameter_groups(groups)

    expected = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert summary["total_parameters"] == expected
    assert summary["group_count"] == len(groups)
    assert "attention" in summary[
        "component_parameter_counts"
    ]
    assert "ffn" in summary[
        "component_parameter_counts"
    ]
