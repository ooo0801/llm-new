from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.paper_in_memory_attacks import _prune_ffn_channels


class ToyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=True)
        self.up_proj = torch.nn.Linear(4, 8, bias=True)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class ToyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()


def test_ffn_channel_pruning_mutates_selected_channels_in_place() -> None:
    layer = ToyLayer()
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                dtype=parameter.dtype,
            ).reshape_as(parameter)
            parameter.copy_(values + index)

    gate_before = layer.mlp.gate_proj.weight.detach().clone()
    up_before = layer.mlp.up_proj.weight.detach().clone()
    down_before = layer.mlp.down_proj.weight.detach().clone()
    gate_bias_before = layer.mlp.gate_proj.bias.detach().clone()
    up_bias_before = layer.mlp.up_proj.bias.detach().clone()

    changed, total, selected = _prune_ffn_channels(
        layer,
        ratio=0.25,
        selection="magnitude",
        seed=7,
    )

    assert len(selected) == 2
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    unselected = [
        index for index in range(8) if index not in set(selected)
    ]

    assert torch.count_nonzero(
        layer.mlp.gate_proj.weight[selected_tensor]
    ) == 0
    assert torch.count_nonzero(
        layer.mlp.up_proj.weight[selected_tensor]
    ) == 0
    assert torch.count_nonzero(
        layer.mlp.down_proj.weight[:, selected_tensor]
    ) == 0
    assert torch.count_nonzero(
        layer.mlp.gate_proj.bias[selected_tensor]
    ) == 0
    assert torch.count_nonzero(
        layer.mlp.up_proj.bias[selected_tensor]
    ) == 0

    assert torch.equal(
        layer.mlp.gate_proj.weight[unselected],
        gate_before[unselected],
    )
    assert torch.equal(
        layer.mlp.up_proj.weight[unselected],
        up_before[unselected],
    )
    assert torch.equal(
        layer.mlp.down_proj.weight[:, unselected],
        down_before[:, unselected],
    )
    assert torch.equal(
        layer.mlp.gate_proj.bias[unselected],
        gate_bias_before[unselected],
    )
    assert torch.equal(
        layer.mlp.up_proj.bias[unselected],
        up_bias_before[unselected],
    )

    expected_changed = 2 * (4 + 4 + 4 + 1 + 1)
    expected_total = (
        gate_before.numel()
        + up_before.numel()
        + down_before.numel()
        + gate_bias_before.numel()
        + up_bias_before.numel()
    )
    assert changed == expected_changed
    assert total == expected_total
