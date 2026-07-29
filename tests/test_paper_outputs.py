from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.paper_outputs import (
    last_non_padding_positions,
    select_last_non_padding_logits,
)


def test_last_non_padding_positions_supports_both_padding_sides() -> None:
    """
    同时验证：
    1. 右侧 padding；
    2. 左侧 padding；
    3. 无 padding。
    """
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],  # 最后有效位置为2
            [0, 0, 1, 1, 1],  # 最后有效位置为4
            [1, 1, 1, 1, 1],  # 最后有效位置为4
        ],
        dtype=torch.long,
    )

    positions = last_non_padding_positions(attention_mask)

    expected = torch.tensor([2, 4, 4], dtype=torch.long)

    assert torch.equal(positions.cpu(), expected)


def test_last_non_padding_positions_rejects_empty_sequence() -> None:
    """
    如果某条输入全部是 padding，应当主动报错，
    避免错误地选择位置-1。
    """
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="at least one non-padding token",
    ):
        last_non_padding_positions(attention_mask)


def test_select_last_non_padding_logits_shape_and_values() -> None:
    """
    验证函数确实选择了各输入最后一个有效位置的完整词表logits。
    """
    batch_size = 3
    sequence_length = 5
    vocabulary_size = 4

    logits = torch.arange(
        batch_size * sequence_length * vocabulary_size,
        dtype=torch.float32,
    ).reshape(
        batch_size,
        sequence_length,
        vocabulary_size,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    selected = select_last_non_padding_logits(
        logits,
        attention_mask,
    )

    expected = torch.stack(
        [
            logits[0, 2, :],
            logits[1, 4, :],
            logits[2, 4, :],
        ],
        dim=0,
    )

    assert selected.shape == (batch_size, vocabulary_size)
    assert selected.dtype == torch.float32
    assert torch.equal(selected, expected)


def test_select_last_non_padding_logits_preserves_gradient() -> None:
    """
    验证选择操作不会切断计算图。

    对选中的全部logits求和后反向传播：
    - 被选中的序列位置梯度应当为1；
    - 其余序列位置梯度应当为0。
    """
    batch_size = 2
    sequence_length = 4
    vocabulary_size = 3

    logits = torch.randn(
        batch_size,
        sequence_length,
        vocabulary_size,
        dtype=torch.float32,
        requires_grad=True,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],  # 最后有效位置为1
            [0, 1, 1, 1],  # 最后有效位置为3
        ],
        dtype=torch.long,
    )

    selected = select_last_non_padding_logits(
        logits,
        attention_mask,
    )

    assert selected.requires_grad
    assert selected.grad_fn is not None

    selected.sum().backward()

    assert logits.grad is not None

    expected_gradient = torch.zeros_like(logits)
    expected_gradient[0, 1, :] = 1.0
    expected_gradient[1, 3, :] = 1.0

    assert torch.equal(logits.grad, expected_gradient)


def test_select_last_non_padding_logits_casts_after_selection() -> None:
    """
    验证低精度模型输出会在选取之后转换为float32，
    同时仍然保留梯度。
    """
    logits = torch.randn(
        1,
        3,
        5,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    attention_mask = torch.tensor(
        [[1, 1, 0]],
        dtype=torch.long,
    )

    selected = select_last_non_padding_logits(
        logits,
        attention_mask,
        cast_float32=True,
    )

    assert selected.dtype == torch.float32
    assert selected.requires_grad

    selected.sum().backward()

    assert logits.grad is not None
    assert logits.grad[0, 1, :].abs().sum().item() > 0
    assert logits.grad[0, 0, :].abs().sum().item() == 0
    assert logits.grad[0, 2, :].abs().sum().item() == 0
