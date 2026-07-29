from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .modeling import ModelBundle, tokenize_prompts

if TYPE_CHECKING:
    from collections.abc import Sequence


def last_non_padding_positions(attention_mask: Tensor) -> Tensor:
    """
    计算批次中每条输入最后一个非 padding Token 的位置。

    同时支持左侧 padding 与右侧 padding。

    Parameters
    ----------
    attention_mask:
        形状为 [batch_size, sequence_length] 的注意力掩码。
        有效 Token 为 1，padding Token 为 0。

    Returns
    -------
    Tensor
        形状为 [batch_size]，每个元素表示对应输入最后一个
        有效 Token 在序列维度中的下标。
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [batch_size, sequence_length], "
            f"but received shape {tuple(attention_mask.shape)}"
        )

    valid_mask = attention_mask.to(dtype=torch.bool)

    if not valid_mask.any(dim=1).all():
        raise ValueError(
            "Every input must contain at least one non-padding token."
        )

    token_positions = torch.arange(
        valid_mask.shape[1],
        device=valid_mask.device,
    ).unsqueeze(0).expand_as(valid_mask)

    return token_positions.masked_fill(
        ~valid_mask,
        -1,
    ).max(dim=1).values


def select_last_non_padding_logits(
    logits: Tensor,
    attention_mask: Tensor,
    *,
    cast_float32: bool = True,
) -> Tensor:
    """
    从全序列 logits 中提取每条输入最后一个有效位置的完整词表 logits。

    输入 logits 的形状：
        [batch_size, sequence_length, vocabulary_size]

    返回结果的形状：
        [batch_size, vocabulary_size]

    该函数不使用 detach、cpu、no_grad 或 inference_mode，
    因而保留从输出 logits 到模型参数的完整计算图。
    """
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape "
            "[batch_size, sequence_length, vocabulary_size], "
            f"but received shape {tuple(logits.shape)}"
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape "
            "[batch_size, sequence_length], "
            f"but received shape {tuple(attention_mask.shape)}"
        )

    if logits.shape[:2] != attention_mask.shape:
        raise ValueError(
            "The batch and sequence dimensions of logits and attention_mask "
            f"must match, but received logits={tuple(logits.shape)} and "
            f"attention_mask={tuple(attention_mask.shape)}"
        )

    last_positions = last_non_padding_positions(attention_mask)

    row_indices = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    selected = logits[row_indices, last_positions, :]

    if cast_float32:
        selected = selected.float()

    return selected


def differentiable_next_token_logits(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    max_length: int = 512,
    system_prompt: str | None = None,
    cast_float32: bool = True,
) -> Tensor:
    """
    计算提示词最后一个有效位置的下一 Token 完整 logits 向量，
    并保留到模型参数的计算图。

    对应申请书中的输出定义：

        f_theta(x) = z_{theta,T(x)}(x) ∈ R^{|V|}

    与 modeling.next_token_logits 的区别：

    1. 不使用 torch.inference_mode；
    2. 不使用 torch.no_grad；
    3. 不调用 detach；
    4. 不将结果移动到 CPU；
    5. 返回结果保留对模型参数求导所需的计算图。

    返回形状：

        [batch_size, vocabulary_size]
    """
    prompt_list = list(prompts)

    if not prompt_list:
        raise ValueError("prompts must contain at least one prompt.")

    encoded = tokenize_prompts(
        bundle,
        prompt_list,
        max_length=max_length,
        system_prompt=system_prompt,
    )

    outputs = bundle.model(
        **encoded,
        use_cache=False,
        return_dict=True,
    )

    return select_last_non_padding_logits(
        outputs.logits,
        encoded["attention_mask"],
        cast_float32=cast_float32,
    )
