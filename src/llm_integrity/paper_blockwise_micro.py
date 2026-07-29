from __future__ import annotations

import gc
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .modeling import ModelBundle
from .paper_micro import (
    estimate_next_token_jacobian_frobenius,
)
from .paper_outputs import (
    differentiable_next_token_logits,
)
from .paper_parameter_groups import (
    ParameterGroup,
    build_parameter_groups,
)


@dataclass(frozen=True)
class GroupJacobianResult:
    """
    单个参数组件的Jacobian贡献。
    """

    name: str
    component: str
    layer_index: int | None
    estimate: float
    estimate_per_parameter: float
    parameter_count: int
    parameter_tensors: int
    probe_estimates: tuple[float, ...]


@dataclass(frozen=True)
class BlockwiseJacobianResult:
    """
    所有参数组件汇总后的完整Jacobian估计结果。
    """

    estimate: float
    standard_error: float
    estimate_per_parameter: float
    parameter_count: int
    output_dimension: int
    probes: int
    seed: int
    probe_estimates: tuple[float, ...]
    component_estimates: dict[str, float]
    groups: tuple[GroupJacobianResult, ...]
    complete_parameter_coverage: bool


def _standard_error(
    values: list[float],
) -> float:
    """
    根据独立探针估计值计算均值的标准误。
    """
    count = len(values)

    if count <= 1:
        return 0.0

    mean = sum(values) / count

    sample_variance = sum(
        (value - mean) ** 2
        for value in values
    ) / (count - 1)

    return math.sqrt(
        sample_variance / count
    )


def _clear_temporary_memory() -> None:
    """
    释放Python对象及CUDA缓存。

    empty_cache不会释放仍被引用的张量，
    所以调用前必须先删除计算图相关对象。
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_jacobian_frobenius_blockwise(
    model: Any,
    output_function: Callable[[], Tensor],
    *,
    groups: list[ParameterGroup] | None = None,
    probes: int = 4,
    seed: int = 42,
    show_progress: bool = False,
) -> BlockwiseJacobianResult:
    """
    按参数组件分块估计完整输出Jacobian的Frobenius范数平方。

    计算过程
    --------
    对每个参数组件依次执行：

    1. 关闭全模型参数梯度；
    2. 只开启当前组件参数梯度；
    3. 重新执行前向传播；
    4. 使用相同Rademacher探针计算当前组件贡献；
    5. 保存数值结果；
    6. 删除计算图与梯度；
    7. 继续计算下一个组件。

    各组件使用相同随机种子和相同探针编号，因此：

        S_k = sum_b S_{k,b}

    最终对完整探针结果S_k计算均值与标准误。

    Parameters
    ----------
    model:
        需要分析的PyTorch模型。

    output_function:
        一个无参数函数。每次调用都必须重新前向传播，
        并返回形状为[output_dimension]的可微输出向量。

    groups:
        参数分组。如果未提供，则自动调用
        build_parameter_groups(model)。

    probes:
        Hutchinson探针数量。

    seed:
        所有参数块共用的随机种子。

    show_progress:
        是否输出参数块计算进度。
    """
    if probes <= 0:
        raise ValueError(
            f"probes must be positive, received {probes}"
        )

    parameter_groups = (
        build_parameter_groups(model)
        if groups is None
        else list(groups)
    )

    if not parameter_groups:
        raise ValueError(
            "No parameter groups were supplied."
        )

    all_named_parameters = list(
        model.named_parameters()
    )

    original_requires_grad = {
        id(parameter): parameter.requires_grad
        for _, parameter in all_named_parameters
    }

    original_training = bool(model.training)

    model_parameter_ids = {
        id(parameter)
        for _, parameter in all_named_parameters
    }

    grouped_parameter_ids = {
        id(parameter)
        for group in parameter_groups
        for _, parameter in group.named_parameters
    }

    complete_parameter_coverage = (
        grouped_parameter_ids == model_parameter_ids
    )

    total_probe_estimates = [
        0.0
        for _ in range(probes)
    ]

    group_results: list[GroupJacobianResult] = []
    component_estimates: dict[str, float] = {}

    output_dimension: int | None = None
    selected_parameter_count = 0

    try:
        model.eval()

        for _, parameter in all_named_parameters:
            parameter.requires_grad_(False)

        total_groups = len(parameter_groups)

        for group_index, group in enumerate(
            parameter_groups,
            start=1,
        ):
            if show_progress:
                print(
                    "[micro-block] "
                    f"{group_index}/{total_groups} "
                    f"{group.name} "
                    f"parameters={group.parameter_count}",
                    flush=True,
                )

            for _, parameter in group.named_parameters:
                parameter.requires_grad_(True)

            output_vector = output_function()

            if output_vector.ndim != 1:
                raise ValueError(
                    "output_function must return a one-dimensional "
                    "output vector, but received shape "
                    f"{tuple(output_vector.shape)}"
                )

            if not output_vector.requires_grad:
                raise ValueError(
                    "output_function returned a detached tensor. "
                    "The output must retain its computation graph."
                )

            if output_dimension is None:
                output_dimension = output_vector.numel()
            elif output_vector.numel() != output_dimension:
                raise ValueError(
                    "output dimension changed between parameter groups: "
                    f"expected {output_dimension}, "
                    f"received {output_vector.numel()}"
                )

            group_estimate = (
                estimate_next_token_jacobian_frobenius(
                    output_vector,
                    group.named_parameters,
                    probes=probes,
                    seed=seed,
                )
            )

            if len(group_estimate.probe_estimates) != probes:
                raise RuntimeError(
                    "Unexpected probe count returned by the "
                    "Hutchinson estimator."
                )

            for probe_index, value in enumerate(
                group_estimate.probe_estimates
            ):
                total_probe_estimates[probe_index] += value

            component_estimates[group.component] = (
                component_estimates.get(
                    group.component,
                    0.0,
                )
                + group_estimate.estimate
            )

            selected_parameter_count += (
                group.parameter_count
            )

            group_results.append(
                GroupJacobianResult(
                    name=group.name,
                    component=group.component,
                    layer_index=group.layer_index,
                    estimate=group_estimate.estimate,
                    estimate_per_parameter=(
                        group_estimate.estimate_per_parameter
                    ),
                    parameter_count=group.parameter_count,
                    parameter_tensors=len(
                        group.named_parameters
                    ),
                    probe_estimates=(
                        group_estimate.probe_estimates
                    ),
                )
            )

            for _, parameter in group.named_parameters:
                parameter.requires_grad_(False)

            del group_estimate
            del output_vector

            _clear_temporary_memory()

    finally:
        for _, parameter in all_named_parameters:
            parameter.requires_grad_(
                original_requires_grad[id(parameter)]
            )

        model.train(original_training)

        _clear_temporary_memory()

    if output_dimension is None:
        raise RuntimeError(
            "No output was produced during blockwise estimation."
        )

    estimate = (
        sum(total_probe_estimates) / probes
    )

    estimate_per_parameter = (
        estimate / selected_parameter_count
        if selected_parameter_count > 0
        else 0.0
    )

    standard_error = _standard_error(
        total_probe_estimates
    )

    return BlockwiseJacobianResult(
        estimate=estimate,
        standard_error=standard_error,
        estimate_per_parameter=estimate_per_parameter,
        parameter_count=selected_parameter_count,
        output_dimension=output_dimension,
        probes=probes,
        seed=seed,
        probe_estimates=tuple(
            total_probe_estimates
        ),
        component_estimates=component_estimates,
        groups=tuple(group_results),
        complete_parameter_coverage=(
            complete_parameter_coverage
        ),
    )


def estimate_prompt_jacobian_blockwise(
    bundle: ModelBundle,
    prompt: str,
    *,
    max_length: int = 512,
    system_prompt: str | None = None,
    probes: int = 4,
    seed: int = 42,
    show_progress: bool = False,
) -> BlockwiseJacobianResult:
    """
    对单条提示词计算全模型下一Token logits Jacobian敏感度。

    这是面向项目流程的上层接口。
    """
    if not prompt.strip():
        raise ValueError(
            "prompt must not be empty."
        )

    groups = build_parameter_groups(
        bundle.model
    )

    def output_function() -> Tensor:
        batch_logits = (
            differentiable_next_token_logits(
                bundle,
                [prompt],
                max_length=max_length,
                system_prompt=system_prompt,
                cast_float32=True,
            )
        )

        return batch_logits[0]

    return estimate_jacobian_frobenius_blockwise(
        bundle.model,
        output_function,
        groups=groups,
        probes=probes,
        seed=seed,
        show_progress=show_progress,
    )
