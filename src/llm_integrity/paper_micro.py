from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter


@dataclass
class HutchinsonJacobianResult:
    """
    Hutchinson方法得到的Jacobian Frobenius范数平方估计结果。
    """

    estimate: float
    standard_error: float
    estimate_per_parameter: float
    parameter_count: int
    output_dimension: int
    probes: int
    seed: int
    probe_estimates: tuple[float, ...]
    per_parameter_tensor: dict[str, float]


def _prepare_named_parameters(
    named_parameters: Iterable[tuple[str, Parameter]],
) -> list[tuple[str, Parameter]]:
    """
    过滤掉不参与梯度计算的参数，并检查参数名称是否重复。
    """
    selected: list[tuple[str, Parameter]] = []
    names: set[str] = set()

    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue

        if name in names:
            raise ValueError(
                f"Duplicate parameter name detected: {name}"
            )

        names.add(name)
        selected.append((name, parameter))

    if not selected:
        raise ValueError(
            "No trainable parameters were supplied to the "
            "Hutchinson Jacobian estimator."
        )

    return selected


def _rademacher_probe(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """
    生成Rademacher随机向量。

    每个元素独立地以相同概率取-1或+1，因此：

        E[r r^T] = I
    """
    integer_probe = torch.randint(
        low=0,
        high=2,
        size=shape,
        generator=generator,
        device=device,
        dtype=torch.int64,
    )

    return integer_probe.to(dtype=dtype).mul_(2).sub_(1)


def estimate_next_token_jacobian_frobenius(
    output_logits: Tensor,
    named_parameters: Iterable[tuple[str, Parameter]],
    *,
    probes: int = 4,
    seed: int = 42,
) -> HutchinsonJacobianResult:
    """
    估计下一Token完整logits相对于指定模型参数的Jacobian Frobenius范数平方。

    数学定义
    --------
    设：

        z = f_theta(x) ∈ R^{|V|}

    Jacobian为：

        J = ∂z / ∂theta

    目标量为：

        ||J||_F^2

    Hutchinson恒等式：

        ||J||_F^2 = E_r[||J^T r||_2^2]

    其中Rademacher随机向量r的每个元素独立取-1或+1。

    Parameters
    ----------
    output_logits:
        单条提示词产生的完整下一Token logits向量，
        形状必须为[vocabulary_size]，并且必须保留计算图。

    named_parameters:
        需要计算敏感度的模型参数，通常来自：

            model.named_parameters()

    probes:
        Hutchinson随机探针数量。调试阶段可使用2，
        正式实验建议至少4，并进一步检查稳定性。

    seed:
        随机种子，用于保证实验可复现。

    Returns
    -------
    HutchinsonJacobianResult
        包含原始估计值、标准误、参数归一化值和参数张量级分解。
    """
    if output_logits.ndim != 1:
        raise ValueError(
            "output_logits must be a one-dimensional next-token "
            f"logits vector, but received shape {tuple(output_logits.shape)}"
        )

    if not output_logits.requires_grad:
        raise ValueError(
            "output_logits does not require gradients. "
            "Use differentiable_next_token_logits instead of "
            "modeling.next_token_logits."
        )

    if probes <= 0:
        raise ValueError(
            f"probes must be positive, but received {probes}"
        )

    selected = _prepare_named_parameters(named_parameters)
    parameter_names = [name for name, _ in selected]
    parameters = [parameter for _, parameter in selected]

    parameter_count = sum(
        parameter.numel()
        for parameter in parameters
    )

    generator = torch.Generator(
        device=output_logits.device,
    )
    generator.manual_seed(seed)

    probe_estimates: list[float] = []

    accumulated_by_parameter = {
        name: 0.0
        for name in parameter_names
    }

    for probe_index in range(probes):
        probe = _rademacher_probe(
            output_logits.shape,
            device=output_logits.device,
            dtype=output_logits.dtype,
            generator=generator,
        )

        projected_output = torch.sum(
            output_logits * probe
        )

        gradients = torch.autograd.grad(
            outputs=projected_output,
            inputs=parameters,
            retain_graph=probe_index < probes - 1,
            create_graph=False,
            allow_unused=True,
        )

        probe_total = 0.0

        for name, gradient in zip(
            parameter_names,
            gradients,
            strict=True,
        ):
            if gradient is None:
                squared_norm = 0.0
            else:
                squared_norm = float(
                    gradient.detach()
                    .to(dtype=torch.float32)
                    .square()
                    .sum()
                    .item()
                )

            probe_total += squared_norm
            accumulated_by_parameter[name] += squared_norm

        probe_estimates.append(probe_total)

    estimate = sum(probe_estimates) / probes

    if probes > 1:
        sample_variance = sum(
            (value - estimate) ** 2
            for value in probe_estimates
        ) / (probes - 1)

        standard_error = math.sqrt(
            sample_variance / probes
        )
    else:
        standard_error = 0.0

    per_parameter_tensor = {
        name: value / probes
        for name, value in accumulated_by_parameter.items()
    }

    estimate_per_parameter = (
        estimate / parameter_count
        if parameter_count > 0
        else 0.0
    )

    return HutchinsonJacobianResult(
        estimate=estimate,
        standard_error=standard_error,
        estimate_per_parameter=estimate_per_parameter,
        parameter_count=parameter_count,
        output_dimension=output_logits.numel(),
        probes=probes,
        seed=seed,
        probe_estimates=tuple(probe_estimates),
        per_parameter_tensor=per_parameter_tensor,
    )
