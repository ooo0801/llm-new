from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RobustCalibration:
    """
    一个敏感度分量的稳健标定参数。

    center:
        log1p 变换后样本的中位数。

    scale:
        log1p 变换后样本的稳健尺度，优先使用
        1.4826 * MAD。
    """

    center: float
    scale: float
    sample_count: int
    transform: str = "log1p"
    scale_method: str = "mad"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridSensitivityResult:
    micro_raw: float
    macro_raw: float

    micro_transformed: float
    macro_transformed: float

    micro_stable: float
    macro_stable: float

    micro_weight: float
    macro_weight: float
    weighting_mode: str
    hybrid_beta: float

    hybrid_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_raw_value(value: float, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative because it represents "
            f"a squared sensitivity or distance, got {value}"
        )

    return value


def _log_transform(value: float) -> float:
    return math.log1p(value)


def fit_robust_calibration(
    values: Iterable[float],
    *,
    minimum_scale: float = 1e-6,
) -> RobustCalibration:
    """
    根据一组历史敏感度样本拟合稳健标定参数。

    处理流程：
    1. 检查敏感度非负且为有限值；
    2. 使用 log1p 压缩极端大值；
    3. 使用中位数作为中心；
    4. 使用 1.4826 * MAD 作为稳健尺度；
    5. MAD 为零时退化为总体标准差；
    6. 仍为零时使用 minimum_scale。
    """

    raw_values = [
        _validate_raw_value(value, "calibration value")
        for value in values
    ]

    if not raw_values:
        raise ValueError("At least one calibration value is required")

    transformed = [_log_transform(value) for value in raw_values]
    center = float(statistics.median(transformed))

    absolute_deviations = [
        abs(value - center)
        for value in transformed
    ]
    mad = float(statistics.median(absolute_deviations))
    scale = 1.4826 * mad
    scale_method = "mad"

    if scale < minimum_scale and len(transformed) >= 2:
        scale = float(statistics.pstdev(transformed))
        scale_method = "standard_deviation_fallback"

    if scale < minimum_scale:
        scale = float(minimum_scale)
        scale_method = "minimum_scale_fallback"

    return RobustCalibration(
        center=center,
        scale=scale,
        sample_count=len(raw_values),
        transform="log1p",
        scale_method=scale_method,
    )


def stabilize_scalar(
    raw_value: float,
    calibration: RobustCalibration,
    *,
    clip: float = 8.0,
) -> tuple[float, float]:
    """
    返回：
        transformed_value, stabilized_value
    """

    raw_value = _validate_raw_value(raw_value, "raw sensitivity")
    transformed = _log_transform(raw_value)

    stable = (
        transformed - float(calibration.center)
    ) / float(calibration.scale)

    if clip > 0:
        stable = max(-clip, min(clip, stable))

    return transformed, stable


def normalize_weights(
    micro_weight: float,
    macro_weight: float,
) -> tuple[float, float]:
    micro_weight = float(micro_weight)
    macro_weight = float(macro_weight)

    if micro_weight < 0 or macro_weight < 0:
        raise ValueError("Hybrid sensitivity weights cannot be negative")

    total = micro_weight + macro_weight

    if total <= 0:
        raise ValueError(
            "At least one hybrid sensitivity weight must be positive"
        )

    return micro_weight / total, macro_weight / total


def _validate_adaptive_settings(
    hybrid_beta: float,
    prior_micro_weight: float,
) -> tuple[float, float]:
    hybrid_beta = float(hybrid_beta)
    prior_micro_weight = float(prior_micro_weight)

    if not math.isfinite(hybrid_beta) or hybrid_beta < 0:
        raise ValueError("hybrid_beta must be finite and non-negative")

    if not 0.0 < prior_micro_weight < 1.0:
        raise ValueError(
            "Adaptive weighting requires a prior micro weight strictly "
            "between zero and one"
        )

    return hybrid_beta, prior_micro_weight


def adaptive_hybrid_weights_scalar(
    *,
    micro_stable: float,
    macro_stable: float,
    hybrid_beta: float = 1.0,
    prior_micro_weight: float = 0.5,
) -> tuple[float, float]:
    """Return the application-style adaptive balance factor.

    Robustly calibrated sensitivities live on a log scale.  Consequently,
    ``exp(micro_stable) / exp(macro_stable)`` is their positive evidence
    ratio, and its logarithm is simply ``micro_stable - macro_stable``.
    Applying a sigmoid to that log ratio implements Eq. (11) without an
    unstable division or an arbitrary positive shift.
    """

    hybrid_beta, prior_micro_weight = _validate_adaptive_settings(
        hybrid_beta,
        prior_micro_weight,
    )
    prior_log_odds = math.log(
        prior_micro_weight / (1.0 - prior_micro_weight)
    )
    log_odds = (
        prior_log_odds
        + hybrid_beta * (float(micro_stable) - float(macro_stable))
    )
    log_odds = max(-60.0, min(60.0, log_odds))
    micro_weight = 1.0 / (1.0 + math.exp(-log_odds))
    return micro_weight, 1.0 - micro_weight


def combine_hybrid_scalar(
    *,
    micro_raw: float,
    macro_raw: float,
    micro_calibration: RobustCalibration,
    macro_calibration: RobustCalibration,
    micro_weight: float = 0.5,
    macro_weight: float = 0.5,
    weighting_mode: str = "fixed",
    hybrid_beta: float = 1.0,
    clip: float = 8.0,
) -> HybridSensitivityResult:
    """
    组合不可微的标量微观、宏观敏感度。

    适用于：
    - 候选提示词离线评分；
    - 实验结果汇总；
    - MCC 前的复合敏感度排序；
    - 校验优化后提示词的真实增益。
    """

    prior_micro_weight, prior_macro_weight = normalize_weights(
        micro_weight,
        macro_weight,
    )

    micro_transformed, micro_stable = stabilize_scalar(
        micro_raw,
        micro_calibration,
        clip=clip,
    )
    macro_transformed, macro_stable = stabilize_scalar(
        macro_raw,
        macro_calibration,
        clip=clip,
    )

    weighting_mode = str(weighting_mode).lower()
    if weighting_mode == "adaptive":
        micro_weight, macro_weight = adaptive_hybrid_weights_scalar(
            micro_stable=micro_stable,
            macro_stable=macro_stable,
            hybrid_beta=hybrid_beta,
            prior_micro_weight=prior_micro_weight,
        )
    elif weighting_mode == "fixed":
        micro_weight = prior_micro_weight
        macro_weight = prior_macro_weight
    else:
        raise ValueError(
            "weighting_mode must be either 'adaptive' or 'fixed'"
        )

    hybrid_score = (
        micro_weight * micro_stable
        + macro_weight * macro_stable
    )

    return HybridSensitivityResult(
        micro_raw=float(micro_raw),
        macro_raw=float(macro_raw),
        micro_transformed=micro_transformed,
        macro_transformed=macro_transformed,
        micro_stable=micro_stable,
        macro_stable=macro_stable,
        micro_weight=micro_weight,
        macro_weight=macro_weight,
        weighting_mode=weighting_mode,
        hybrid_beta=float(hybrid_beta),
        hybrid_score=hybrid_score,
    )


def combine_hybrid_tensor(
    *,
    micro_raw,
    macro_raw,
    micro_calibration: RobustCalibration,
    macro_calibration: RobustCalibration,
    micro_weight: float = 0.5,
    macro_weight: float = 0.5,
    weighting_mode: str = "fixed",
    hybrid_beta: float = 1.0,
    clip: float = 8.0,
    return_weights: bool = False,
):
    """
    可微的复合敏感度组合函数。

    micro_raw 和 macro_raw 应为 PyTorch 标量 Tensor。
    本函数不调用 .item()、detach() 或 numpy()，因此梯度能够继续
    传回提示词嵌入。

    后续连续嵌入优化将使用这个接口。
    """

    import torch

    if not isinstance(micro_raw, torch.Tensor):
        raise TypeError("micro_raw must be a torch.Tensor")

    if not isinstance(macro_raw, torch.Tensor):
        raise TypeError("macro_raw must be a torch.Tensor")

    prior_micro_weight, prior_macro_weight = normalize_weights(
        micro_weight,
        macro_weight,
    )

    micro_value = torch.clamp(micro_raw, min=0.0)
    macro_value = torch.clamp(macro_raw, min=0.0)

    micro_stable = (
        torch.log1p(micro_value)
        - float(micro_calibration.center)
    ) / float(micro_calibration.scale)

    macro_stable = (
        torch.log1p(macro_value)
        - float(macro_calibration.center)
    ) / float(macro_calibration.scale)

    if clip > 0:
        micro_stable = torch.clamp(
            micro_stable,
            min=-clip,
            max=clip,
        )
        macro_stable = torch.clamp(
            macro_stable,
            min=-clip,
            max=clip,
        )

    weighting_mode = str(weighting_mode).lower()
    if weighting_mode == "adaptive":
        hybrid_beta, prior_micro_weight = _validate_adaptive_settings(
            hybrid_beta,
            prior_micro_weight,
        )
        prior_log_odds = math.log(
            prior_micro_weight / (1.0 - prior_micro_weight)
        )
        micro_weight_tensor = torch.sigmoid(
            prior_log_odds
            + hybrid_beta * (micro_stable - macro_stable)
        )
        macro_weight_tensor = 1.0 - micro_weight_tensor
    elif weighting_mode == "fixed":
        micro_weight_tensor = torch.as_tensor(
            prior_micro_weight,
            dtype=micro_stable.dtype,
            device=micro_stable.device,
        )
        macro_weight_tensor = torch.as_tensor(
            prior_macro_weight,
            dtype=macro_stable.dtype,
            device=macro_stable.device,
        )
    else:
        raise ValueError(
            "weighting_mode must be either 'adaptive' or 'fixed'"
        )

    score = (
        micro_weight_tensor * micro_stable
        + macro_weight_tensor * macro_stable
    )

    if return_weights:
        return score, micro_weight_tensor, macro_weight_tensor

    return score
