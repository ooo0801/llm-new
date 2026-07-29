from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .paper_hybrid_sensitivity import (
    RobustCalibration,
    combine_hybrid_tensor,
)


@dataclass
class HybridObjectiveTensorResult:
    """
    连续嵌入优化阶段的可微目标结果。

    所有字段均保留 PyTorch 计算图，不能在这里调用：
    - detach()
    - item()
    - numpy()
    """

    objective: Any
    hybrid_score: Any
    micro_raw: Any
    macro_raw: Any
    semantic_penalty: Any
    micro_weight: Any
    macro_weight: Any
    weighting_mode: str


@dataclass(frozen=True)
class HybridObjectiveValueResult:
    """
    离散 Token 投影后的普通数值评分结果。
    """

    objective: float
    hybrid_score: float
    micro_raw: float
    macro_raw: float
    semantic_penalty: float
    micro_weight: float
    macro_weight: float
    weighting_mode: str


class RobustHybridPromptObjective:
    """
    连续优化和离散重评分共用的稳健复合敏感度接口。

    数学目标：

        objective
        = micro_weight * stable(micro_raw)
        + macro_weight * stable(macro_raw)
        - semantic_weight * semantic_penalty

    其中 stable() 使用预先拟合的 log1p + median/MAD 稳健标定。
    """

    def __init__(
        self,
        *,
        micro_calibration: RobustCalibration,
        macro_calibration: RobustCalibration,
        micro_weight: float = 0.5,
        macro_weight: float = 0.5,
        weighting_mode: str = "adaptive",
        hybrid_beta: float = 1.0,
        semantic_weight: float = 0.0,
        clip: float = 8.0,
    ) -> None:
        if semantic_weight < 0:
            raise ValueError("semantic_weight must be non-negative")

        self.micro_calibration = micro_calibration
        self.macro_calibration = macro_calibration
        self.micro_weight = float(micro_weight)
        self.macro_weight = float(macro_weight)
        self.weighting_mode = str(weighting_mode).lower()
        self.hybrid_beta = float(hybrid_beta)
        self.semantic_weight = float(semantic_weight)
        self.clip = float(clip)

        if self.weighting_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                "weighting_mode must be either 'adaptive' or 'fixed'"
            )

    def score_soft(
        self,
        *,
        micro_raw,
        macro_raw,
        semantic_penalty=None,
    ) -> HybridObjectiveTensorResult:
        """
        连续嵌入空间中的可微评分。

        micro_raw、macro_raw 和 semantic_penalty 应当是
        PyTorch 标量 Tensor。返回值保留完整梯度链。
        """

        import torch

        if not isinstance(micro_raw, torch.Tensor):
            raise TypeError("micro_raw must be a torch.Tensor")

        if not isinstance(macro_raw, torch.Tensor):
            raise TypeError("macro_raw must be a torch.Tensor")

        if semantic_penalty is None:
            semantic_penalty = torch.zeros_like(micro_raw)
        elif not isinstance(semantic_penalty, torch.Tensor):
            raise TypeError(
                "semantic_penalty must be a torch.Tensor or None"
            )

        (
            hybrid_score,
            micro_weight,
            macro_weight,
        ) = combine_hybrid_tensor(
            micro_raw=micro_raw,
            macro_raw=macro_raw,
            micro_calibration=self.micro_calibration,
            macro_calibration=self.macro_calibration,
            micro_weight=self.micro_weight,
            macro_weight=self.macro_weight,
            weighting_mode=self.weighting_mode,
            hybrid_beta=self.hybrid_beta,
            clip=self.clip,
            return_weights=True,
        )

        objective = (
            hybrid_score
            - self.semantic_weight * semantic_penalty
        )

        return HybridObjectiveTensorResult(
            objective=objective,
            hybrid_score=hybrid_score,
            micro_raw=micro_raw,
            macro_raw=macro_raw,
            semantic_penalty=semantic_penalty,
            micro_weight=micro_weight,
            macro_weight=macro_weight,
            weighting_mode=self.weighting_mode,
        )

    def score_hard(
        self,
        *,
        micro_raw: float,
        macro_raw: float,
        semantic_penalty: float = 0.0,
    ) -> HybridObjectiveValueResult:
        """
        离散 Token 投影后的重评分。

        它与 score_soft 使用完全相同的复合公式，
        但返回普通 float，供候选比较、排序和写入结果文件。
        """

        import torch

        micro_tensor = torch.tensor(
            float(micro_raw),
            dtype=torch.float64,
        )
        macro_tensor = torch.tensor(
            float(macro_raw),
            dtype=torch.float64,
        )
        penalty_tensor = torch.tensor(
            float(semantic_penalty),
            dtype=torch.float64,
        )

        result = self.score_soft(
            micro_raw=micro_tensor,
            macro_raw=macro_tensor,
            semantic_penalty=penalty_tensor,
        )

        return HybridObjectiveValueResult(
            objective=float(result.objective.item()),
            hybrid_score=float(result.hybrid_score.item()),
            micro_raw=float(micro_raw),
            macro_raw=float(macro_raw),
            semantic_penalty=float(semantic_penalty),
            micro_weight=float(result.micro_weight.item()),
            macro_weight=float(result.macro_weight.item()),
            weighting_mode=result.weighting_mode,
        )
