from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .paper_hybrid_sensitivity import (
    RobustCalibration,
    fit_robust_calibration,
)
from .paper_prompt_objective import RobustHybridPromptObjective


@dataclass(frozen=True)
class HybridCandidateDecision:
    prompt_id: str
    proxy_accepted: bool
    accepted: bool
    initial_objective: float
    optimized_objective: float
    objective_gain: float
    initial_micro: float
    optimized_micro: float
    initial_macro: float
    optimized_macro: float
    initial_alpha: float
    optimized_alpha: float
    weighting_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_prompt_macro(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str = "macro_l2_raw",
) -> dict[str, float]:
    """Compute the five-family expectation separately for every prompt."""

    values: dict[str, dict[str, list[float]]] = {}
    weights: dict[str, list[float]] = {}

    for record in records:
        prompt_id = str(record["prompt_id"])
        family = str(record["family"])
        if metric not in record:
            raise ValueError(
                f"Macro record for {prompt_id!r} has no {metric!r}"
            )
        values.setdefault(prompt_id, {}).setdefault(family, []).append(
            float(record[metric])
        )
        weights.setdefault(family, []).append(
            float(record.get("family_weight", 1.0))
        )

    if not values:
        raise ValueError("No macro records were supplied")

    family_weights = {
        family: sum(items) / len(items)
        for family, items in weights.items()
    }
    total_weight = sum(family_weights.values())
    if total_weight <= 0:
        raise ValueError("Macro family weights must have a positive sum")
    family_weights = {
        family: weight / total_weight
        for family, weight in family_weights.items()
    }

    expected_families = set(family_weights)
    result: dict[str, float] = {}
    for prompt_id, by_family in values.items():
        missing = expected_families - set(by_family)
        if missing:
            raise ValueError(
                f"Prompt {prompt_id!r} is missing macro families: "
                + ", ".join(sorted(missing))
            )
        result[prompt_id] = sum(
            family_weights[family]
            * (sum(by_family[family]) / len(by_family[family]))
            for family in expected_families
        )

    return result


def index_micro_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in records:
        prompt_id = str(record["prompt_id"])
        if prompt_id in result:
            raise ValueError(f"Duplicate micro score for {prompt_id!r}")
        raw = record.get("micro_raw", record.get("estimate"))
        if raw is None:
            raise ValueError(
                f"Micro record for {prompt_id!r} has no micro_raw"
            )
        result[prompt_id] = float(raw)
    if not result:
        raise ValueError("No micro records were supplied")
    return result


def validate_proxy_candidates(
    proxy_rows: Sequence[Mapping[str, Any]],
    micro_records: Sequence[Mapping[str, Any]],
    macro_records: Sequence[Mapping[str, Any]],
    *,
    metric: str = "macro_l2_raw",
    weighting_mode: str = "adaptive",
    hybrid_beta: float = 1.0,
    micro_weight: float = 0.5,
    macro_weight: float = 0.5,
    minimum_gain: float = 0.0,
    micro_calibration: RobustCalibration | None = None,
    macro_calibration: RobustCalibration | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply expensive hard hybrid acceptance after proxy optimization.

    Prompt identifiers must follow ``<source id>::initial`` and
    ``<source id>::optimized``.  Production evaluation should pass fixed
    calibrations fitted on a separate development pool.  Fitting from the
    current initial pool remains available for calibration-building runs.
    """

    micro = index_micro_scores(micro_records)
    macro = aggregate_prompt_macro(macro_records, metric=metric)

    source_ids = [
        str(row.get("id", row.get("prompt_id")))
        for row in proxy_rows
    ]
    initial_ids = [f"{source_id}::initial" for source_id in source_ids]

    for prompt_id in initial_ids:
        if prompt_id not in micro or prompt_id not in macro:
            raise ValueError(f"Missing initial hard score for {prompt_id!r}")

    if (micro_calibration is None) != (macro_calibration is None):
        raise ValueError(
            "micro_calibration and macro_calibration must be provided "
            "together"
        )
    if micro_calibration is None:
        micro_calibration = fit_robust_calibration(
            micro[prompt_id] for prompt_id in initial_ids
        )
        macro_calibration = fit_robust_calibration(
            macro[prompt_id] for prompt_id in initial_ids
        )
        calibration_source = "fitted_current_initial_pool"
    else:
        calibration_source = "provided_fixed_calibration"
    objective = RobustHybridPromptObjective(
        micro_calibration=micro_calibration,
        macro_calibration=macro_calibration,
        micro_weight=micro_weight,
        macro_weight=macro_weight,
        weighting_mode=weighting_mode,
        hybrid_beta=hybrid_beta,
    )

    outputs: list[dict[str, Any]] = []
    for source_id, source_row in zip(source_ids, proxy_rows, strict=True):
        initial_id = f"{source_id}::initial"
        optimized_id = f"{source_id}::optimized"
        if optimized_id not in micro or optimized_id not in macro:
            raise ValueError(
                f"Missing optimized hard score for {optimized_id!r}"
            )

        initial = objective.score_hard(
            micro_raw=micro[initial_id],
            macro_raw=macro[initial_id],
        )
        optimized = objective.score_hard(
            micro_raw=micro[optimized_id],
            macro_raw=macro[optimized_id],
        )
        optimization = dict(source_row.get("optimization", {}))
        proxy_accepted = bool(
            optimization.get(
                "proxy_accepted",
                optimization.get("accepted", False),
            )
        )
        gain = optimized.objective - initial.objective
        accepted = proxy_accepted and gain > float(minimum_gain)
        decision = HybridCandidateDecision(
            prompt_id=source_id,
            proxy_accepted=proxy_accepted,
            accepted=accepted,
            initial_objective=initial.objective,
            optimized_objective=optimized.objective,
            objective_gain=gain,
            initial_micro=initial.micro_raw,
            optimized_micro=optimized.micro_raw,
            initial_macro=initial.macro_raw,
            optimized_macro=optimized.macro_raw,
            initial_alpha=initial.micro_weight,
            optimized_alpha=optimized.micro_weight,
            weighting_mode=optimized.weighting_mode,
        )

        output = dict(source_row)
        optimization["proxy_accepted"] = proxy_accepted
        optimization["accepted"] = accepted
        optimization["acceptance_stage"] = "full_hybrid_hard_validation"
        optimization["hybrid_validation"] = decision.to_dict()
        output["optimization"] = optimization
        outputs.append(output)

    report = {
        "weighting_mode": weighting_mode,
        "hybrid_beta": float(hybrid_beta),
        "macro_metric": metric,
        "minimum_gain": float(minimum_gain),
        "calibration_source": calibration_source,
        "micro_calibration": micro_calibration.to_dict(),
        "macro_calibration": macro_calibration.to_dict(),
        "rows": len(outputs),
        "accepted": sum(
            bool(row["optimization"]["accepted"])
            for row in outputs
        ),
    }
    return outputs, report
