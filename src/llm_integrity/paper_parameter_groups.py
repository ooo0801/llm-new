from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from torch.nn import Parameter


_LAYER_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<layer_index>\d+)\.(?P<suffix>.+)$"
)


@dataclass(frozen=True)
class ParameterGroup:
    """
    一个互不重叠的模型参数组件。

    attributes
    ----------
    name:
        分组的唯一名称，例如layer_00_attention。

    component:
        参数组件类别，例如attention、ffn、normalization。

    layer_index:
        Transformer层编号。非逐层参数为None。

    named_parameters:
        该组包含的参数名称和参数对象。

    parameter_count:
        该组参数标量总数。
    """

    name: str
    component: str
    layer_index: int | None
    named_parameters: tuple[tuple[str, Parameter], ...]
    parameter_count: int


def classify_parameter_name(
    parameter_name: str,
) -> tuple[str, str, int | None]:
    """
    根据参数名称判断其所在组件。

    返回：
        (group_name, component_name, layer_index)
    """
    layer_match = _LAYER_PATTERN.search(parameter_name)

    if layer_match is not None:
        layer_index = int(
            layer_match.group("layer_index")
        )
        suffix = layer_match.group("suffix")

        if suffix.startswith("self_attn."):
            component = "attention"
        elif suffix.startswith("mlp."):
            component = "ffn"
        elif (
            "layernorm" in suffix.lower()
            or suffix.startswith("norm.")
        ):
            component = "normalization"
        else:
            component = "layer_other"

        group_name = (
            f"layer_{layer_index:02d}_{component}"
        )

        return group_name, component, layer_index

    lowered = parameter_name.lower()

    if (
        "embed_tokens" in lowered
        or ".wte." in lowered
        or lowered.startswith("wte.")
    ):
        return "token_embeddings", "embedding", None

    if (
        lowered == "lm_head.weight"
        or lowered.startswith("lm_head.")
        or ".lm_head." in lowered
    ):
        return "lm_head", "output", None

    if (
        lowered == "model.norm.weight"
        or lowered.startswith("model.norm.")
        or lowered.endswith(".final_layernorm.weight")
        or ".final_layernorm." in lowered
    ):
        return "final_normalization", "normalization", None

    return "other_parameters", "other", None


def build_parameter_groups(
    model: Any,
) -> list[ParameterGroup]:
    """
    将模型全部可微参数划分为互不重叠的组件组。

    该函数保证：

    1. 每个参数只进入一个分组；
    2. 没有参数被遗漏；
    3. 只接受浮点或复数参数；
    4. 分组顺序稳定，便于复现实验。
    """
    grouped: dict[
        str,
        dict[str, Any],
    ] = {}

    all_named_parameters = list(
        model.named_parameters()
    )

    if not all_named_parameters:
        raise ValueError(
            "The model does not contain any parameters."
        )

    seen_parameter_ids: set[int] = set()

    for parameter_name, parameter in all_named_parameters:
        parameter_id = id(parameter)

        if parameter_id in seen_parameter_ids:
            raise ValueError(
                "The same parameter object appeared more than once: "
                f"{parameter_name}"
            )

        seen_parameter_ids.add(parameter_id)

        if not (
            parameter.is_floating_point()
            or parameter.is_complex()
        ):
            raise TypeError(
                "Micro-sensitivity requires differentiable floating-point "
                f"parameters, but {parameter_name} has dtype "
                f"{parameter.dtype}."
            )

        (
            group_name,
            component,
            layer_index,
        ) = classify_parameter_name(parameter_name)

        if group_name not in grouped:
            grouped[group_name] = {
                "component": component,
                "layer_index": layer_index,
                "named_parameters": [],
            }

        grouped[group_name]["named_parameters"].append(
            (parameter_name, parameter)
        )

    component_order = {
        "embedding": 0,
        "attention": 1,
        "ffn": 2,
        "normalization": 3,
        "layer_other": 4,
        "output": 5,
        "other": 6,
    }

    def sort_key(
        item: tuple[str, dict[str, Any]],
    ) -> tuple[int, int, str]:
        group_name, information = item

        layer_index = information["layer_index"]
        component = information["component"]

        effective_layer_index = (
            -1
            if component == "embedding"
            else (
                layer_index
                if layer_index is not None
                else 1_000_000
            )
        )

        return (
            effective_layer_index,
            component_order.get(component, 99),
            group_name,
        )

    results: list[ParameterGroup] = []

    for group_name, information in sorted(
        grouped.items(),
        key=sort_key,
    ):
        named_parameters = tuple(
            information["named_parameters"]
        )

        parameter_count = sum(
            parameter.numel()
            for _, parameter in named_parameters
        )

        results.append(
            ParameterGroup(
                name=group_name,
                component=information["component"],
                layer_index=information["layer_index"],
                named_parameters=named_parameters,
                parameter_count=parameter_count,
            )
        )

    grouped_parameter_ids = {
        id(parameter)
        for group in results
        for _, parameter in group.named_parameters
    }

    if grouped_parameter_ids != seen_parameter_ids:
        missing = seen_parameter_ids - grouped_parameter_ids
        duplicated_or_unknown = (
            grouped_parameter_ids - seen_parameter_ids
        )

        raise RuntimeError(
            "Parameter grouping coverage check failed. "
            f"missing={len(missing)}, "
            f"unknown={len(duplicated_or_unknown)}"
        )

    grouped_parameter_count = sum(
        group.parameter_count
        for group in results
    )

    model_parameter_count = sum(
        parameter.numel()
        for _, parameter in all_named_parameters
    )

    if grouped_parameter_count != model_parameter_count:
        raise RuntimeError(
            "Parameter count mismatch after grouping: "
            f"grouped={grouped_parameter_count}, "
            f"model={model_parameter_count}"
        )

    return results


def summarize_parameter_groups(
    groups: list[ParameterGroup],
) -> dict[str, Any]:
    """
    生成便于写入JSON和实验日志的分组摘要。
    """
    component_totals: dict[str, int] = {}

    for group in groups:
        component_totals[group.component] = (
            component_totals.get(group.component, 0)
            + group.parameter_count
        )

    total_parameters = sum(
        group.parameter_count
        for group in groups
    )

    return {
        "group_count": len(groups),
        "total_parameters": total_parameters,
        "component_parameter_counts": component_totals,
        "groups": [
            {
                "name": group.name,
                "component": group.component,
                "layer_index": group.layer_index,
                "parameter_tensors": len(
                    group.named_parameters
                ),
                "parameter_count": group.parameter_count,
            }
            for group in groups
        ],
    }
