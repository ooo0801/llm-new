from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .modeling import ModelBundle, load_model


@dataclass(frozen=True)
class QuantizationLoadReport:
    variant_id: str
    requested_method: str
    realized_method: str
    exact_requested_method: bool
    compute_dtype: str
    double_quant: bool
    int8_threshold: float | None
    target_scope: str
    loaded_in_8bit: bool
    loaded_in_4bit: bool
    quantized_linear_modules: int
    quantized_module_examples: tuple[str, ...]
    memory_gib: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _configuration(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = manifest.get("configuration", manifest)

    if not isinstance(configuration, Mapping):
        raise TypeError("Quantization configuration must be a mapping")

    return dict(configuration)


def _normalize_dtype(name: str) -> str:
    normalized = str(name).lower()

    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
    }

    if normalized not in aliases:
        raise ValueError(
            "Quantized compute dtype must be bfloat16 or float16, "
            f"got {name!r}"
        )

    return aliases[normalized]


def _build_model_config(
    base_model_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bool, tuple[str, ...]]:
    method = str(configuration.get("method", "int8")).lower()
    compute_dtype = _normalize_dtype(
        str(configuration.get("compute_dtype", "bfloat16"))
    )
    double_quant = bool(configuration.get("double_quant", False))
    notes: list[str] = []

    model_config = deepcopy(dict(base_model_config))
    model_config["compute_dtype"] = compute_dtype

    if method == "int8":
        model_config["quantization"] = "int8"
        int8_threshold = float(configuration.get("llm_int8_threshold", 6.0))
        if not math.isfinite(int8_threshold) or int8_threshold <= 0.0:
            raise ValueError("INT8 threshold must be finite and positive")
        model_config["llm_int8_threshold"] = int8_threshold
        realized_method = "bitsandbytes_int8"
        exact_requested_method = True

        if double_quant:
            notes.append(
                "double_quant does not apply to INT8 and was ignored"
            )

    elif method == "nf4":
        model_config["quantization"] = "int4"
        model_config["quant_type"] = "nf4"
        model_config["double_quant"] = double_quant
        realized_method = "bitsandbytes_nf4"
        exact_requested_method = True

    elif method == "fp4":
        model_config["quantization"] = "int4"
        model_config["quant_type"] = "fp4"
        model_config["double_quant"] = double_quant
        realized_method = "bitsandbytes_fp4"
        exact_requested_method = True

    elif method in {"int4", "4bit"}:
        # bitsandbytes并不提供严格的整数INT4量化类型。
        # 为兼容旧配置，明确映射到NF4，并在报告中标记为非精确实现。
        model_config["quantization"] = "int4"
        model_config["quant_type"] = "nf4"
        model_config["double_quant"] = double_quant
        realized_method = "bitsandbytes_nf4"
        exact_requested_method = False
        notes.append(
            "Requested INT4 was realized as bitsandbytes NF4. "
            "This is a 4-bit quantized model, but not strict integer INT4."
        )

    elif method == "mixed_int4_int8":
        raise NotImplementedError(
            "mixed_int4_int8 cannot be truthfully implemented by one "
            "standard BitsAndBytesConfig load. Remove this variant from "
            "the formal manifest or implement a dedicated mixed backend."
        )

    else:
        raise ValueError(
            f"Unsupported quantization method: {method!r}"
        )

    return (
        model_config,
        realized_method,
        exact_requested_method,
        tuple(notes),
    )


def inspect_quantized_model(
    bundle: ModelBundle,
) -> tuple[bool, bool, list[str], float]:
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required to inspect a quantized model"
        ) from exc

    loaded_in_8bit = bool(
        getattr(bundle.model, "is_loaded_in_8bit", False)
    )
    loaded_in_4bit = bool(
        getattr(bundle.model, "is_loaded_in_4bit", False)
    )

    quantized_names: list[str] = []

    for name, module in bundle.model.named_modules():
        if isinstance(
            module,
            (
                bnb.nn.Linear8bitLt,
                bnb.nn.Linear4bit,
            ),
        ):
            quantized_names.append(name)

    if hasattr(bundle.model, "get_memory_footprint"):
        memory_bytes = int(bundle.model.get_memory_footprint())
    else:
        memory_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in bundle.model.parameters()
        )

    memory_gib = memory_bytes / (1024 ** 3)

    return (
        loaded_in_8bit,
        loaded_in_4bit,
        quantized_names,
        float(memory_gib),
    )


def load_quantized_manifest_variant(
    base_model_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[ModelBundle, QuantizationLoadReport]:
    family = str(
        manifest.get(
            "family",
            _configuration(manifest).get("type", ""),
        )
    ).lower()

    if family != "quantization":
        raise ValueError(
            f"Expected a quantization manifest, got {family!r}"
        )

    configuration = _configuration(manifest)
    target_scope = str(
        configuration.get("target_scope", "full_model")
    ).lower()

    if target_scope not in {
        "full_model",
        "transformer_linear",
    }:
        raise NotImplementedError(
            "The current bitsandbytes backend quantizes the transformer's "
            "supported linear modules during loading and cannot guarantee "
            f"the requested partial scope {target_scope!r}."
        )

    (
        model_config,
        realized_method,
        exact_requested_method,
        notes,
    ) = _build_model_config(
        base_model_config,
        configuration,
    )

    bundle = load_model(model_config)

    (
        loaded_in_8bit,
        loaded_in_4bit,
        quantized_names,
        memory_gib,
    ) = inspect_quantized_model(bundle)

    requested_method = str(
        configuration.get("method", "int8")
    ).lower()

    if requested_method == "int8" and not loaded_in_8bit:
        bundle.close()
        raise RuntimeError(
            "The model was requested in INT8, but the loaded model "
            "does not report is_loaded_in_8bit=True"
        )

    if requested_method in {
        "int4",
        "4bit",
        "nf4",
        "fp4",
    } and not loaded_in_4bit:
        bundle.close()
        raise RuntimeError(
            "The model was requested in 4-bit mode, but the loaded model "
            "does not report is_loaded_in_4bit=True"
        )

    if not quantized_names:
        bundle.close()
        raise RuntimeError(
            "No bitsandbytes quantized linear modules were found"
        )

    report = QuantizationLoadReport(
        variant_id=str(manifest.get("variant_id", "unknown")),
        requested_method=requested_method,
        realized_method=realized_method,
        exact_requested_method=exact_requested_method,
        compute_dtype=_normalize_dtype(
            str(configuration.get("compute_dtype", "bfloat16"))
        ),
        double_quant=bool(
            configuration.get("double_quant", False)
        ),
        int8_threshold=(
            float(configuration.get("llm_int8_threshold", 6.0))
            if requested_method == "int8"
            else None
        ),
        target_scope=target_scope,
        loaded_in_8bit=loaded_in_8bit,
        loaded_in_4bit=loaded_in_4bit,
        quantized_linear_modules=len(quantized_names),
        quantized_module_examples=tuple(
            quantized_names[:20]
        ),
        memory_gib=memory_gib,
        notes=notes,
    )

    return bundle, report
