from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .modeling import ModelBundle, load_model
from .paper_in_memory_attacks import (
    apply_manifest_in_memory,
    execution_mode,
)
from .paper_quantization import (
    load_quantized_manifest_variant,
)


@dataclass(frozen=True)
class VariantExecutionReport:
    """Unified traceability report for one realized attack variant."""

    variant_id: str
    family: str
    execution_mode: str
    realized_method: str
    isolated_base_reload: bool
    details: dict[str, Any]


@dataclass
class LoadedPaperVariant:
    """
    A loaded model variant and its realization report.

    Every instance owns its ModelBundle. Call close() after extracting
    the required model outputs so GPU memory can be released.
    """

    bundle: ModelBundle
    report: VariantExecutionReport
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.bundle.close()
            self._closed = True

    def __enter__(self) -> "LoadedPaperVariant":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise TypeError("manifest['configuration'] must be a mapping")
    return dict(configuration)


def _family(manifest: Mapping[str, Any]) -> str:
    configuration = _configuration(manifest)
    family = str(
        manifest.get(
            "family",
            configuration.get("type", ""),
        )
    ).lower()

    if not family:
        raise ValueError("Attack manifest has no family or configuration type")

    return family


def _variant_id(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("variant_id", "unknown"))


def _resolve_adapter_path(
    manifest: Mapping[str, Any],
    adapter_path: str | Path | None,
) -> Path:
    """
    Resolve a previously trained LoRA adapter.

    Training is intentionally kept separate from evaluation so a validation
    or test run cannot silently train on evaluation data.
    """

    configuration = _configuration(manifest)

    candidate = (
        adapter_path
        or manifest.get("adapter_path")
        or configuration.get("adapter_path")
    )

    if candidate is None:
        raise FileNotFoundError(
            "This finetuning manifest requires a trained LoRA adapter. "
            "Train it with train_lora_manifest_variant() and pass the "
            "resulting adapter_path explicitly."
        )

    path = Path(candidate).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"LoRA adapter path is not a directory: {path}")

    return path


def _load_finetuned_variant(
    base_model_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    adapter_path: str | Path | None,
) -> LoadedPaperVariant:
    """Load an already trained LoRA adapter over a fresh base model."""

    from peft import PeftModel

    resolved_adapter = _resolve_adapter_path(
        manifest,
        adapter_path,
    )

    bundle = load_model(dict(base_model_config))

    try:
        bundle.model = PeftModel.from_pretrained(
            bundle.model,
            str(resolved_adapter),
            is_trainable=False,
        )
        bundle.model.eval()

        trainable_parameters = sum(
            parameter.numel()
            for parameter in bundle.model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(
            parameter.numel()
            for parameter in bundle.model.parameters()
        )

        report = VariantExecutionReport(
            variant_id=_variant_id(manifest),
            family=_family(manifest),
            execution_mode="adapter_reload",
            realized_method="lora",
            isolated_base_reload=True,
            details={
                "adapter_path": str(resolved_adapter),
                "trainable_parameters_during_evaluation": trainable_parameters,
                "total_parameters": total_parameters,
                "adapter_loaded": True,
            },
        )

        return LoadedPaperVariant(
            bundle=bundle,
            report=report,
        )

    except Exception:
        bundle.close()
        raise


def load_manifest_variant(
    base_model_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    adapter_path: str | Path | None = None,
) -> LoadedPaperVariant:
    """
    Materialize one isolated model variant described by a paper manifest.

    Lifecycle rules
    ---------------
    in_memory:
        Load a fresh base model, then apply pruning or Gaussian noise.

    reload:
        Load the base checkpoint directly through the requested real
        quantization backend.

    training:
        Load a previously trained LoRA adapter over a fresh base model.
        Training itself is deliberately performed by paper_finetuning.py.
    """

    family = _family(manifest)
    mode = execution_mode(manifest)

    if mode == "in_memory":
        bundle = load_model(dict(base_model_config))

        try:
            attack_report = apply_manifest_in_memory(
                bundle.model,
                manifest,
            )

            report = VariantExecutionReport(
                variant_id=_variant_id(manifest),
                family=family,
                execution_mode=mode,
                realized_method=attack_report.name,
                isolated_base_reload=True,
                details=asdict(attack_report),
            )

            return LoadedPaperVariant(
                bundle=bundle,
                report=report,
            )

        except Exception:
            bundle.close()
            raise

    if mode == "reload":
        bundle, quantization_report = load_quantized_manifest_variant(
            dict(base_model_config),
            manifest,
        )

        report = VariantExecutionReport(
            variant_id=_variant_id(manifest),
            family=family,
            execution_mode=mode,
            realized_method=quantization_report.realized_method,
            isolated_base_reload=True,
            details=asdict(quantization_report),
        )

        return LoadedPaperVariant(
            bundle=bundle,
            report=report,
        )

    if mode == "training":
        return _load_finetuned_variant(
            base_model_config,
            manifest,
            adapter_path,
        )

    raise ValueError(
        f"Unsupported execution mode {mode!r} "
        f"for variant {_variant_id(manifest)!r}"
    )
