from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from _bootstrap import ROOT
from llm_integrity.activations import TransformerActivationProfiler
from llm_integrity.config import load_config
from llm_integrity.modeling import generate_texts, load_model, next_token_logits
from llm_integrity.paper_blockwise_micro import estimate_prompt_jacobian_blockwise


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    temporary.replace(path)


def gpu_memory() -> list[dict[str, int]]:
    return [
        {
            "gpu": index,
            "allocated": int(torch.cuda.memory_allocated(index)),
            "reserved": int(torch.cuda.memory_reserved(index)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


def devices(device_map: dict[str, Any]) -> set[int]:
    result = set()
    for value in device_map.values():
        if isinstance(value, int):
            result.add(value)
        elif isinstance(value, str) and value.startswith("cuda:"):
            result.add(int(value.split(":", 1)[1]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_h6_qwen32b_development.yaml")
    parser.add_argument("--output", default="results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/engineering_smoke.json")
    parser.add_argument("--prompt", default="请只回答：连接正常")
    args = parser.parse_args()
    output = ROOT / args.output
    report: dict[str, Any] = {"schema_version": "experiment_h6_engineering_smoke_1.0", "status": "running", "stages": {}, "started_at_unix": time.time()}
    atomic_json(output, report)
    bundle = None
    try:
        if torch.cuda.device_count() != 3:
            raise RuntimeError(f"expected 3 GPUs, found {torch.cuda.device_count()}")
        torch.cuda.init()
        config = load_config(ROOT / args.config)
        bundle = load_model(config["model"])
        device_map = dict(getattr(bundle.model, "hf_device_map", {}))
        used = devices(device_map)
        forbidden = sorted({str(value) for value in device_map.values() if str(value) in {"cpu", "disk"}})
        if used != {0, 1, 2} or forbidden:
            raise RuntimeError(f"invalid device map: GPUs={sorted(used)} forbidden={forbidden}")
        report["stages"]["load"] = {"passed": True, "device_map": device_map, "used_gpu_indices": sorted(used), "cpu_or_disk_offload": False, "gpu_memory": gpu_memory()}
        atomic_json(output, report)

        generation = {"max_input_tokens": 64, "max_new_tokens": 4, "do_sample": False, "system_prompt": None}
        first = generate_texts(bundle, [args.prompt], generation, seed=2026082019)[0]
        second = generate_texts(bundle, [args.prompt], generation, seed=2026082019)[0]
        if not first.strip() or first.encode("utf-8") != second.encode("utf-8"):
            raise RuntimeError("same-seed deterministic generation replay failed")
        report["stages"]["generation_replay"] = {"passed": True, "first": first, "second": second, "byte_exact": True, "gpu_memory": gpu_memory()}
        atomic_json(output, report)

        before = next_token_logits(bundle, [args.prompt], max_length=64)
        profiler = TransformerActivationProfiler(attention_entropy_fraction=0.70, ffn_quantile=0.95, residual_threshold=0.50, max_ffn_units_per_layer=128)
        profile = profiler.profile(bundle, "h6_engineering_smoke", args.prompt, 64, None).as_dict()
        after = next_token_logits(bundle, [args.prompt], max_length=64)
        max_difference = float((before - after).abs().max().item())
        if max_difference != 0.0 or not profile.get("components"):
            raise RuntimeError(f"activation hook gate failed: components={len(profile.get('components', []))} max_diff={max_difference}")
        report["stages"]["activation_hook"] = {"passed": True, "components": len(profile["components"]), "max_abs_logit_difference": max_difference, "gpu_memory": gpu_memory()}
        atomic_json(output, report)

        micro = estimate_prompt_jacobian_blockwise(bundle, args.prompt, max_length=64, probes=1, seed=2026082010, show_progress=True)
        if not micro.complete_parameter_coverage or not (micro.estimate > 0):
            raise RuntimeError("complete blockwise Hutchinson gate failed")
        report["stages"]["complete_blockwise_micro"] = {"passed": True, "estimate": micro.estimate, "estimate_per_parameter": micro.estimate_per_parameter, "parameter_count": micro.parameter_count, "groups": len(micro.groups), "complete_parameter_coverage": micro.complete_parameter_coverage, "gpu_memory": gpu_memory()}
        report["status"] = "complete_go"
        report["passed"] = True
        report["finished_at_unix"] = time.time()
        atomic_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report["status"] = "engineering_no_go"
        report["passed"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["gpu_memory"] = gpu_memory() if torch.cuda.is_available() else []
        report["finished_at_unix"] = time.time()
        atomic_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    finally:
        if bundle is not None:
            bundle.close()


if __name__ == "__main__":
    raise SystemExit(main())
