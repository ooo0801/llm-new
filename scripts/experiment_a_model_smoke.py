from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from _bootstrap import ROOT
from llm_integrity.config import load_config
from llm_integrity.modeling import generate_texts, load_model


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def normalized_devices(device_map: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for value in device_map.values():
        if isinstance(value, int):
            result.add(value)
        elif isinstance(value, str) and value.startswith("cuda:"):
            result.add(int(value.split(":", 1)[1]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment_a_qwen14b.yaml"
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/prompt-transfer-14b-a1/results/"
            "model_smoke.json"
        ),
    )
    args = parser.parse_args()
    if torch.cuda.device_count() != 3:
        raise RuntimeError(f"expected 3 GPUs, found {torch.cuda.device_count()}")
    # Each audit runs in a fresh process, so allocator peaks start at zero. Resetting
    # unopened devices is invalid on the Torch build used by the experiment server.
    torch.cuda.init()

    config = load_config(ROOT / args.config)
    output = ROOT / args.output
    started = time.time()
    bundle = load_model(config["model"])
    try:
        device_map = dict(getattr(bundle.model, "hf_device_map", {}))
        devices = normalized_devices(device_map)
        if devices != {0, 1, 2}:
            raise RuntimeError(f"expected GPUs 0/1/2 in device map, got {devices}")
        forbidden = {
            str(value)
            for value in device_map.values()
            if str(value) in {"cpu", "disk"}
        }
        if forbidden:
            raise RuntimeError(f"CPU/disk offload detected: {forbidden}")
        generated = generate_texts(
            bundle,
            ["请只回答：连接正常"],
            {
                "max_input_tokens": 64,
                "max_new_tokens": 4,
                "do_sample": False,
                "system_prompt": None,
            },
        )[0]
        report = {
            "schema_version": "experiment_a_model_smoke_1.0",
            "model_id": bundle.name,
            "revision": bundle.revision,
            "dtype": str(next(bundle.model.parameters()).dtype),
            "device_map": device_map,
            "used_gpu_indices": sorted(devices),
            "cpu_or_disk_offload": False,
            "generated_text": generated,
            "generated_nonempty": bool(generated.strip()),
            "elapsed_seconds": time.time() - started,
            "gpu_memory": gpu_memory(),
            "passed": bool(generated.strip()),
        }
        atomic_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
