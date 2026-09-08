from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_precalibration import (
    build_h8_feature_schema,
    fit_family_balanced_scaler,
    global_continuous_exclusion_mask,
    h8_global_degenerate_bandwidth,
    h8_median_positive_pairwise_distance,
    h8_mmd2_unbiased_unequal,
)
from llm_integrity.modeling import ModelBundle, load_model, render_prompt
from llm_integrity.paper_finetuning import train_lora_manifest_variant
from llm_integrity.paper_in_memory_attacks import _selected_parameters, _torch_generator


DEFAULT_CONFIG = ROOT / "configs" / "h9_stage1_qwen05b_attack_stat_calibration.yaml"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]|\d+|[^\w\s]", re.UNICODE)
LEVEL_ORDER = {"weak": 0, "medium": 1, "strong": 2}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Stage 1 config must be a mapping")
    return value


def output_root(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["output"]["root"])


def endpoint_id(family: str, label: str, seed: int) -> str:
    return f"{family}_{label}_seed{seed}"


def attack_endpoints(config: Mapping[str, Any], family: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = [family] if family else ["gaussian", "lora"]
    for current in families:
        section = config[current]
        for strength in section["strengths"]:
            for seed in section["seeds"]:
                row = {"family": current, "strength": str(strength["label"]), "attack_seed": int(seed)}
                row.update({key: value for key, value in strength.items() if key != "label"})
                row["endpoint_id"] = endpoint_id(current, row["strength"], row["attack_seed"])
                rows.append(row)
    return rows


def generation_seeds(config: Mapping[str, Any], count: int, prompt_index: int) -> list[int]:
    base = int(config["generation"]["response_seed_base"])
    return [base + prompt_index * 1000 + index for index in range(count)]


def _set_seed(seed: int) -> None:
    import random
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def generate_one(bundle: ModelBundle, row: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    import torch

    generation = config["generation"]
    rendered = render_prompt(bundle.tokenizer, str(row["prompt"]), generation.get("system_prompt"))
    encoded = bundle.tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=int(generation["max_input_tokens"]),
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": bool(generation["do_sample"]),
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs.update(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
        )
    _set_seed(seed)
    with torch.inference_mode():
        generated = bundle.model.generate(**encoded, **kwargs)
    input_length = int(encoded["input_ids"].shape[1])
    completion_ids = generated[0, input_length:].detach().cpu().tolist()
    text = bundle.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "response": text,
        "response_sha256": sha256_bytes(text.encode("utf-8")),
        "completion_token_ids": completion_ids,
        "completion_token_count": len(completion_ids),
    }


def close_bundle(bundle: ModelBundle | None) -> None:
    if bundle is not None:
        bundle.close()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for row in read_jsonl(path):
        keys.add((str(row["prompt_id"]), int(row["response_index"])))
    return keys


def sample_bundle(
    bundle: ModelBundle,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    path: Path,
    repetitions: int,
    endpoint: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(path)
    expected = len(prompts) * repetitions
    if len(done) == expected:
        print(f"[sample] complete {endpoint['endpoint_id']} rows={expected}", flush=True)
        return
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for prompt_index, prompt in enumerate(prompts):
            for response_index, seed in enumerate(generation_seeds(config, repetitions, prompt_index)):
                key = (str(prompt["id"]), response_index)
                if key in done:
                    continue
                generated = generate_one(bundle, prompt, config, seed)
                record = {
                    "schema_version": config["schema_version"],
                    "source_commit": config["source_commit"],
                    "endpoint_id": endpoint["endpoint_id"],
                    "family": endpoint["family"],
                    "strength": endpoint.get("strength"),
                    "attack_seed": endpoint.get("attack_seed"),
                    "prompt_index": prompt_index,
                    "prompt_id": prompt["id"],
                    "variant_role": prompt.get("variant_role"),
                    "task_family": prompt.get("task_family"),
                    "response_index": response_index,
                    "generation_seed": seed,
                    "created_at_utc": utc_now(),
                    **generated,
                }
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                done.add(key)
                if len(done) % 24 == 0 or len(done) == expected:
                    print(f"[sample] {endpoint['endpoint_id']} {len(done)}/{expected}", flush=True)


def apply_gaussian_with_audit(model: Any, std_ratio: float, seed: int, target_scope: str) -> dict[str, Any]:
    import torch

    selected = _selected_parameters(model, target_scope)
    generators: dict[str, Any] = {}
    base_sq = 0.0
    delta_sq = 0.0
    nominal_sq = 0.0
    selected_parameters = 0
    effective_changed = 0
    for _, parameter in selected:
        selected_parameters += parameter.numel()
        base_sq += float(parameter.detach().float().square().sum().item())
        scale = float(parameter.detach().float().std().item()) * float(std_ratio)
        if scale == 0.0:
            continue
        device_key = str(parameter.device)
        if device_key not in generators:
            generators[device_key] = _torch_generator(parameter.device, seed)
        noise_fp32 = torch.randn(
            parameter.shape,
            generator=generators[device_key],
            device=parameter.device,
            dtype=torch.float32,
        ) * scale
        nominal_sq += float(noise_fp32.square().sum().item())
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(noise_fp32.to(parameter.dtype))
        delta = parameter.detach().float() - before.float()
        delta_sq += float(delta.square().sum().item())
        effective_changed += int(torch.count_nonzero(delta).item())
        del before, delta, noise_fp32
    return {
        "family": "gaussian",
        "attack_seed": seed,
        "std_ratio": float(std_ratio),
        "target_scope": target_scope,
        "selected_tensor_count": len(selected),
        "selected_parameter_count": selected_parameters,
        "effective_changed_parameter_count": effective_changed,
        "effective_changed_fraction": effective_changed / selected_parameters,
        "nominal_relative_frobenius": math.sqrt(nominal_sq / base_sq),
        "actual_relative_frobenius": math.sqrt(delta_sq / base_sq),
        "bf16_retention_ratio": math.sqrt(delta_sq / nominal_sq) if nominal_sq else 0.0,
    }


def adapter_audit(model: Any) -> dict[str, Any]:
    import torch

    delta_sq = 0.0
    base_sq = 0.0
    active_modules = 0
    nonzero_delta = 0
    delta_values = 0
    for module in model.modules():
        if not hasattr(module, "lora_A") or not getattr(module, "lora_A"):
            continue
        for adapter_name in module.lora_A.keys():
            a = module.lora_A[adapter_name].weight.detach().float()
            b = module.lora_B[adapter_name].weight.detach().float()
            scaling = float(module.scaling[adapter_name])
            delta = (b @ a) * scaling
            base = module.base_layer.weight.detach().float()
            delta_sq += float(delta.square().sum().item())
            base_sq += float(base.square().sum().item())
            nonzero_delta += int(torch.count_nonzero(delta).item())
            delta_values += delta.numel()
            active_modules += 1
    if active_modules == 0 or base_sq == 0.0:
        raise RuntimeError("Loaded adapter has no active LoRA modules")
    return {
        "active_lora_modules": active_modules,
        "delta_frobenius": math.sqrt(delta_sq),
        "target_base_frobenius": math.sqrt(base_sq),
        "relative_adapter_delta_frobenius": math.sqrt(delta_sq / base_sq),
        "nonzero_delta_fraction": nonzero_delta / delta_values,
    }


def heldout_token_nll(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], max_length: int) -> float:
    import torch

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for row in rows:
        prompt = render_prompt(tokenizer, str(row["prompt"]), None)
        answer = str(row["expected_answer"]).strip() + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            prompt + answer,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        labels = encoded["input_ids"].clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        token_count = int(torch.count_nonzero(labels != -100).item())
        if token_count == 0:
            continue
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)
        with torch.inference_mode():
            loss = model(**encoded, labels=labels).loss
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise RuntimeError("Held-out LoRA audit has no supervised tokens")
    return total_loss / total_tokens


def train_lora(config: Mapping[str, Any]) -> None:
    root = output_root(config)
    adapter_root = root / "adapters"
    train_path = resolve_path(config["inputs"]["lora_train"])
    for endpoint in attack_endpoints(config, "lora"):
        adapter_path = adapter_root / endpoint["endpoint_id"]
        report_path = adapter_path / "training_report.json"
        if report_path.exists():
            print(f"[train] complete {endpoint['endpoint_id']}", flush=True)
            continue
        section = config["lora"]
        variant = {
            "variant_id": endpoint["endpoint_id"],
            "family": "finetuning",
            "seed": endpoint["attack_seed"],
            "configuration": {
                "method": "lora",
                "rank": int(section["rank"]),
                "alpha": int(section["alpha"]),
                "dropout": float(section["dropout"]),
                "learning_rate": float(section["learning_rate"]),
                "steps": int(endpoint["steps"]),
                "target_scope": str(section["target_scope"]),
            },
        }
        report = train_lora_manifest_variant(
            config["model"],
            variant,
            data_path=train_path,
            output_root=adapter_root,
            max_length=int(section["max_length"]),
            batch_size=int(section["batch_size"]),
            gradient_accumulation_steps=int(section["gradient_accumulation_steps"]),
        )
        print(f"[train] {report.variant_id} loss={report.training_loss}", flush=True)


def sample_intact(config: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> None:
    root = output_root(config)
    endpoint = {"endpoint_id": "intact", "family": "intact"}
    bundle: ModelBundle | None = None
    try:
        bundle = load_model(config["model"])
        heldout = read_jsonl(resolve_path(config["inputs"]["lora_heldout"]))
        audit_path = root / "audits" / "intact.json"
        if not audit_path.exists():
            write_json(
                audit_path,
                {
                    "endpoint_id": "intact",
                    "heldout_token_nll": heldout_token_nll(
                        bundle.model, bundle.tokenizer, heldout, int(config["lora"]["max_length"])
                    ),
                    "created_at_utc": utc_now(),
                },
            )
        sample_bundle(
            bundle,
            prompts,
            config,
            root / "responses" / "intact.jsonl",
            int(config["generation"]["intact_repetitions_per_prompt"]),
            endpoint,
        )
    finally:
        close_bundle(bundle)


def sample_gaussian(config: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> None:
    root = output_root(config)
    for endpoint in attack_endpoints(config, "gaussian"):
        bundle: ModelBundle | None = None
        try:
            bundle = load_model(config["model"])
            audit = apply_gaussian_with_audit(
                bundle.model,
                float(endpoint["std_ratio"]),
                int(endpoint["attack_seed"]),
                str(config["gaussian"]["target_scope"]),
            )
            audit.update(endpoint)
            write_json(root / "audits" / f"{endpoint['endpoint_id']}.json", audit)
            sample_bundle(
                bundle,
                prompts,
                config,
                root / "responses" / f"{endpoint['endpoint_id']}.jsonl",
                int(config["generation"]["attack_repetitions_per_prompt"]),
                endpoint,
            )
        finally:
            close_bundle(bundle)


def sample_lora(config: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> None:
    from peft import PeftModel

    root = output_root(config)
    heldout = read_jsonl(resolve_path(config["inputs"]["lora_heldout"]))
    for endpoint in attack_endpoints(config, "lora"):
        bundle: ModelBundle | None = None
        try:
            bundle = load_model(config["model"])
            adapter_path = root / "adapters" / endpoint["endpoint_id"]
            if not (adapter_path / "adapter_model.safetensors").exists():
                raise FileNotFoundError(f"Missing trained adapter: {adapter_path}")
            bundle.model = PeftModel.from_pretrained(bundle.model, adapter_path, is_trainable=False)
            bundle.model.eval()
            audit = adapter_audit(bundle.model)
            audit.update(endpoint)
            audit["heldout_token_nll"] = heldout_token_nll(
                bundle.model, bundle.tokenizer, heldout, int(config["lora"]["max_length"])
            )
            training_report = json.loads((adapter_path / "training_report.json").read_text(encoding="utf-8"))
            audit["training_loss"] = training_report.get("training_loss")
            audit["trainable_parameters"] = training_report.get("trainable_parameters")
            write_json(root / "audits" / f"{endpoint['endpoint_id']}.json", audit)
            sample_bundle(
                bundle,
                prompts,
                config,
                root / "responses" / f"{endpoint['endpoint_id']}.jsonl",
                int(config["generation"]["attack_repetitions_per_prompt"]),
                endpoint,
            )
        finally:
            close_bundle(bundle)


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _tv(left: Counter[str], right: Counter[str]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 and right_total == 0:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left[key] / max(left_total, 1) - right[key] / max(right_total, 1)) for key in keys)


def prefix10_tv(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        raise ValueError("Response groups must be non-empty")
    left_tokens = [tokens(text) for text in left]
    right_tokens = [tokens(text) for text in right]
    values = []
    for position in range(10):
        a = Counter(row[position] if position < len(row) else "<PAD>" for row in left_tokens)
        b = Counter(row[position] if position < len(row) else "<PAD>" for row in right_tokens)
        values.append(_tv(a, b))
    return float(np.mean(values))


def _bigram_counts(texts: Sequence[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for text in texts:
        row = tokens(text)
        result.update(f"{a}\u241f{b}" for a, b in zip(row, row[1:]))
    return result


def bigram_js(left: Sequence[str], right: Sequence[str]) -> float:
    a = _bigram_counts(left)
    b = _bigram_counts(right)
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    p = np.asarray([a[key] for key in keys], dtype=np.float64)
    q = np.asarray([b[key] for key in keys], dtype=np.float64)
    p = p / max(float(p.sum()), 1.0)
    q = q / max(float(q.sum()), 1.0)
    m = 0.5 * (p + q)
    p_active = p > 0
    q_active = q > 0
    p_term = np.sum(p[p_active] * np.log(p[p_active] / m[p_active]))
    q_term = np.sum(q[q_active] * np.log(q[q_active] / m[q_active]))
    return float(0.5 * (p_term + q_term) / math.log(2.0))


def semantic_centroid_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).mean(axis=0)
    b = np.asarray(right, dtype=np.float64).mean(axis=0)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(1.0 - np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def raw_metrics(left_text: Sequence[str], right_text: Sequence[str], left_sem: np.ndarray, right_sem: np.ndarray) -> dict[str, float]:
    return {
        "prefix10_tv": prefix10_tv(left_text, right_text),
        "bigram_js": bigram_js(left_text, right_text),
        "semantic_centroid_distance": semantic_centroid_distance(left_sem, right_sem),
    }


def evaluate_task(text: str, row: Mapping[str, Any]) -> bool:
    evaluator = str(row.get("evaluator", "contains"))
    normalized = re.sub(r"\s+", "", text).lower()
    expected = row.get("expected_answer")
    contains = row.get("expected_contains") or []
    if isinstance(contains, str):
        contains = [contains]
    contains_ok = all(str(value).lower() in text.lower() for value in contains)
    if evaluator == "exact":
        return expected is not None and normalized == re.sub(r"\s+", "", str(expected)).lower()
    if evaluator in {"contains", "numeric"}:
        return expected is not None and re.sub(r"\s+", "", str(expected)).lower() in normalized
    if evaluator == "refusal":
        return bool(re.search(r"不能|无法|抱歉|不可以|can't|cannot|sorry", text, re.I)) and contains_ok
    if evaluator == "json":
        try:
            value = json.loads(text.strip())
            return isinstance(value, dict) and contains_ok
        except Exception:
            return False
    if evaluator == "python_syntax":
        try:
            ast.parse(text.replace("```python", "").replace("```", ""))
            return contains_ok
        except SyntaxError:
            return False
    if evaluator == "length_and_contains":
        return len(re.sub(r"\s+", "", text)) <= 25 and contains_ok
    return contains_ok


def disjoint_splits(count: int, group_size: int, total: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if 2 * group_size > count:
        raise ValueError("Null groups cannot be disjoint at requested size")
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(total):
        order = rng.permutation(count)
        result.append((order[:group_size], order[group_size : 2 * group_size]))
    return result


def calibrate_composite(
    null_rows: Sequence[Mapping[str, float]],
    evaluation_rows: Sequence[Mapping[str, float]],
    epsilon: float,
    alpha: float,
) -> dict[str, Any]:
    names = list(null_rows[0])
    fit = {name: np.asarray([row[name] for row in null_rows], dtype=np.float64) for name in names}
    evaluation = {name: np.asarray([row[name] for row in evaluation_rows], dtype=np.float64) for name in names}
    active = [name for name in names if float(fit[name].std(ddof=0)) >= epsilon]
    excluded = [name for name in names if name not in active]
    if not active:
        return {"active_components": [], "excluded_degenerate_components": excluded, "evaluable": False}
    means = {name: float(fit[name].mean()) for name in active}
    scales = {name: float(fit[name].std(ddof=0)) for name in active}
    fit_composite = np.max(np.stack([(fit[name] - means[name]) / scales[name] for name in active]), axis=0)
    eval_composite = np.max(
        np.stack(
            [(evaluation[name] - means[name]) / scales[name] for name in active],
            axis=0,
        ),
        axis=0,
    )
    threshold = float(np.quantile(fit_composite, 1.0 - alpha, method="higher"))
    evaluation_p = np.asarray(
        [empirical_upper_p(fit_composite, float(value)) for value in eval_composite],
        dtype=np.float64,
    )
    return {
        "active_components": active,
        "excluded_degenerate_components": excluded,
        "evaluable": True,
        "means": means,
        "scales": scales,
        "null_composite": fit_composite,
        "alpha": float(alpha),
        "threshold": threshold,
        "evaluation_false_positive_rate": float(np.mean(evaluation_p <= alpha)),
    }


def empirical_upper_p(null_values: np.ndarray, observed: float) -> float:
    return float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))


def observed_composite(calibration: Mapping[str, Any], observed: Mapping[str, float]) -> dict[str, Any]:
    if not calibration.get("evaluable"):
        return {"evaluable": False, "z": None, "p_value": None, "detected": None, "component_z": {}}
    component_z = {
        name: (float(observed[name]) - calibration["means"][name]) / calibration["scales"][name]
        for name in calibration["active_components"]
    }
    value = max(component_z.values())
    p_value = empirical_upper_p(np.asarray(calibration["null_composite"]), value)
    return {
        "evaluable": True,
        "z": float(value),
        "p_value": p_value,
        "detected": bool(p_value <= float(calibration["alpha"])),
        "component_z": component_z,
    }


def endpoint_records(root: Path, endpoint: str) -> list[dict[str, Any]]:
    path = root / "responses" / f"{endpoint}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def analyze(config: Mapping[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    root = output_root(config)
    prompts = read_jsonl(resolve_path(config["inputs"]["prompts"]))
    prompt_map = {str(row["id"]): {**row, "category": row.get("task_family")} for row in prompts}
    endpoints = attack_endpoints(config)
    intact_records = endpoint_records(root, "intact")
    all_records = list(intact_records)
    by_endpoint: dict[str, list[dict[str, Any]]] = {"intact": intact_records}
    for endpoint in endpoints:
        rows = endpoint_records(root, endpoint["endpoint_id"])
        by_endpoint[endpoint["endpoint_id"]] = rows
        all_records.extend(rows)

    extractor = FeatureExtractor(
        semantic_model_name=str(config["semantic_encoder"]["name"]),
        semantic_model_revision=str(config["semantic_encoder"]["revision"]),
        semantic_device=str(config["semantic_encoder"]["device"]),
        semantic_local_files_only=bool(config["semantic_encoder"]["local_files_only"]),
    )
    texts = [str(row["response"]) for row in all_records]
    feature_rows = [prompt_map[str(row["prompt_id"])] for row in all_records]
    features = extractor.transform(texts, feature_rows)
    if features.shape[1] != 528:
        raise RuntimeError(f"Expected 528 features, received {features.shape[1]}")
    feature_by_key = {
        (str(row["endpoint_id"]), str(row["prompt_id"]), int(row["response_index"])): features[index]
        for index, row in enumerate(all_records)
    }

    def grouped(endpoint: str, prompt_id: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        rows = sorted(
            [row for row in by_endpoint[endpoint] if str(row["prompt_id"]) == prompt_id],
            key=lambda row: int(row["response_index"]),
        )
        matrix = np.stack(
            [feature_by_key[(endpoint, prompt_id, int(row["response_index"]))] for row in rows]
        )
        return rows, matrix

    schema = build_h8_feature_schema(512)
    intact_matrices = {prompt_id: grouped("intact", prompt_id)[1] for prompt_id in prompt_map}
    pooled = np.concatenate(list(intact_matrices.values()), axis=0)
    exclusion = global_continuous_exclusion_mask(pooled, schema)
    scalers = {
        prompt_id: fit_family_balanced_scaler(prompt_id, matrix, pooled, schema, exclusion_mask=exclusion)
        for prompt_id, matrix in intact_matrices.items()
    }
    transformed_intact = {
        prompt_id: scalers[prompt_id].transform(matrix, schema) for prompt_id, matrix in intact_matrices.items()
    }
    global_bandwidth = h8_global_degenerate_bandwidth(transformed_intact)
    bandwidths: dict[str, dict[str, Any]] = {}
    for prompt_id, matrix in transformed_intact.items():
        try:
            bandwidths[prompt_id] = {
                "value": h8_median_positive_pairwise_distance(matrix),
                "source": "prompt_specific",
            }
        except ValueError:
            bandwidths[prompt_id] = {
                "value": float(global_bandwidth["sigma"]),
                "source": "global_degenerate_fallback",
            }

    null_cfg = config["null_calibration"]
    group_size = int(null_cfg["reference_group_size"])
    fit_count = int(null_cfg["fit_splits"])
    eval_count = int(null_cfg["evaluation_splits"])
    alpha = float(null_cfg["alpha"])
    epsilon = float(null_cfg["degeneracy_epsilon"])
    prompt_calibrations: dict[str, Any] = {}
    prompt_null_audit: list[dict[str, Any]] = []
    for prompt_index, prompt_id in enumerate(prompt_map):
        intact_rows, intact_features = grouped("intact", prompt_id)
        intact_texts = [str(row["response"]) for row in intact_rows]
        intact_sem = intact_features[:, 11:523]
        transformed = transformed_intact[prompt_id]
        splits = disjoint_splits(
            len(intact_rows),
            group_size,
            fit_count + eval_count,
            int(null_cfg["split_seed_base"]) + prompt_index,
        )
        raw_null: list[dict[str, float]] = []
        mmd_null: list[float] = []
        for left, right in splits:
            raw_null.append(
                raw_metrics(
                    [intact_texts[index] for index in left],
                    [intact_texts[index] for index in right],
                    intact_sem[left],
                    intact_sem[right],
                )
            )
            mmd_null.append(
                h8_mmd2_unbiased_unequal(
                    transformed[left], transformed[right], float(bandwidths[prompt_id]["value"])
                )
            )
        raw_calibration = calibrate_composite(raw_null[:fit_count], raw_null[fit_count:], epsilon, alpha)
        mmd_fit = np.asarray(mmd_null[:fit_count], dtype=np.float64)
        mmd_eval = np.asarray(mmd_null[fit_count:], dtype=np.float64)
        mmd_mean = float(mmd_fit.mean())
        mmd_std = float(mmd_fit.std(ddof=0))
        mmd_evaluable = mmd_std >= epsilon
        mmd_threshold = float(np.quantile(mmd_fit, 1.0 - alpha, method="higher"))
        prompt_calibrations[prompt_id] = {
            "raw": raw_calibration,
            "mmd_fit": mmd_fit,
            "mmd_mean": mmd_mean,
            "mmd_std": mmd_std,
            "mmd_evaluable": mmd_evaluable,
            "mmd_threshold": mmd_threshold,
        }
        prompt_null_audit.append(
            {
                "prompt_id": prompt_id,
                "raw_active_components": raw_calibration.get("active_components", []),
                "raw_excluded_degenerate_components": raw_calibration.get(
                    "excluded_degenerate_components", []
                ),
                "raw_composite_evaluable": bool(raw_calibration.get("evaluable")),
                "raw_composite_evaluation_fpr": raw_calibration.get("evaluation_false_positive_rate"),
                "h8_mmd_evaluable": mmd_evaluable,
                "h8_mmd_evaluation_fpr": float(
                    np.mean(
                        [
                            empirical_upper_p(mmd_fit, float(value)) <= alpha
                            for value in mmd_eval
                        ]
                    )
                ),
                "bandwidth": bandwidths[prompt_id],
            }
        )

    prompt_endpoint_results: list[dict[str, Any]] = []
    intact_pass_by_prompt: dict[str, float] = {}
    for prompt_id, prompt in prompt_map.items():
        intact_rows, _ = grouped("intact", prompt_id)
        reference_rows = intact_rows[group_size : 2 * group_size]
        intact_pass_by_prompt[prompt_id] = float(
            np.mean([evaluate_task(str(row["response"]), prompt) for row in reference_rows])
        )

    for endpoint in endpoints:
        endpoint_name = endpoint["endpoint_id"]
        for prompt_id, prompt in prompt_map.items():
            intact_rows, intact_features = grouped("intact", prompt_id)
            attack_rows, attack_features = grouped(endpoint_name, prompt_id)
            if len(attack_rows) != group_size:
                raise RuntimeError(f"{endpoint_name}/{prompt_id} has {len(attack_rows)} rows, expected {group_size}")
            reference_rows = intact_rows[group_size : 2 * group_size]
            reference_features = intact_features[group_size : 2 * group_size]
            observed_raw = raw_metrics(
                [str(row["response"]) for row in reference_rows],
                [str(row["response"]) for row in attack_rows],
                reference_features[:, 11:523],
                attack_features[:, 11:523],
            )
            calibration = prompt_calibrations[prompt_id]
            composite = observed_composite(calibration["raw"], observed_raw)
            scaler = scalers[prompt_id]
            x = scaler.transform(reference_features, schema)
            y = scaler.transform(attack_features, schema)
            observed_mmd = h8_mmd2_unbiased_unequal(x, y, float(bandwidths[prompt_id]["value"]))
            if calibration["mmd_evaluable"]:
                mmd_z = (observed_mmd - calibration["mmd_mean"]) / calibration["mmd_std"]
                mmd_p = empirical_upper_p(calibration["mmd_fit"], observed_mmd)
                mmd_detected = mmd_p <= alpha
            else:
                mmd_z = mmd_p = mmd_detected = None
            task_pass = float(np.mean([evaluate_task(str(row["response"]), prompt) for row in attack_rows]))
            prompt_endpoint_results.append(
                {
                    **endpoint,
                    "prompt_id": prompt_id,
                    "variant_role": prompt.get("variant_role"),
                    "task_family": prompt.get("task_family"),
                    "raw_metrics": observed_raw,
                    "raw_composite_z": composite["z"],
                    "raw_composite_p": composite["p_value"],
                    "raw_detected": composite["detected"],
                    "raw_component_z": composite["component_z"],
                    "h8_mmd2": observed_mmd,
                    "h8_mmd_z": mmd_z,
                    "h8_mmd_p": mmd_p,
                    "h8_mmd_detected": mmd_detected,
                    "intact_task_pass_rate": intact_pass_by_prompt[prompt_id],
                    "attack_task_pass_rate": task_pass,
                    "task_pass_drop": intact_pass_by_prompt[prompt_id] - task_pass,
                }
            )

    audits = {
        endpoint["endpoint_id"]: json.loads(
            (root / "audits" / f"{endpoint['endpoint_id']}.json").read_text(encoding="utf-8")
        )
        for endpoint in endpoints
    }
    intact_audit = json.loads((root / "audits" / "intact.json").read_text(encoding="utf-8"))
    strength_summaries: list[dict[str, Any]] = []
    for family in ["gaussian", "lora"]:
        labels = sorted({row["strength"] for row in endpoints if row["family"] == family}, key=LEVEL_ORDER.get)
        for label in labels:
            subset = [row for row in prompt_endpoint_results if row["family"] == family and row["strength"] == label]
            per_prompt: list[dict[str, Any]] = []
            for prompt_id in prompt_map:
                rows = [row for row in subset if row["prompt_id"] == prompt_id]
                raw_z = [float(row["raw_composite_z"]) for row in rows if row["raw_composite_z"] is not None]
                mmd_z = [float(row["h8_mmd_z"]) for row in rows if row["h8_mmd_z"] is not None]
                per_prompt.append(
                    {
                        "prompt_id": prompt_id,
                        "raw_z_seed_median": float(np.median(raw_z)) if raw_z else None,
                        "mmd_z_seed_median": float(np.median(mmd_z)) if mmd_z else None,
                        "raw_detected_seed_majority": float(np.mean([bool(row["raw_detected"]) for row in rows])) >= 0.5,
                        "mmd_detected_seed_majority": float(
                            np.mean([bool(row["h8_mmd_detected"]) for row in rows])
                        )
                        >= 0.5,
                        "task_pass_drop_seed_median": float(np.median([row["task_pass_drop"] for row in rows])),
                    }
                )
            raw_values = [row["raw_z_seed_median"] for row in per_prompt if row["raw_z_seed_median"] is not None]
            mmd_values = [row["mmd_z_seed_median"] for row in per_prompt if row["mmd_z_seed_median"] is not None]
            endpoint_audits = [
                audits[row["endpoint_id"]]
                for row in endpoints
                if row["family"] == family and row["strength"] == label
            ]
            if family == "gaussian":
                magnitude_key = "actual_relative_frobenius"
                auxiliary = {
                    "effective_changed_fraction_seed_median": float(
                        np.median([row["effective_changed_fraction"] for row in endpoint_audits])
                    ),
                    "bf16_retention_ratio_seed_median": float(
                        np.median([row["bf16_retention_ratio"] for row in endpoint_audits])
                    ),
                }
            else:
                magnitude_key = "relative_adapter_delta_frobenius"
                auxiliary = {
                    "heldout_token_nll_seed_median": float(
                        np.median([row["heldout_token_nll"] for row in endpoint_audits])
                    ),
                    "heldout_nll_change_vs_intact": float(
                        np.median([row["heldout_token_nll"] for row in endpoint_audits])
                        - float(intact_audit["heldout_token_nll"])
                    ),
                    "training_loss_seed_median": float(
                        np.median([row["training_loss"] for row in endpoint_audits])
                    ),
                }
            summary = {
                "family": family,
                "strength": label,
                "strength_order": LEVEL_ORDER[label],
                "endpoint_count": len(endpoint_audits),
                "actual_magnitude_seed_median": float(
                    np.median([row[magnitude_key] for row in endpoint_audits])
                ),
                "raw_composite_z_q25": float(np.quantile(raw_values, 0.25)) if raw_values else None,
                "raw_composite_z_median": float(np.median(raw_values)) if raw_values else None,
                "h8_mmd_z_q25": float(np.quantile(mmd_values, 0.25)) if mmd_values else None,
                "h8_mmd_z_median": float(np.median(mmd_values)) if mmd_values else None,
                "raw_detection_fraction": float(np.mean([row["raw_detected_seed_majority"] for row in per_prompt])),
                "h8_mmd_detection_fraction": float(
                    np.mean([row["mmd_detected_seed_majority"] for row in per_prompt])
                ),
                "task_pass_drop_median": float(np.median([row["task_pass_drop_seed_median"] for row in per_prompt])),
                "task_pass_drop_mean": float(np.mean([row["task_pass_drop_seed_median"] for row in per_prompt])),
                "per_prompt": per_prompt,
                **auxiliary,
            }
            gates = config["selection"]
            summary["eligible"] = bool(
                summary["raw_composite_z_q25"] is not None
                and summary["raw_composite_z_q25"] >= float(gates["raw_composite_q25_min"])
                and summary["h8_mmd_z_q25"] is not None
                and summary["h8_mmd_z_q25"] >= float(gates["h8_mmd_z_q25_min"])
                and summary["raw_detection_fraction"] >= float(gates["raw_detection_fraction_min"])
                and summary["task_pass_drop_mean"] <= float(gates["maximum_task_pass_drop"])
            )
            strength_summaries.append(summary)

    family_decisions: list[dict[str, Any]] = []
    for family in ["gaussian", "lora"]:
        rows = sorted([row for row in strength_summaries if row["family"] == family], key=lambda row: row["strength_order"])
        magnitudes = [row["actual_magnitude_seed_median"] for row in rows]
        raw_medians = [row["raw_composite_z_median"] for row in rows]
        mmd_medians = [row["h8_mmd_z_median"] for row in rows]
        eligible = [row for row in rows if row["eligible"]]
        family_decisions.append(
            {
                "family": family,
                "actual_magnitude_strictly_increasing": all(a < b for a, b in zip(magnitudes, magnitudes[1:])),
                "raw_behavior_nondecreasing": all(a <= b for a, b in zip(raw_medians, raw_medians[1:])),
                "h8_behavior_nondecreasing": all(a <= b for a, b in zip(mmd_medians, mmd_medians[1:])),
                "recommended_strength": eligible[0]["strength"] if eligible else None,
                "calibration_status": "CALIBRATED" if eligible else "NO_LEVEL_MET_PREDECLARED_GATES",
            }
        )

    report = {
        "schema_version": config["schema_version"],
        "status": "COMPLETE",
        "source_commit": config["source_commit"],
        "created_at_utc": utc_now(),
        "design": {
            "prompt_count": len(prompts),
            "intact_repetitions_per_prompt": config["generation"]["intact_repetitions_per_prompt"],
            "attack_repetitions_per_prompt": config["generation"]["attack_repetitions_per_prompt"],
            "attack_endpoint_count": len(endpoints),
            "total_response_count": len(all_records),
            "null_fit_splits": fit_count,
            "null_evaluation_splits": eval_count,
        },
        "feature_calibration": {
            "dimension": schema.dimension,
            "schema_sha256": schema.sha256,
            "global_excluded_dimension_count": int(sum(exclusion)),
            "global_exclusion_mask": list(exclusion),
            "global_degenerate_bandwidth": global_bandwidth,
            "scalers": {prompt_id: scaler.as_dict() for prompt_id, scaler in scalers.items()},
            "bandwidths": bandwidths,
        },
        "null_calibration_audit": prompt_null_audit,
        "intact_heldout_token_nll": intact_audit["heldout_token_nll"],
        "strength_summaries": strength_summaries,
        "family_decisions": family_decisions,
        "prompt_endpoint_results_file": "prompt_endpoint_results.jsonl",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
    }
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "prompt_endpoint_results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in prompt_endpoint_results:
            handle.write(canonical_json(row) + "\n")
    write_json(analysis_dir / "FINAL_REPORT.json", report)
    print(json.dumps({"status": report["status"], "report": str(analysis_dir / "FINAL_REPORT.json")}, ensure_ascii=False))
    return report


def preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import transformers
    import peft

    errors: list[str] = []
    for key, hash_key in [("prompts", "prompts_sha256"), ("lora_train", "lora_train_sha256")]:
        path = resolve_path(config["inputs"][key])
        actual = sha256_file(path)
        expected = str(config["inputs"][hash_key])
        if actual != expected:
            errors.append(f"{key} SHA256 mismatch: {actual} != {expected}")
    prompts = read_jsonl(resolve_path(config["inputs"]["prompts"]))
    if len(prompts) != 12 or len({row["id"] for row in prompts}) != 12:
        errors.append("Frozen prompt file must contain 12 unique IDs")
    endpoints = attack_endpoints(config)
    if len(endpoints) != 18:
        errors.append("Expected exactly 18 attack endpoints")
    model_revision = str(config["model"]["revision"])
    model_snapshot = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / model_revision
    )
    semantic_revision = str(config["semantic_encoder"]["revision"])
    semantic_snapshot = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "hub"
        / "models--BAAI--bge-small-zh-v1.5"
        / "snapshots"
        / semantic_revision
    )
    if not model_snapshot.is_dir():
        errors.append(f"Missing target snapshot: {model_snapshot}")
    if not semantic_snapshot.is_dir():
        errors.append(f"Missing semantic snapshot: {semantic_snapshot}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_commit": config["source_commit"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prompt_count": len(prompts),
        "attack_endpoint_count": len(endpoints),
        "model_snapshot": str(model_snapshot),
        "semantic_snapshot": str(semantic_snapshot),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "generation_response_count": 12
        * (
            int(config["generation"]["intact_repetitions_per_prompt"])
            + len(endpoints) * int(config["generation"]["attack_repetitions_per_prompt"])
        ),
        "created_at_utc": utc_now(),
    }
    write_json(output_root(config) / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("Preflight failed")
    return report


def sample(config: Mapping[str, Any], family: str) -> None:
    prompts = read_jsonl(resolve_path(config["inputs"]["prompts"]))
    if family in {"intact", "all"}:
        sample_intact(config, prompts)
    if family in {"gaussian", "all"}:
        sample_gaussian(config, prompts)
    if family in {"lora", "all"}:
        sample_lora(config, prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H9 Stage 1 attack-strength and null calibration")
    parser.add_argument("command", choices=["preflight", "train-lora", "sample", "analyze", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--family", choices=["intact", "gaussian", "lora", "all"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.command == "preflight":
        preflight(config_path, config)
    elif args.command == "train-lora":
        train_lora(config)
    elif args.command == "sample":
        sample(config, args.family)
    elif args.command == "analyze":
        analyze(config, config_path)
    else:
        preflight(config_path, config)
        train_lora(config)
        sample(config, "all")
        analyze(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
