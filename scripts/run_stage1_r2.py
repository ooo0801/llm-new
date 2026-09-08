"""Frozen, resumable R2 independent intact sampling and analysis. No attacks."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from llm_integrity.stage1_r1 import canonical, digest, permutation_test
from llm_integrity.stage1_r2 import atomic_json, freeze_json, read_chain, schedule, schedule_hash, summarize_validation
from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_precalibration import (
    FamilyBalancedScaler, build_h8_feature_schema, fit_family_balanced_scaler,
    global_continuous_exclusion_mask, h8_global_degenerate_bandwidth, h8_median_positive_pairwise_distance,
)


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def json_rows(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(("stage1-r2-independent/" + key).encode()).digest()[:8], "little")


def torch_setup(strict=True):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    return torch


def snapshot_files(repo, revision, expected_weight):
    from huggingface_hub import snapshot_download
    folder = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    files = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model", ".jinja"):
            files[str(p.relative_to(folder))] = digest(p)
    if files.get("model.safetensors") != expected_weight:
        raise RuntimeError(f"Frozen model weight mismatch: {repo}")
    return {"repo": repo, "revision": revision, "snapshot": str(folder), "files_sha256": files}


def preflight(config_path, config, prompts, out, historical_root):
    torch = torch_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("R2 generation requires existing CUDA server")
    if digest(ROOT / config["prompts"]) != config["prompts_sha256"]:
        raise RuntimeError("Prompt manifest mismatch")
    plans = {r: schedule(config, prompts, r) for r in ("fit", "calibration", "validation")}
    seeds = {row["generation_seed"] for rows in plans.values() for row in rows}
    if len(seeds) != sum(map(len, plans.values())) or len(seeds) != config["sampling"]["response_budget"]:
        raise RuntimeError("Sampling seed collision or budget mismatch")
    old = []
    for name in ("h9_stage1_qwen05b_attack_stat_calibration_20260904", "h9_stage1_qwen05b_gaussian_escalation_20260904"):
        parent = historical_root / "results" / name / "responses"
        files = sorted(parent.glob("*.jsonl"))
        if len(files) != 19:
            raise RuntimeError("Cannot verify freshness against prior Stage1 banks")
        for path in files:
            rows = json_rows(path)
            if seeds & {r["generation_seed"] for r in rows}:
                raise RuntimeError("Fresh R2 generation seeds overlap old Stage1")
            old.append({"file": str(path), "sha256": digest(path), "records": len(rows)})
    model = snapshot_files(config["model"]["name"], config["model"]["revision"], config["target_weight_sha256"])
    sem = snapshot_files(config["semantic"]["name"], config["semantic"]["revision"], config["semantic"]["weight_sha256"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
    if not tok.chat_template:
        raise RuntimeError("Frozen target chat template absent")
    code = [Path(__file__), ROOT / "src/llm_integrity/stage1_r1.py", ROOT / "src/llm_integrity/stage1_r2.py",
            ROOT / "src/llm_integrity/features.py", ROOT / "src/llm_integrity/modeling.py", ROOT / "src/llm_integrity/h8_precalibration.py"]
    manifest = {"config": config, "config_sha256": digest(config_path),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code},
                "target_model": model, "semantic_model": sem,
                "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                "plans": {r: {"count": len(rows), "sha256": schedule_hash(rows)} for r, rows in plans.items()},
                "old_stage1_banks": old, "old_seed_overlap": 0,
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scipy", "sentence-transformers")},
                "gpu": torch.cuda.get_device_name(), "torch_cuda_version": torch.version.cuda,
                "semantic_deterministic_algorithms": "strict", "generation_deterministic_algorithms": "warn_only",
                "generation_determinism_limitation": "CUDA top-p cumsum lacks guaranteed deterministic implementation; fixed seeds do not promise bitwise resampling.",
                "tf32": False, "generation_batch_size": 1}
    freeze_json(out / "PRE_DATA_MANIFEST.json", manifest)
    return plans, manifest


def status(out, **fields):
    value = {"updated_utc": utc(), **fields}
    atomic_json(out / "STATUS.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def generate_role(config, prompts, plans, out, role):
    from llm_integrity.modeling import load_model, render_prompt
    torch = torch_setup(strict=False)
    plan = plans[role]
    path = out / "responses" / f"{role}.jsonl"
    manifest_hash = digest(out / "PRE_DATA_MANIFEST.json")
    done = read_chain(path, plan, manifest_hash)
    if len(done) == len(plan):
        status(out, phase="generate_" + role, state="COMPLETE", completed=len(done), total=len(plan))
        return
    if role == "validation":
        if not (out / "FIT_FROZEN.json").exists() or not (out / "CALIBRATION_AUDIT.json").exists():
            raise RuntimeError("Fit and calibration technical gates must precede validation generation")
        if read_json(out / "CALIBRATION_AUDIT.json")["technical_status"] != "PASS":
            raise RuntimeError("Calibration technical gate did not pass")
    bundle = load_model(config["model"])
    generation = config["generation"]
    encodings = {}
    for p in prompts:
        rendered = render_prompt(bundle.tokenizer, p["prompt"], generation["system_prompt"])
        tokens = bundle.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=generation["max_input_tokens"])
        encodings[p["id"]] = {k: v.to(bundle.device) for k, v in tokens.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_count = len(done)
    previous = done[-1]["record_sha256"] if done else "0" * 64
    start = time.monotonic()
    status(out, phase="generate_" + role, state="RUNNING", completed=len(done), total=len(plan))
    try:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            for index in range(initial_count, len(plan)):
                spec = plan[index]
                seed = spec["generation_seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                encoded = encodings[spec["prompt_id"]]
                with torch.inference_mode():
                    generated = bundle.model.generate(**encoded, max_new_tokens=generation["max_new_tokens"],
                        do_sample=True, temperature=generation["temperature"], top_p=generation["top_p"], top_k=generation["top_k"],
                        pad_token_id=bundle.tokenizer.pad_token_id, eos_token_id=bundle.tokenizer.eos_token_id)
                ids = generated[0, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
                text = bundle.tokenizer.decode(ids, skip_special_tokens=True)
                row = {**spec, "created_utc": utc(), "manifest_sha256": manifest_hash,
                       "previous_sha256": previous, "response": text,
                       "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "completion_token_ids": ids, "completion_token_count": len(ids),
                       "last_token_is_tokenizer_eos": bool(ids and ids[-1] == bundle.tokenizer.eos_token_id)}
                row["record_sha256"] = hashlib.sha256(canonical(row).encode()).hexdigest()
                f.write(canonical(row) + "\n")
                f.flush()
                previous = row["record_sha256"]
                if (index + 1) % 24 == 0 or index + 1 == len(plan):
                    os.fsync(f.fileno())
                    elapsed = time.monotonic() - start
                    rate = (index + 1 - initial_count) / max(elapsed, 1e-9)
                    status(out, phase="generate_" + role, state="RUNNING", completed=index + 1, total=len(plan),
                           responses_per_second=rate, estimated_remaining_seconds=(len(plan) - index - 1) / max(rate, 1e-9))
    finally:
        bundle.close()
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    read_chain(path, plan, manifest_hash, complete=True)


class NewSemanticCache:
    """Singleton FP32 encoding: cache assignment never depends on a text's batch peers."""
    def __init__(self, config, out, manifest):
        self.config = config["semantic"]
        self.folder = out / "semantic_cache"
        self.folder.mkdir(exist_ok=True)
        self.manifest = {"identity": {**self.config, "model_files": manifest["semantic_model"]["files_sha256"],
                                      "torch": manifest["packages"]["torch"],
                                      "sentence_transformers": manifest["packages"]["sentence-transformers"],
                                      "device": "cuda", "tf32": False, "deterministic_algorithms": True}}
        freeze_json(self.folder / "IDENTITY.json", self.manifest)
        self.identity_hash = digest(self.folder / "IDENTITY.json")
        self.encoder = None

    def transform(self, texts):
        result = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            path = self.folder / (text_hash + ".json")
            if path.exists():
                data = read_json(path)
                if data["text"] != text or data["identity_sha256"] != self.identity_hash:
                    raise RuntimeError("Semantic text/cache identity mismatch")
                payload = {k: v for k, v in data.items() if k != "content_sha256"}
                if hashlib.sha256(canonical(payload).encode()).hexdigest() != data["content_sha256"]:
                    raise RuntimeError("Semantic cache corruption")
                vector = np.asarray(data["vector"], dtype=np.float64)
            else:
                torch = torch_setup()
                if self.encoder is None:
                    from sentence_transformers import SentenceTransformer
                    self.encoder = SentenceTransformer(self.config["name"], revision=self.config["revision"],
                        device="cuda", local_files_only=True,
                        model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.float32})
                    self.encoder.max_seq_length = self.config["max_seq_length"]
                    self.encoder.eval()
                with torch.inference_mode():
                    vector = np.asarray(self.encoder.encode([text], batch_size=1, normalize_embeddings=True,
                                         precision="float32", show_progress_bar=False)[0], dtype=np.float64)
                data = {"text": text, "identity_sha256": self.identity_hash, "vector": vector.tolist()}
                data["content_sha256"] = hashlib.sha256(canonical(data).encode()).hexdigest()
                atomic_json(path, data)
            if vector.shape != (512,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 2e-6:
                raise RuntimeError("Invalid semantic vector")
            result.append(vector)
        return np.stack(result)

    def close(self):
        self.encoder = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def complete_rows(out, role, plans):
    return read_chain(out / "responses" / (role + ".jsonl"), plans[role], digest(out / "PRE_DATA_MANIFEST.json"), True)


def fit(config, prompts, plans, out, manifest):
    rows = complete_rows(out, "fit", plans)
    fit_hash = digest(out / "responses/fit.jsonl")
    if (out / "FIT_FROZEN.json").exists():
        existing = read_json(out / "FIT_FROZEN.json")
        if existing["fit_responses_sha256"] != fit_hash or existing["pre_data_manifest_sha256"] != digest(out / "PRE_DATA_MANIFEST.json"):
            raise RuntimeError("Fit freeze provenance mismatch")
        return
    status(out, phase="fit_features", state="RUNNING", records=len(rows))
    cache = NewSemanticCache(config, out, manifest)
    extractor = FeatureExtractor(semantic_cache=cache)
    schema = build_h8_feature_schema(512)
    matrices = {}
    for p in prompts:
        group = sorted([r for r in rows if r["prompt_id"] == p["id"]], key=lambda r: r["response_index"])
        matrices[p["id"]] = extractor.transform([r["response"] for r in group], [{**p, "category": p["task_family"]}] * len(group))
    pooled = np.concatenate(list(matrices.values()))
    exclusion = global_continuous_exclusion_mask(pooled, schema)
    scalers = {pid: fit_family_balanced_scaler(pid, x, pooled, schema, exclusion_mask=exclusion) for pid, x in matrices.items()}
    transformed = {pid: scalers[pid].transform(x, schema) for pid, x in matrices.items()}
    global_sigma = h8_global_degenerate_bandwidth(transformed)
    bandwidths = {}
    for pid, x in transformed.items():
        try:
            bandwidths[pid] = {"value": h8_median_positive_pairwise_distance(x), "source": "fit_prompt_only"}
        except ValueError:
            bandwidths[pid] = {"value": global_sigma["sigma"], "source": "fit_global_degenerate_fallback"}
    probes = list(dict.fromkeys(r["response"] for r in rows))[:40]
    a = cache.transform(probes)
    b = cache.transform(list(reversed(probes)))[::-1]
    reloaded = NewSemanticCache(config, out, manifest).transform(probes)
    if not np.array_equal(a, b) or not np.array_equal(a, reloaded):
        raise RuntimeError("Semantic cache order/reload invariance failure")
    payload = {"pre_data_manifest_sha256": digest(out / "PRE_DATA_MANIFEST.json"), "fit_responses_sha256": fit_hash,
               "schema_sha256": schema.sha256, "scalers": {pid: s.as_dict() for pid, s in scalers.items()},
               "bandwidths": bandwidths, "global_bandwidth": global_sigma,
               "data_role": "fresh_fit_only", "calibration_or_validation_used": False,
               "semantic_identity_sha256": cache.identity_hash, "cache_invariance": "PASS"}
    freeze_json(out / "FIT_FROZEN.json", payload)
    cache.close()
    status(out, phase="fit_features", state="COMPLETE", fit_sha256=digest(out / "FIT_FROZEN.json"))


def analyze_role(config, prompts, plans, out, manifest, role):
    rows = complete_rows(out, role, plans)
    frozen = read_json(out / "FIT_FROZEN.json")
    if frozen["fit_responses_sha256"] != digest(out / "responses/fit.jsonl"):
        raise RuntimeError("Fit bank changed")
    schema = build_h8_feature_schema(512)
    if frozen["schema_sha256"] != schema.sha256:
        raise RuntimeError("Schema changed")
    cache = NewSemanticCache(config, out, manifest)
    if cache.identity_hash != frozen["semantic_identity_sha256"]:
        raise RuntimeError("Semantic identity differs from Fit")
    extractor = FeatureExtractor(semantic_cache=cache)
    scalers = {pid: FamilyBalancedScaler.from_dict(s) for pid, s in frozen["scalers"].items()}
    fit_hash = digest(out / "FIT_FROZEN.json")
    data_hash = digest(out / "responses" / (role + ".jsonl"))
    units = config["sampling"]["validation_units_per_prompt"] if role == "validation" else 1
    results = []
    grouped = {}
    for r in rows:
        grouped.setdefault((r["prompt_id"], r["unit"]), []).append(r)
    for unit in range(units):
        for p in prompts:
            key = (p["id"], unit)
            group = sorted(grouped[key], key=lambda r: r["response_index"])
            path = out / "decisions" / role / (p["id"] + "_" + str(unit) + ".json")
            if path.exists():
                result = read_json(path)
                payload = {k: v for k, v in result.items() if k != "content_sha256"}
                if result["prompt_id"] != p["id"] or result["unit"] != unit or result["role"] != role or result["fit_sha256"] != fit_hash or result["data_sha256"] != data_hash or result["content_sha256"] != hashlib.sha256(canonical(payload).encode()).hexdigest():
                    raise RuntimeError("Decision resume provenance/content mismatch")
            else:
                texts = [r["response"] for r in group]
                feat = extractor.transform(texts, [{**p, "category": p["task_family"]}] * 48)
                x = scalers[p["id"]].transform(feat, schema)
                settings = config["test"]
                tested = permutation_test(texts, feat[:, 11:523], x, 24, frozen["bandwidths"][p["id"]]["value"],
                    stable_seed(f"{role}/{p['id']}/{unit}"), settings["permutations"], settings["alpha"], settings["tie_atol"])
                result = {"prompt_id": p["id"], "unit": unit, "role": role, "fit_sha256": fit_hash,
                          "data_sha256": data_hash, "reference_seeds": [r["generation_seed"] for r in group[:24]],
                          "target_seeds": [r["generation_seed"] for r in group[24:]], **tested}
                result["content_sha256"] = hashlib.sha256(canonical(result).encode()).hexdigest()
                freeze_json(path, result)
            results.append(result)
        status(out, phase="analyze_" + role, state="RUNNING", completed=len(results), total=units * len(prompts))
    cache.close()
    if role == "calibration":
        technical = all(r[ch]["detected"] is not None for r in results for ch in config["test"]["channels"])
        audit = {"technical_status": "PASS" if technical else "FAIL", "role": "smoke_only_no_threshold_tuning",
                 "decisions": results, "frozen_fit_sha256": fit_hash, "validation_generated_before_audit": False}
        freeze_json(out / "CALIBRATION_AUDIT.json", audit)
        if not technical:
            raise RuntimeError("Calibration technical failure; validation not started")
    else:
        final = summarize_validation(results, prompts, config)
        final.update({"completed_utc": utc(), "pre_data_manifest_sha256": digest(out / "PRE_DATA_MANIFEST.json"),
                      "fit_sha256": fit_hash, "calibration_audit_sha256": digest(out / "CALIBRATION_AUDIT.json"),
                      "response_count": sum(len(complete_rows(out, r, plans)) for r in plans),
                      "decisions": results, "model": config["model"], "old_responses_reused": 0,
                      "server_shutdown_performed": False})
        atomic_json(out / "FINAL_REPORT.json", final)
        status(out, phase="complete", state="COMPLETE", result=final["status"], response_count=final["response_count"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_r2_independent_null.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            config = read_json(args.config)
            prompts = json_rows(ROOT / config["prompts"])
            status(out, phase="preflight", state="RUNNING")
            plans, manifest = preflight(args.config, config, prompts, out, args.historical_root.resolve())
            status(out, phase="preflight", state="PASS", planned_responses=35712)
            if args.preflight_only:
                return
            generate_role(config, prompts, plans, out, "fit")
            fit(config, prompts, plans, out, manifest)
            generate_role(config, prompts, plans, out, "calibration")
            analyze_role(config, prompts, plans, out, manifest, "calibration")
            generate_role(config, prompts, plans, out, "validation")
            analyze_role(config, prompts, plans, out, manifest, "validation")
        except Exception as exc:
            status(out, phase="error", state="FAILED", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
