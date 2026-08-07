from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from _bootstrap import ROOT
from llm_integrity.io import read_jsonl
from llm_integrity.modeling import render_prompt


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="experiments/prompt-union-14b-e1/inputs/final30_pairs.jsonl")
    parser.add_argument("--output", default="experiments/prompt-union-14b-e1/results/tokenizer_preflight.json")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / args.pairs)
    if len(rows) != 30:
        raise ValueError(f"expected 30 frozen pairs, received {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    records: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"prompt_id": row["prompt_id"], "category": row["category"], "component": row["component"]}
        for role in ("initial", "optimized"):
            text = str(row[f"{role}_prompt"])
            rendered = render_prompt(tokenizer, text, None)
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            item[role] = {
                "text_sha256": sha256_text(text), "rendered_sha256": sha256_text(rendered),
                "token_count": len(token_ids), "fits_task_512": len(token_ids) <= 512,
                "fits_sensitivity_128": len(token_ids) <= 128,
                "first_token_ids": token_ids[:8], "last_token_ids": token_ids[-8:],
            }
        item["technical_pass"] = all(item[role][limit] for role in ("initial", "optimized") for limit in ("fits_task_512", "fits_sensitivity_128"))
        records.append(item)

    report = {
        "schema_version": "experiment_e_tokenizer_preflight_1.0", "model_id": MODEL_ID,
        "revision": REVISION, "tokenizer_class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
        "chat_template_sha256": sha256_text(tokenizer.chat_template or ""), "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None), "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None), "records": records,
        "technical_passed": sum(bool(item["technical_pass"]) for item in records),
        "max_initial_tokens": max(item["initial"]["token_count"] for item in records),
        "max_optimized_tokens": max(item["optimized"]["token_count"] for item in records),
        "passed": all(bool(item["technical_pass"]) for item in records),
    }
    atomic_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
