"""Readable tables and baseline diagnostics for the completed R1 run."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_integrity.stage1_r1 import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.output / "prompt_endpoint_results.json").read_text(encoding="utf-8"))
    comparison = {r["unit_id"]: r for r in json.loads((args.output / "legacy_comparison.json").read_text(encoding="utf-8"))}
    fields = ["unit_id", "prompt_id", "family", "strength", "attack_seed", "raw_status", "raw_p", "h8_status", "h8_p",
              "legacy_raw", "legacy_h8", "strict_intact_rate", "strict_attack_rate", "strict_drop",
              "intact_proxy_rate", "attack_proxy_rate", "task_reason"]
    with (args.output / "prompt_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            old = comparison[r["unit_id"]]
            writer.writerow({**{key: r[key] for key in fields[:5]},
                "raw_status": r["raw"]["status"], "raw_p": r["raw"]["p_value"],
                "h8_status": r["h8"]["status"], "h8_p": r["h8"]["p_value"],
                "legacy_raw": old["legacy_raw"], "legacy_h8": old["legacy_h8"],
                "strict_intact_rate": r["intact_task"]["pass_rate"], "strict_attack_rate": r["attack_task"]["pass_rate"],
                "strict_drop": r["task_pass_drop"], "intact_proxy_rate": r["intact_task"]["proxy_rate"],
                "attack_proxy_rate": r["attack_task"]["proxy_rate"], "task_reason": r["attack_task"]["reason"]})
    p = args.input / "results/h9_stage1_qwen05b_attack_stat_calibration_20260904/responses/intact.jsonl"
    intact = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
    prompt_order = list(dict.fromkeys(r["prompt_id"] for r in rows))
    diagnostics = []
    for pid in prompt_order:
        bank = [r for r in intact if r["prompt_id"] == pid]
        task = next(r["intact_task"] for r in rows if r["prompt_id"] == pid)
        diagnostics.append({"prompt_id": pid, "task": task, "at_64_token_cap": sum(r["completion_token_count"] == 64 for r in bank),
                            "top_responses": Counter(r["response"] for r in bank).most_common(3)})
    save_json(args.output / "BASELINE_DIAGNOSTICS.json", {
        "interpretation": "64 tokens indicates budget reached, not independently verified stop reason; proxies are not task correctness",
        "rows": diagnostics,
    })
    print(json.dumps([{k: v for k, v in r.items() if k != "top_responses"} for r in diagnostics], ensure_ascii=True))


if __name__ == "__main__":
    main()
