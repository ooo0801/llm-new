from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/experiment_c_qwen14b_20260806")
    parser.add_argument("--output-dir", default="results/experiment_c_qwen14b_20260806/analysis")
    args = parser.parse_args()
    root = resolve(args.root)
    output = resolve(args.output_dir)

    design = read_json(ROOT / "experiments/prompt-construction-14b-c1/inputs/design_manifest.json")
    generation = read_json(root / "02_candidate_generation/joint_summary_60.json")
    portfolio = read_json(root / "03_development/portfolio_build_report.json")
    development = read_json(root / "03_development/development_selection_report.json")
    final_report_path = root / "04_heldout_test/final_test_report.json"
    final_rows_path = root / "04_heldout_test/final_test_validated_prompts.jsonl"
    heldout = read_json(final_report_path) if final_report_path.exists() else None
    final_rows = read_jsonl(final_rows_path) if final_rows_path.exists() else []
    accepted = [row for row in final_rows if row.get("optimization", {}).get("accepted") is True]

    technical = bool(generation.get("technical_passed") and development.get("gate_passed") and heldout is not None and heldout.get("technical_passed"))
    observed = len(accepted)
    status = "supported" if technical and observed >= 4 else "refuted" if technical else "technical_no_go"
    report = {
        "schema_version": "experiment_c_construction_report_1.0",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "candidate_pool_rows": design["candidate_pool"]["rows"],
        "construction_prompt_rows": design["prompt_sample"]["rows"],
        "inner_technically_complete": generation.get("technically_complete"),
        "inner_proxy_accepted": generation.get("accepted"),
        "portfolio_candidates": portfolio.get("portfolio_candidates", portfolio.get("output_candidates")),
        "development_accepted": development.get("development_accepted"),
        "development_gate_passed": development.get("gate_passed"),
        "heldout_rows": heldout.get("rows") if heldout else 0,
        "heldout_accepted": observed,
        "heldout_technical_passed": heldout.get("technical_passed") if heldout else False,
        "h_c1": {"required": 4, "observed": observed, "status": status},
        "accepted_prompt_ids": [str(row["prompt_id"]) for row in accepted],
        "accepted_categories": {category: sum(str(row.get("category")) == category for row in accepted) for category in sorted({str(row.get("category")) for row in accepted})},
        "selection_leakage_guard": "development calibration and selection precede held-out test; held-out outcomes do not alter candidates or thresholds",
    }
    write_json(output / "FINAL_REPORT.json", report)
    write_jsonl(output / "accepted14b_complement.jsonl", accepted)
    lines = [
        "# Experiment C Final Report", "",
        f"- Construction technical completion: {generation.get('technically_complete')}/60",
        f"- Inner proxy accepted: {generation.get('accepted')}/60",
        f"- Development accepted: {development.get('development_accepted')}",
        f"- Held-out accepted: {observed}",
        f"- H-C1 (at least 4 held-out): {status}", "",
        "Held-out results were not used to edit, rerank, or repair the frozen development-selected candidates.", "",
        "## Accepted 14B-specific complement", "",
    ]
    lines.extend(f"- `{row['prompt_id']}` ({row.get('category', 'unknown')})" for row in accepted)
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
