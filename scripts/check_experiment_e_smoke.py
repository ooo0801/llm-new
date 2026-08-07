from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FAMILIES = {"unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--micro", required=True, type=Path)
    parser.add_argument("--macro", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    task, micro, macro = rows(args.task), rows(args.micro), rows(args.macro)
    families = {str(row["family"]) for row in macro}
    finite_macro = all(math.isfinite(float(row["macro_l2_raw"])) and float(row["macro_l2_raw"]) > 0 for row in macro)
    report = {
        "schema_version": "experiment_e_smoke_1.0", "task_records": len(task), "micro_records": len(micro),
        "macro_records": len(macro), "macro_families": sorted(families),
        "complete_micro_coverage": all(row.get("complete_parameter_coverage") is True for row in micro),
        "finite_positive_macro": finite_macro,
    }
    report["passed"] = bool(len(task) == 2 and len(micro) == 2 and len(macro) == 20 and families == FAMILIES and report["complete_micro_coverage"] and finite_macro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
