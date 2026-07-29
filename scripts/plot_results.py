from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="results/figures/verification_summary.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for path in Path(project_path(args.results)).rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "test" in value and "attack" in value:
            rows.append(
                {
                    "attack": value["attack"],
                    "method": value.get("selection_method", "unknown"),
                    "statistic": value["test"]["statistic"],
                    "p_value": value["test"]["p_value"],
                    "modified": value["modified"],
                }
            )
    if not rows:
        raise ValueError("No verification result JSON files found")
    frame = pd.DataFrame(rows)
    target = project_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(frame, x="attack", y="statistic", hue="method", ax=axes[0])
    sns.barplot(frame, x="attack", y="p_value", hue="method", ax=axes[1])
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1)
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    frame.to_csv(target.with_suffix(".csv"), index=False)
    print(json.dumps({"figure": str(target), "rows": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
