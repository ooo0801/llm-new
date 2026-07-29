from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--output", default="results/figures/fingerprint_diagnostics.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for path in Path(project_path(args.fingerprints)).glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = value.get("metadata", {}).get("diagnostics", {})
        rows.append(
            {
                "name": path.stem,
                "method": value.get("selection_method", path.stem),
                "k": diagnostics.get("k", len(value.get("entries", []))),
                "coverage": diagnostics.get("coverage_rate"),
                "complementarity": diagnostics.get("complementarity"),
                "sensitivity": diagnostics.get("mean_sensitivity"),
            }
        )
    frame = pd.DataFrame(rows)
    target = project_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column in zip(axes, ["coverage", "complementarity", "sensitivity"]):
        sns.lineplot(frame, x="k", y=column, hue="method", marker="o", ax=axis)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    frame.to_csv(target.with_suffix(".csv"), index=False)
    print(json.dumps({"figure": str(target), "rows": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
