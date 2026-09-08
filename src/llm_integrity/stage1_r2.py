"""Pure R2 schedules, immutable response chains, and fixed-sample acceptance."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta, binom
from .stage1_r1 import canonical, digest, save_json


def schedule(config, prompts, role):
    s = config["sampling"]
    offset = {"fit": 0, "calibration": 100000, "validation": 200000}[role]
    units = s["validation_units_per_prompt"] if role == "validation" else 1
    n = s["reference_per_unit"] + s["target_per_unit"] if role == "validation" else s[role + "_per_prompt"]
    if n != 48 or len(prompts) != s["prompt_count"] or units * n >= 100000:
        raise ValueError("Unexpected frozen R2 schedule dimensions")
    rows = []
    for unit in range(units):
        for pi, prompt in enumerate(prompts):
            # Interleave reference and target to reduce time-order imbalance.
            for i in range(24):
                for side in (0, 1):
                    index = side * 24 + i
                    rows.append({"role": role, "unit": unit, "prompt_id": prompt["id"], "prompt_index": pi,
                                 "response_index": index, "side": "reference" if side == 0 else "target",
                                 "generation_seed": s["seed_base"] + pi * 1000000 + offset + unit * n + index})
    return rows


def schedule_hash(rows):
    return hashlib.sha256(canonical(rows).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_json(tmp, value)
    tmp.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen artifact mismatch: {path.name}")
    else:
        atomic_json(path, value)


def read_chain(path, expected, manifest_hash, complete=False):
    path = Path(path)
    if not path.exists():
        if complete:
            raise ValueError("Required response bank is absent")
        return []
    rows = []
    previous = "0" * 64
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.endswith("\n"):
                raise ValueError("Partial response line: preserve file and audit before resume")
            row = json.loads(line)
            if index >= len(expected) or any(row.get(k) != v for k, v in expected[index].items()):
                raise ValueError("Response schedule mismatch")
            if row["manifest_sha256"] != manifest_hash or row["previous_sha256"] != previous:
                raise ValueError("Response provenance chain mismatch")
            if row["response_sha256"] != hashlib.sha256(row["response"].encode()).hexdigest():
                raise ValueError("Response text hash mismatch")
            payload = {k: v for k, v in row.items() if k != "record_sha256"}
            if hashlib.sha256(canonical(payload).encode()).hexdigest() != row["record_sha256"]:
                raise ValueError("Response record hash mismatch")
            previous = row["record_sha256"]
            rows.append(row)
    if complete and len(rows) != len(expected):
        raise ValueError("Incomplete response bank")
    return rows


def confidence_interval(k, n, error=.05):
    return [0.0 if k == 0 else float(beta.ppf(error / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - error / 2, k + 1, n - k))]


def summarize_validation(rows, prompts, config):
    """Fixed n; heterogeneous prompt probabilities are NOT pooled as binomial."""
    n_per = config["sampling"]["validation_units_per_prompt"]
    channels = config["test"]["channels"]
    a = config["acceptance"]
    expected = {(p["id"], i) for p in prompts for i in range(n_per)}
    keys = [(r["prompt_id"], r["unit"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Validation must have all unique frozen units before acceptance")
    per_prompt = []
    macro = {}
    unknown = 0
    local_flag = False
    for ch in channels:
        total_rejections = 0
        for prompt in prompts:
            group = [r for r in rows if r["prompt_id"] == prompt["id"]]
            decisions = [r[ch]["detected"] for r in group]
            missing = sum(v is None for v in decisions)
            unknown += missing
            k = sum(v is True for v in decisions)
            total_rejections += k
            p_inflation = float(binom.sf(k - 1, n_per, a["localized_null_rate"])) if not missing else None
            flag = p_inflation is not None and p_inflation <= a["localized_inflation_alpha_familywise"] / (len(prompts) * len(channels))
            local_flag |= flag
            per_prompt.append({"channel": ch, "prompt_id": prompt["id"], "rejections": k, "units": n_per,
                               "unevaluable": missing, "fpr": k / n_per if not missing else None,
                               "pointwise_CP95_interval": confidence_interval(k, n_per) if not missing else None,
                               "inflation_test_p": p_inflation, "bonferroni_local_inflation_flag": flag})
        n = len(rows)
        observed = total_rejections / n
        margin = math.sqrt(math.log(len(channels) / a["confidence_error_across_two_macro_bounds"]) / (2 * n))
        # With missing decisions, no reported FPR/bound. Never silently count missing as negatives.
        incomplete = any(r["channel"] == ch and r["unevaluable"] for r in per_prompt)
        upper = min(1.0, observed + margin) if not incomplete else None
        macro[ch] = {"rejections": total_rejections, "units": n, "fpr": None if incomplete else observed,
                     "simultaneous_Hoeffding_upper": upper, "hoeffding_margin": margin,
                     "engineering_gate_pass": upper is not None and upper <= a["macro_fpr_upper_max"]}
    if unknown:
        status = "TECHNICAL_FAILURE_UNEVALUABLE_UNITS"
    elif local_flag:
        status = "FAIL_LOCALIZED_INFLATION_FLAG"
    elif all(m["engineering_gate_pass"] for m in macro.values()):
        status = "PASS_MACRO_ENGINEERING_CHECK_ONLY"
    else:
        status = "INCONCLUSIVE_MACRO_UPPER_BOUND_EXCEEDS_10_PERCENT"
    return {"status": status, "per_prompt": per_prompt, "macro": macro,
            "claim_scope": a["scope"], "formal_every_prompt_FPR_le_5_percent_proven": False,
            "stage2_release": "NOT_AUTHORIZED_R3_REMAINS", "unevaluable_channel_units": unknown,
            "interpretation": "Independent units conditional on frozen Fit and fixed generation; pointwise intervals not simultaneous. No optional extension."}
