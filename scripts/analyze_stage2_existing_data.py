"""Audit frozen Stage2 pilot evidence and emit explicitly derived tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROXIES = (
    "raw_logit_l2",
    "centered_logit_l2",
    "probability_l2",
    "js",
    "topk_change",
    "top1_flip",
)
BEHAVIOR_FAMILIES = ("gaussian_noise", "finetuning")
COVERAGE_FAMILIES = ("unstructured_pruning", "structured_pruning", "quantization")
STRENGTHS = ("weak", "medium", "strong")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bool_value(value: Any) -> bool | None:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    return None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            result[k] = rank
        i = j
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y) or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    a, b = rankdata(x), rankdata(y)
    am, bm = mean(a), mean(b)
    numerator = sum((u - am) * (v - bm) for u, v in zip(a, b))
    denominator = math.sqrt(sum((u - am) ** 2 for u in a) * sum((v - bm) ** 2 for v in b))
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify(snapshot: Path, evaluation: Path, summary: dict, plan: dict, selection: dict) -> dict:
    missing, mismatched = [], []
    for relative, expected in summary["evidence_hashes"].items():
        path = evaluation / relative
        if not path.exists():
            missing.append(relative)
        elif sha256(path) != expected:
            mismatched.append(relative)
    plan_hash = sha256(evaluation / "PLAN.json")
    search_missing, search_mismatched = [], []
    for relative, expected in plan["search_files"].items():
        path = snapshot / relative
        if not path.exists():
            search_missing.append(relative)
        elif sha256(path) != expected:
            search_mismatched.append(relative)
    response_lines = sum(
        sum(1 for _ in path.open("r", encoding="utf-8"))
        for path in (evaluation / "responses").glob("*.jsonl")
    )
    result = {
        "status": "PASS" if not (missing or mismatched or search_missing or search_mismatched) else "FAIL",
        "plan_sha256": plan_hash,
        "plan_binding_matches": plan_hash == summary["plan_sha256"] == selection["plan_sha256"],
        "evidence_hashes_checked": len(summary["evidence_hashes"]),
        "evidence_missing": missing,
        "evidence_mismatched": mismatched,
        "search_hashes_checked": len(plan["search_files"]),
        "search_missing": search_missing,
        "search_mismatched": search_mismatched,
        "response_rows_counted": response_lines,
        "response_rows_declared": summary["response_rows"],
    }
    if result["status"] != "PASS" or not result["plan_binding_matches"] or response_lines != summary["response_rows"]:
        raise ValueError(f"Frozen evidence verification failed: {result}")
    return result


def group_status(utility_seeds: int, good: int, required: int = 2) -> str:
    if utility_seeds < required:
        return "not_evaluable_attack_utility"
    if good == 0:
        return "behavior_insensitive"
    if good < required:
        return "cross_seed_unstable"
    return "pass"


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot, output = args.snapshot.resolve(), args.output.resolve()
    evaluation = snapshot / "evaluation"
    output.mkdir(parents=True, exist_ok=True)

    design = read_json(snapshot / "DESIGN.json")
    plan = read_json(evaluation / "PLAN.json")
    summary = read_json(evaluation / "SUMMARY.json")
    selection = read_json(evaluation / "CONFIRMATION_SELECTION.json")
    verification = verify(snapshot, evaluation, summary, plan, selection)

    prompts = {row["id"]: row for row in plan["candidates"]["prompts"]}
    aliases = plan["candidates"]["aliases"]
    candidates = [row for row in aliases if row["method"] != "source"]
    source_ids = {row["source_id"]: row["prompt_id"] for row in aliases if row["method"] == "source"}
    variants = {row["variant_id"]: row for row in design["variants"]}

    with (evaluation / "response_matrix.csv").open("r", encoding="utf-8-sig", newline="") as f:
        response_rows = list(csv.DictReader(f))
    response_by_key = {(r["prompt_id"], r["endpoint"]): r for r in response_rows}
    proxy_records = []
    for path in sorted((evaluation / "proxies").glob("*.json")):
        proxy_records.extend(read_json(path)["records"])
    proxy_by_key = {(r["prompt_id"], r["endpoint"]): r for r in proxy_records}

    utility_rows = []
    for split in ("train", "confirmation", "heldout"):
        for endpoint, value in summary["utility"][split].items():
            variant = variants[endpoint]
            utility_rows.append(
                {
                    "split": split,
                    "endpoint": endpoint,
                    "family": variant["family"],
                    "strength": variant["strength"],
                    "attack_seed": variant["seed"],
                    "qualified": value["qualified"],
                    "status": value["status"],
                    "intact_correct": value.get("intact_correct"),
                    "attack_correct": value.get("attack_correct"),
                    "drop_on_intact_correct_subset": value.get("drop"),
                    "configuration": json_cell(variant["configuration"]),
                }
            )
    write_csv(output / "attack_utility_detail.csv", utility_rows)

    endpoint_rows = []
    for alias in candidates:
        prompt = prompts[alias["prompt_id"]]
        for row in response_rows:
            if row["prompt_id"] != alias["prompt_id"]:
                continue
            proxy = proxy_by_key[(row["prompt_id"], row["endpoint"])]
            endpoint_rows.append(
                {
                    "method": alias["method"],
                    "source_id": alias["source_id"],
                    "prompt_id": alias["prompt_id"],
                    "category": prompt["category"],
                    "prompt": prompt["prompt"],
                    "inner_proxy_gain": alias["inner_proxy_gain"],
                    "inner_family_count": alias["inner_family_count"],
                    **row,
                    **{name: proxy[name] for name in PROXIES},
                    "kl_intact_to_attack": proxy["kl_intact_to_attack"],
                }
            )
    write_csv(output / "candidate_endpoint_detail.csv", endpoint_rows)

    selection_lookup = {(r["method"], r["source_id"], r["prompt_id"]): r for r in selection["candidates"]}
    transfer_lookup = {(r["method"], r["source_id"], r["prompt_id"]): r for r in summary["transfer"]}
    candidate_rows = []
    for alias in candidates:
        key = (alias["method"], alias["source_id"], alias["prompt_id"])
        prompt, source_prompt = prompts[alias["prompt_id"]], prompts[source_ids[alias["source_id"]]]
        result = {
            "method": alias["method"],
            "source_id": alias["source_id"],
            "prompt_id": alias["prompt_id"],
            "category": prompt["category"],
            "prompt": prompt["prompt"],
            "source_prompt": source_prompt["prompt"],
            "round": alias["round"],
            "candidate_index": alias["candidate_index"],
            "final_committed": alias["final_committed"],
            "inner_proxy_gain": alias["inner_proxy_gain"],
            "inner_family_count": alias["inner_family_count"],
        }
        all_labels = set()
        for split, frozen in (("confirmation", selection_lookup[key]), ("heldout", transfer_lookup[key])):
            group_rows = [r for r in response_rows if r["prompt_id"] == alias["prompt_id"] and r["split"] == split]
            source_rows = [r for r in response_rows if r["prompt_id"] == source_ids[alias["source_id"]] and r["split"] == split]
            intact_correct = int(group_rows[0]["intact_correct"])
            source_correct = int(source_rows[0]["intact_correct"])
            reasons = frozen["reasons"] if split == "confirmation" else frozen["heldout_reasons"]
            sensitivity, coverage = {}, {}
            for family in BEHAVIOR_FAMILIES:
                for strength in STRENGTHS:
                    rows = [r for r in group_rows if r["family"] == family and r["strength"] == strength]
                    valid = [r for r in rows if bool_value(r["utility_qualified"]) is True]
                    good = sum(bool_value(r["raw_detected"]) is True and float(r["mmd2"]) > 0 for r in valid)
                    status = group_status(len(valid), good)
                    sensitivity[f"{family}/{strength}"] = {
                        "utility_qualified_seeds": len(valid),
                        "raw_detected_valid_seeds": sum(bool_value(r["raw_detected"]) is True for r in valid),
                        "h8_detected_valid_seeds": sum(bool_value(r["h8_detected"]) is True for r in valid),
                        "qualifying_seed_count": good,
                        "status": status,
                    }
                    if status == "not_evaluable_attack_utility":
                        all_labels.add("attack_configuration_utility_failure")
                    elif status == "behavior_insensitive":
                        all_labels.add(f"{family}_behavior_insensitive")
                    elif status == "cross_seed_unstable":
                        all_labels.add("cross_seed_unstable")
            for family in COVERAGE_FAMILIES:
                row = next(r for r in group_rows if r["family"] == family)
                source_row = next(r for r in source_rows if r["family"] == family)
                utility_ok = bool_value(row["utility_qualified"]) is True
                delta = float(row["mmd2"]) - float(source_row["mmd2"])
                status = "not_evaluable_attack_utility" if not utility_ok else ("pass" if delta >= -0.01 else "coverage_degraded")
                coverage[family] = {
                    "utility_qualified": utility_ok,
                    "candidate_mmd2": float(row["mmd2"]),
                    "source_mmd2": float(source_row["mmd2"]),
                    "candidate_minus_source_mmd2": delta,
                    "status": status,
                }
                if not utility_ok:
                    all_labels.add("attack_configuration_utility_failure")
                elif status == "coverage_degraded":
                    all_labels.add("pruning_quantization_coverage_degradation")
            if "intact_task" in reasons or "task_drop" in reasons:
                all_labels.add("candidate_task_correctness_failure")
            result.update(
                {
                    f"{split}_intact_correct_of_8": intact_correct,
                    f"{split}_source_intact_correct_of_8": source_correct,
                    f"{split}_task_stable": intact_correct >= 6 and (source_correct - intact_correct) / 8 <= 0.125,
                    f"{split}_passed_frozen_gate": frozen["passed"] if split == "confirmation" else frozen["heldout_audit_passed"],
                    f"{split}_failed_component_count": len(reasons),
                    f"{split}_passed_component_count_of_11": 11 - len(reasons),
                    f"{split}_reasons": ";".join(reasons),
                    f"{split}_sensitivity": json_cell(sensitivity),
                    f"{split}_coverage": json_cell(coverage),
                }
            )
        result["failure_categories_without_proxy_gap"] = ";".join(sorted(all_labels))
        candidate_rows.append(result)

    alignment_rows = []
    for method in ("legacy", "enhanced"):
        method_aliases = [a for a in candidates if a["method"] == method]
        for family in (*BEHAVIOR_FAMILIES, *COVERAGE_FAMILIES):
            strengths = STRENGTHS if family in BEHAVIOR_FAMILIES else ("coverage",)
            train_endpoints = [v["variant_id"] for v in design["variants"] if v["split"] == "train" and v["family"] == family]
            for strength in strengths:
                heldout_endpoints = [v["variant_id"] for v in design["variants"] if v["split"] == "heldout" and v["family"] == family and v["strength"] == strength]
                all_utility = all(summary["utility"]["heldout"][e]["qualified"] for e in heldout_endpoints)
                for proxy_name in PROXIES:
                    block = []
                    for alias in method_aliases:
                        proxy_values = [float(proxy_by_key[(alias["prompt_id"], e)][proxy_name]) for e in train_endpoints]
                        proxy_score = mean(proxy_values) if proxy_name == "top1_flip" else statistics.median(proxy_values)
                        held = [response_by_key[(alias["prompt_id"], e)] for e in heldout_endpoints]
                        block.append(
                            {
                                "method": method,
                                "source_id": alias["source_id"],
                                "prompt_id": alias["prompt_id"],
                                "family": family,
                                "strength": strength,
                                "proxy": proxy_name,
                                "train_proxy_score": proxy_score,
                                "heldout_raw_distance_mean": mean(float(r["raw_distance"]) for r in held),
                                "heldout_h8_mmd2_mean": mean(float(r["mmd2"]) for r in held),
                                "heldout_raw_detected_fraction": mean(float(bool_value(r["raw_detected"]) is True) for r in held),
                                "heldout_h8_detected_fraction": mean(float(bool_value(r["h8_detected"]) is True) for r in held),
                                "all_heldout_endpoints_utility_qualified": all_utility,
                            }
                        )
                    proxy_ranks = rankdata([r["train_proxy_score"] for r in block])
                    behavior_ranks = rankdata([r["heldout_raw_distance_mean"] for r in block])
                    n = len(block)
                    for row, pr, br in zip(block, proxy_ranks, behavior_ranks):
                        row["proxy_rank_percentile"] = (pr - 1) / (n - 1) if n > 1 else None
                        row["behavior_rank_percentile"] = (br - 1) / (n - 1) if n > 1 else None
                        row["high_proxy_low_behavior_flag"] = bool(
                            all_utility and n >= 8 and row["proxy_rank_percentile"] >= 0.75 and row["behavior_rank_percentile"] <= 0.25
                        )
                        alignment_rows.append(row)
    write_csv(output / "proxy_behavior_alignment.csv", alignment_rows)

    flagged = defaultdict(set)
    for row in alignment_rows:
        if row["high_proxy_low_behavior_flag"]:
            flagged[(row["method"], row["source_id"], row["prompt_id"])].add(f"{row['family']}/{row['strength']}/{row['proxy']}")
    for row in candidate_rows:
        key = (row["method"], row["source_id"], row["prompt_id"])
        values = sorted(flagged[key])
        row["high_proxy_low_behavior_flags"] = ";".join(values)
        labels = set(filter(None, row["failure_categories_without_proxy_gap"].split(";")))
        if values:
            labels.add("internal_proxy_high_behavior_effect_low")
        row["failure_categories"] = ";".join(sorted(labels))
    write_csv(output / "candidate_failure_detail.csv", candidate_rows)

    failure_counts = []
    for split in ("confirmation", "heldout"):
        for method in ("legacy", "enhanced"):
            subset = [r for r in candidate_rows if r["method"] == method]
            counts = Counter(reason for r in subset for reason in r[f"{split}_reasons"].split(";") if reason)
            for reason, count in sorted(counts.items()):
                failure_counts.append({"split": split, "method": method, "failure_component": reason, "candidate_count": count, "total_candidates": len(subset)})
    write_csv(output / "failure_component_counts.csv", failure_counts)

    detection_rows = []
    for split in ("confirmation", "heldout"):
        for family in (*BEHAVIOR_FAMILIES, *COVERAGE_FAMILIES):
            rows = [r for r in response_rows if r["split"] == split and r["family"] == family]
            valid = [r for r in rows if bool_value(r["utility_qualified"]) is True]
            for scope, values in (("all", rows), ("utility_qualified_only", valid)):
                detection_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "scope": scope,
                        "cells": len(values),
                        "raw_detected": sum(bool_value(r["raw_detected"]) is True for r in values),
                        "h8_detected": sum(bool_value(r["h8_detected"]) is True for r in values),
                        "both_detected": sum(bool_value(r["raw_detected"]) is True and bool_value(r["h8_detected"]) is True for r in values),
                        "raw_only": sum(bool_value(r["raw_detected"]) is True and bool_value(r["h8_detected"]) is False for r in values),
                        "h8_only": sum(bool_value(r["raw_detected"]) is False and bool_value(r["h8_detected"]) is True for r in values),
                    }
                )
    write_csv(output / "detection_channel_summary.csv", detection_rows)

    correlation_rows = []
    for method in ("legacy", "enhanced"):
        valid = [r for r in alignment_rows if r["method"] == method and r["all_heldout_endpoints_utility_qualified"]]
        groups = sorted({(r["family"], r["strength"], r["proxy"]) for r in valid})
        for family, strength, proxy in groups:
            rows = [r for r in valid if (r["family"], r["strength"], r["proxy"]) == (family, strength, proxy)]
            correlation_rows.append(
                {
                    "method": method,
                    "family": family,
                    "strength": strength,
                    "proxy": proxy,
                    "n_candidate_aliases": len(rows),
                    "spearman_with_heldout_raw_distance": spearman([r["train_proxy_score"] for r in rows], [r["heldout_raw_distance_mean"] for r in rows]),
                    "spearman_with_heldout_h8_mmd2": spearman([r["train_proxy_score"] for r in rows], [r["heldout_h8_mmd2_mean"] for r in rows]),
                    "interpretation": "descriptive_only_no_iid_significance",
                }
            )
    write_csv(output / "candidate_level_proxy_correlations.csv", correlation_rows)

    source_correlation_rows = []
    for row in summary["correlations"]:
        source_correlation_rows.append({k: row.get(k) for k in (
            "method", "family", "strength", "proxy", "target_channel", "target",
            "all_endpoints_utility_qualified", "rho", "n", "status", "p_value", "reason"
        )})
    write_csv(output / "source_aggregated_proxy_correlations.csv", source_correlation_rows)

    method_rows = []
    for method in ("legacy", "enhanced"):
        subset = [r for r in candidate_rows if r["method"] == method]
        failed = [r["confirmation_failed_component_count"] for r in subset]
        frozen = next(r for r in summary["methods"] if r["method"] == method)
        method_rows.append(
            {
                **frozen,
                "confirmation_min_failed_components": min(failed),
                "confirmation_median_failed_components": statistics.median(failed),
                "confirmation_mean_failed_components": mean(failed),
                "confirmation_max_passed_components_of_11": max(r["confirmation_passed_component_count_of_11"] for r in subset),
                "candidates_with_stable_intact_task": sum(r["confirmation_task_stable"] for r in subset),
                "candidates_with_high_proxy_low_behavior_flag": sum(bool(r["high_proxy_low_behavior_flags"]) for r in subset),
            }
        )
    write_csv(output / "method_closeness_summary.csv", method_rows)

    verification.update(
        {
            "response_matrix_rows": len(response_rows),
            "expected_response_matrix_rows": len(prompts) * sum(v["split"] in ("confirmation", "heldout") for v in design["variants"]),
            "proxy_records": len(proxy_records),
            "expected_proxy_records": len(prompts) * len(design["variants"]),
            "candidate_aliases": len(candidates),
            "unique_response_prompts": len(prompts),
        }
    )
    if verification["response_matrix_rows"] != verification["expected_response_matrix_rows"]:
        raise ValueError("Response decision matrix is incomplete")
    if verification["proxy_records"] != verification["expected_proxy_records"]:
        raise ValueError("Proxy matrix is incomplete")
    (output / "VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    utility_counts = Counter((r["split"], r["family"], bool_value(r["qualified"])) for r in utility_rows)
    gate_counts = Counter(reason for r in candidate_rows for reason in r["confirmation_reasons"].split(";") if reason)
    report = [
        "# Stage2现有数据失败拆解报告",
        "",
        "本文件是对冻结实验结果的派生分析，不是新的实验，也未改变候选筛选门槛。",
        "",
        "## 证据完整性",
        "",
        f"- SUMMARY绑定的证据哈希：{verification['evidence_hashes_checked']}项，缺失0项、不一致0项。",
        f"- H6搜索结果：{verification['search_hashes_checked']}份，哈希全部一致。",
        f"- 原始响应：{verification['response_rows_counted']}行；判定矩阵：{len(response_rows)}个提示词×攻击端点单元。",
        f"- 候选别名：{len(candidates)}个；去重响应提示词（含6个source基线）：{len(prompts)}个。",
        "",
        "## 直接结果",
        "",
        "| 方法 | 候选 | confirmation最少失败项 | 失败项中位数 | 最接近候选通过项/11 | 搜索秒数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        report.append(f"| {row['method']} | {row['candidate_aliases']} | {row['confirmation_min_failed_components']} | {row['confirmation_median_failed_components']} | {row['confirmation_max_passed_components_of_11']} | {row['search_seconds']} |")
    report += [
        "",
        "两组均无候选通过冻结门槛；表中的通过项只表示距离门槛的描述性接近程度，不构成重新筛选。",
        "",
        "## 攻击效用",
        "",
        "| 阶段 | 攻击族 | 合格端点/总端点 |",
        "|---|---|---:|",
    ]
    for split in ("confirmation", "heldout"):
        for family in (*BEHAVIOR_FAMILIES, *COVERAGE_FAMILIES):
            total = sum(utility_counts[(split, family, flag)] for flag in (True, False))
            report.append(f"| {split} | {family} | {utility_counts[(split, family, True)]}/{total} |")
    report += [
        "",
        "攻击效用失败是端点级阻塞：当一个强度少于2个合格seed，任何候选都无法通过该强度；覆盖攻击效用不合格时，任何候选都无法通过对应覆盖门槛。",
        "",
        "## Confirmation冻结门槛失败频次",
        "",
        "| 失败组件 | 候选数/21 |",
        "|---|---:|",
    ]
    for reason, count in sorted(gate_counts.items()):
        report.append(f"| {reason} | {count}/21 |")
    report += [
        "",
        "## 解释规则",
        "",
        "- `not_evaluable_attack_utility`：有效seed不足，不能归因为提示词行为不敏感。",
        "- `behavior_insensitive`：已有至少2个效用合格seed，但合格seed上没有满足冻结行为条件的检出。",
        "- `cross_seed_unstable`：已有足够合格seed，但只有1个seed满足条件。",
        "- `high_proxy_low_behavior_flag`：仅在全部held-out端点效用合格、同方法候选数至少8时，代理位于前25%而Raw连续行为距离位于后25%；这是派生诊断标记，不是显著性检验。",
        "",
        "## 结论边界",
        "",
        "本分析可以区分攻击不可判定与候选真实不敏感，并比较两种H6离冻结门槛的距离；不能把六源重复模板试跑升级为完整Stage2，也不能用描述性相关性宣布某个代理显著最优。",
        "",
        "逐候选、逐端点与代理对齐结果见同目录CSV。",
    ]
    (output / "现有数据失败拆解报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
