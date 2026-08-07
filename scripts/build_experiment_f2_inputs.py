from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from llm_integrity.io import stable_id


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
FAMILIES = [
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
]
DEVELOPMENT_SEEDS = [2026080811, 2026080812, 2026080813]
CONFIRMATION_SEEDS = [2026080821, 2026080822]
EXPERIMENT_C_CALIBRATION_SHA256 = (
    "66944c94c523840938c8444c5b696138fd0f4cc8ce2e628be6ef2629d066ad0c"
)
EXPERIMENT_C_TRAIN_REGISTRY_SHA256 = (
    "288791e63732aa4410ee25e6f8f81c4fc1ee461c35c10964bf840a004e385f5f"
)


LOGIC_ITEMS = [
    ("甲在乙左边，乙在丙左边。从左到右最先是谁？只回答名字。", "甲"),
    ("红盒比蓝盒重，蓝盒比绿盒重。三个盒子中最轻的是哪个？只回答颜色。", "绿"),
    ("小林比小周早到，小周比小吴早到。三人中最后到的是谁？只回答名字。", "小吴"),
    ("A队得分高于B队，B队得分高于C队。得分第二的是哪队？只回答队名。", "B队"),
    ("所有玫瑰都是花，植株甲是玫瑰。植株甲是不是花？只回答“是”或“否”。", "是"),
    ("若开关打开，则指示灯亮。现在开关已经打开，指示灯是否会亮？只回答“会”或“不会”。", "会"),
    ("丙的年龄小于乙，乙的年龄小于甲。三人中年龄最大的是谁？只回答名字。", "甲"),
    ("书A在书B上面，书B在书C上面。最下面是哪本书？只回答书名。", "书C"),
    ("东门离大厅比西门近，西门又比南门近。离大厅最远的是哪个门？只回答名称。", "南门"),
    ("整数7是奇数。命题“7能被2整除”是真是假？只回答“真”或“假”。", "假"),
    ("如果今天是星期三，那么明天是星期四。今天确实是星期三，明天是星期几？只回答星期。", "星期四"),
    ("甲、乙、丙排队，甲不在最前，乙在甲前面，丙在乙前面。最前面是谁？只回答名字。", "丙"),
    ("只有一个盒子装有球。已知甲盒为空、乙盒有球，球在哪个盒子？只回答盒名。", "乙盒"),
    ("所有方形都是四边形，图形P不是四边形。图形P是否可能是方形？只回答“不可能”或“可能”。", "不可能"),
    ("比赛中丁领先戊，戊领先己，己领先庚。排名第三的是谁？只回答名字。", "己"),
    ("若文件通过校验则标记为有效。文件X没有被标记为有效，能否断定它通过了校验？只回答“不能”或“能”。", "不能"),
]


SUMMARY_ITEMS = [
    ("本次实验固定随机种子后重复运行三次，并比较每次输出是否一致。", ["实验", "随机种子"]),
    ("研究人员将模型量化为低精度格式，显存占用明显下降，同时记录推理速度。", ["量化", "显存"]),
    ("团队对模型执行结构化剪枝，再测量剪枝前后的准确率变化。", ["剪枝", "准确率"]),
    ("测试向参数加入不同强度的噪声，并用多次重复评估输出稳定性。", ["噪声", "稳定性"]),
    ("模型微调五十步，训练过程中持续记录损失并保存最终适配器。", ["微调", "损失"]),
    ("系统向原始模型和修改模型发送相同问题，然后逐项比较两者响应。", ["模型", "响应"]),
    ("完整指纹由十二条提示组成，验证一次只需要进行少量查询。", ["指纹", "查询"]),
    ("研究比较两组特征分布，并通过置换检验判断差异是否显著。", ["分布", "检验"]),
    ("程序收集各层稳定组件，计算候选提示对组件集合的覆盖率。", ["组件", "覆盖率"]),
    ("服务器连续执行评测任务，日志记录了每个阶段的运行时间。", ["服务器", "运行时间"]),
    ("实验开始前冻结数据版本，并在异地保存一份只读备份。", ["数据", "备份"]),
    ("每次运行都保存配置文件，同时写明模型、代码和依赖版本。", ["配置", "版本"]),
    ("为保证结果可以复现，报告完整列出了软件环境和随机设置。", ["复现", "环境"]),
    ("抽样过程覆盖不同任务类别，以降低单一类别造成的样本偏差。", ["样本", "偏差"]),
    ("开发集用于确定固定阈值，独立测试集只用于最后一次验证。", ["阈值", "验证"]),
    ("报告汇总主要结果，并为检出率给出百分之九十五置信区间。", ["结果", "置信区间"]),
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def base_configuration(family: str) -> dict[str, Any]:
    return {
        "unstructured_pruning": {
            "type": family,
            "ratio": 0.30,
            "method": "global_magnitude",
            "target_scope": "attention",
        },
        "quantization": {
            "type": family,
            "method": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "target_scope": "full_model",
        },
        "gaussian_noise": {
            "type": family,
            "std_ratio": 0.0005,
            "target_scope": "ffn",
            "scale_rule": "parameter_tensor_std",
        },
        "finetuning": {
            "type": family,
            "method": "lora",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "learning_rate": 1e-5,
            "steps": 50,
            "target_scope": "attention_ffn",
            "data_source": "isolated_attack_training_data",
        },
    }[family]


def structured_configuration(split: str, index: int) -> dict[str, Any]:
    development = [
        {"structure": "ffn_channels", "selection": "random", "layer_scope": "all_layers", "ratio": 0.05},
        {"structure": "attention_heads", "selection": "random", "layer_scope": "all_layers", "ratio": 0.05},
        {"structure": "attention_heads", "selection": "magnitude", "layer_scope": "random_layer_subset", "ratio": 0.05},
    ]
    confirmation = [
        {"structure": "ffn_channels", "selection": "magnitude", "layer_scope": "all_layers", "ratio": 0.06},
        {"structure": "attention_heads", "selection": "random", "layer_scope": "random_layer_subset", "ratio": 0.06},
    ]
    value = dict((development if split == "validation" else confirmation)[index])
    value.update({"type": "structured_pruning", "implementation": "mask"})
    return value


def build_manifest(split: str, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "f2_dev" if split == "validation" else "f2_confirm"
    for family in FAMILIES:
        for index, seed in enumerate(seeds):
            config = structured_configuration(split, index) if family == "structured_pruning" else base_configuration(family)
            identity = {"split": split, "family": family, "seed": seed, "configuration": config}
            rows.append(
                {
                    "configuration": config,
                    "family": family,
                    "seed": seed,
                    "split": split,
                    "variant_id": f"{prefix}_{family}_{canonical_sha256(identity)[:12]}",
                    "weight": 0.2,
                }
            )
    return rows


def build_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (body, answer) in enumerate(LOGIC_ITEMS, start=1):
        prompt = f"独立逻辑题F2-{index:02d}：{body}"
        rows.append(
            {
                "id": stable_id(f"f2|logic|{prompt}", "logic_f2"),
                "prompt_id": stable_id(f"f2|logic|{prompt}", "logic_f2"),
                "category": "logic",
                "language": "zh",
                "prompt": prompt,
                "evaluator": "exact",
                "expected_answer": answer,
                "source": "f2_targeted_expansion_v1",
                "split": "construction",
            }
        )
    for index, (paragraph, keywords) in enumerate(SUMMARY_ITEMS, start=1):
        quoted = "、".join(f"“{item}”" for item in keywords)
        prompt = (
            f"独立摘要题F2-{index:02d}：请将下文概括为不超过25个汉字的一句话；"
            f"摘要必须同时包含{quoted}，不要添加标题或解释。下文：{paragraph}"
        )
        rows.append(
            {
                "id": stable_id(f"f2|summary|{prompt}", "summary_f2"),
                "prompt_id": stable_id(f"f2|summary|{prompt}", "summary_f2"),
                "category": "summary",
                "language": "zh",
                "prompt": prompt,
                "evaluator": "length_and_contains",
                "expected_contains": keywords,
                "source": "f2_targeted_expansion_v1",
                "split": "construction",
            }
        )
    return rows


def main() -> None:
    output = ROOT / "experiments/prompt-robust-14b-f2/inputs"
    old_pool = ROOT / "experiments/prompt-robust-14b-f1/inputs/candidate_pairs64.jsonl"
    sources = build_sources()
    if len(sources) != 32 or Counter(row["category"] for row in sources) != {"logic": 16, "summary": 16}:
        raise AssertionError("F2 expansion must contain exactly 16 logic and 16 summary sources")
    if len({row["id"] for row in sources}) != len(sources) or len({row["prompt"] for row in sources}) != len(sources):
        raise ValueError("F2 expansion source IDs and texts must be unique")
    old_rows = [json.loads(line) for line in old_pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_initial = {str(row["initial_prompt"]).strip() for row in old_rows}
    old_optimized = {str(row["optimized_prompt"]).strip() for row in old_rows}
    overlap = {row["prompt"] for row in sources} & (old_initial | old_optimized)
    if overlap:
        raise ValueError(f"new source text overlaps the frozen F1 pool: {sorted(overlap)}")

    development = build_manifest("validation", DEVELOPMENT_SEEDS)
    confirmation = build_manifest("test", CONFIRMATION_SEEDS)
    write_jsonl(output / "expansion_sources32.jsonl", sources)
    write_jsonl(output / "smoke_sources2.jsonl", [sources[0], sources[16]])
    write_jsonl(output / "attack_manifest_development_3each.jsonl", development)
    write_jsonl(output / "attack_manifest_confirmation_2each.jsonl", confirmation)
    design = {
        "schema_version": "experiment_f2_design_1.0",
        "model": {"id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "placement": "balanced_3gpu_no_cpu_disk_offload"},
        "expansion": {"rows": 32, "category_counts": {"logic": 16, "summary": 16}, "source": "f2_targeted_expansion_v1"},
        "construction": {
            "method": "experiment_c_discrete_joint_hotflip_v4",
            "rounds": 3,
            "maximum_candidates_per_source": 2,
            "minimum_nondegraded_families": 3,
            "minimum_unique_candidate_sources_per_expanded_category": 3,
            "calibration_path": "results/experiment_c_qwen14b_20260806/01_calibration/discrete_calibration_v2.json",
            "calibration_sha256": EXPERIMENT_C_CALIBRATION_SHA256,
            "training_registry_path": "results/experiment_c_qwen14b_20260806/adapters/registry_train.json",
            "training_registry_sha256": EXPERIMENT_C_TRAIN_REGISTRY_SHA256,
        },
        "pool": {
            "base": "experiments/prompt-robust-14b-f1/inputs/candidate_pairs64.jsonl",
            "base_rows": 64,
            "base_sha256": sha256(old_pool),
            "assembly": "base64_plus_all_valid_new_portfolio_candidates; exact ID/text deduplication; one-source rule deferred to frozen selection",
        },
        "development": {"micro_seed": 2026080810, "bootstrap_seed": 2026080814, "attack_seeds": DEVELOPMENT_SEEDS, "variants_per_family": 3},
        "confirmation": {"micro_seed": 2026080820, "bootstrap_seed": 2026080823, "attack_seeds": CONFIRMATION_SEEDS, "variants_per_family": 2},
        "selection": {
            "rows": 30,
            "minimum_per_category": 3,
            "maximum_per_source_prompt": 1,
            "required_categories": ["code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"],
        },
        "confirmation_gate": {"required_legacy_retained": 23, "family_relative_tolerance": 0.01, "minimum_nondegraded_families": 3, "all_categories_required": True},
        "inputs_sha256": {
            "sources": sha256(output / "expansion_sources32.jsonl"),
            "development_manifest": sha256(output / "attack_manifest_development_3each.jsonl"),
            "confirmation_manifest": sha256(output / "attack_manifest_confirmation_2each.jsonl"),
        },
        "leakage_guard": "Construction uses only Experiment C training/calibration evidence. F2 development reranks the assembled pool. Confirmation cannot change candidates, ranking, quotas, attacks, evaluators, or gates.",
    }
    write_json(output / "design_manifest.json", design)
    print(json.dumps(design, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
