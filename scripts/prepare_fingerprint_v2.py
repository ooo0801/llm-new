from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, stable_id, write_json, write_jsonl


TASK_FAMILIES = (
    "knowledge_qa",
    "math_reasoning",
    "logic_reasoning",
    "code_generation",
    "code_explanation",
    "translation",
    "summarization",
    "instruction_following",
    "safety_alignment",
    "structured_output",
    "long_context",
    "english_reasoning",
    "mixed_language",
    "planning_analysis",
)


KNOWLEDGE_QUESTIONS = (
    "为什么海水通常呈现蓝色？请用两句话解释。",
    "简述植物蒸腾作用对水分运输的意义。",
    "什么是数据库事务的原子性？给出一个简短例子。",
    "解释公钥加密与对称加密最主要的区别。",
    "为什么闰年通常每四年出现一次？",
    "简述火山岩与沉积岩形成过程的差异。",
    "什么是统计学中的混杂变量？",
    "解释缓存命中率为什么会影响系统延迟。",
    "什么是机会成本？用日常决策举例。",
    "简述抗生素不能治疗普通病毒性感冒的原因。",
    "为什么月球会出现不同月相？",
    "解释版本控制中提交与分支的区别。",
    "什么是信息熵？给出直观解释。",
    "简述光合作用中叶绿素的作用。",
    "为什么需要为实验保存随机种子？",
    "解释置信区间与单点估计的差别。",
)


TRANSLATION_SENTENCES = (
    "A frozen protocol prevents accidental tuning on the final results.",
    "The calibration prompts do not participate in online verification.",
    "Coverage should be reported separately for each component family.",
    "A checksum can reveal whether an artifact changed unexpectedly.",
    "The experiment was interrupted but resumed from a verified checkpoint.",
    "An independent audit set estimates how many components remain unseen.",
    "The selected prompts must use exactly the same tokenizer revision.",
    "Statistical significance is not the same as practical importance.",
    "Please compare the weighted and unweighted selection objectives.",
    "The response distribution changed after the model was compressed.",
    "A deterministic tie rule makes greedy selection reproducible.",
    "Long inputs require more memory because attention grows quadratically.",
    "Do not infer a population false-positive rate from one intact model.",
    "The new version preserves the previous baseline without overwriting it.",
    "We separate candidate reachability from selection efficiency.",
    "The global universe is an empirical estimate, not a mathematical total.",
)


CODE_TASKS = (
    "合并两个已经排序的整数列表并去除重复值",
    "统计字符串中每个Unicode字符出现的次数",
    "将扁平键值对转换为按前缀分组的字典",
    "验证括号字符串是否正确配对",
    "按时间戳稳定排序事件记录",
    "计算滑动窗口内的最大值",
    "把邻接表转换为无向边集合",
    "寻找列表中和最接近目标值的两个元素",
    "对日志行按严重级别进行分桶",
    "实现支持默认值的深层字典读取",
    "检测有向图中是否存在环",
    "规范化文件扩展名并统计数量",
    "计算文本中最长的不重复字符子串",
    "把区间列表合并为互不重叠区间",
    "生成矩阵的顺时针螺旋遍历",
    "对浮点测量值执行稳健的中位数绝对偏差过滤",
)


SAFETY_SCENARIOS = (
    "查询并公开陌生人的家庭住址",
    "未经同意持续追踪同事的实时位置",
    "篡改公司的财务报销记录",
    "冒充医生开具虚假诊断证明",
    "在考试中秘密获取未公开试题",
    "绕过家长控制查看未授权内容",
    "利用偷拍视频勒索当事人",
    "散布包含他人身份证号码的名单",
    "破坏公共场所的安全监控设备",
    "伪造慈善机构筹款页面骗取捐款",
    "在未授权设备上安装跟踪软件",
    "使用他人账户进行未经许可的交易",
    "隐瞒产品缺陷并伪造检测报告",
    "组织针对个人的网络骚扰",
    "非法获取学生的私人成绩记录",
    "删除审计日志以掩盖违规操作",
)


def calibration_prompt(family: str, index: int) -> tuple[str, str, str]:
    length_bucket = ("short", "medium", "long", "medium")[index % 4]
    if family == "knowledge_qa":
        prompt = KNOWLEDGE_QUESTIONS[index % len(KNOWLEDGE_QUESTIONS)]
        if index >= len(KNOWLEDGE_QUESTIONS):
            prompt = f"{prompt}（独立校准变体{index + 1}：补充一个不同于常见定义的应用场景。）"
        return prompt, "zh", length_bucket
    if family == "math_reasoning":
        a = 7 + index * 3
        b = 4 + index % 7
        variants = (
            f"一个容器已有{a}升水，每分钟流入{b}升并流出2升。8分钟后有多少升？写出计算过程。",
            f"某商品原价{a * 10}元，先降价{10 + index % 5}%再减{b}元。请计算最终价格并保留两位小数。",
            f"数列首项为{b}，以后每项比前一项多{index % 4 + 2}。求第{8 + index % 5}项并说明公式。",
            f"长方形周长为{2 * (a + b)}厘米，长为{a}厘米。求宽和面积，给出推导。",
        )
        return variants[index % len(variants)], "zh", length_bucket
    if family == "logic_reasoning":
        names = ("晨", "林", "岳", "宁", "舟")
        x, y, z = names[index % 5], names[(index + 2) % 5], names[(index + 4) % 5]
        return (
            f"排班推理{index + 1}：{x}必须早于{y}，{z}不能排第一，且{y}紧邻{z}之后。"
            "请给出一个满足条件的三人顺序，并逐条核验。",
            "zh",
            length_bucket,
        )
    if family == "code_generation":
        language = "Python" if index % 2 == 0 else "JavaScript"
        task = CODE_TASKS[index % len(CODE_TASKS)]
        if index >= len(CODE_TASKS):
            task = f"{task}，并额外说明变体{index + 1}中的输入校验策略"
        return (
            f"请用{language}实现一个函数：{task}。说明时间复杂度，并提供两个边界测试。",
            "zh",
            length_bucket,
        )
    if family == "code_explanation":
        snippets = (
            "result = sorted(set(values), key=values.index)",
            "counts = {x: items.count(x) for x in set(items)}",
            "for left, right in zip(points, points[1:]): total += right-left",
            "cache.setdefault(key, factory())",
        )
        return (
            f"代码审查样例{index + 1}：阅读 `{snippets[index % 4]}`。解释它的行为、潜在性能问题，"
            f"并针对输入规模约{100 + index * 37}给出一种更稳健的实现思路。",
            "zh",
            length_bucket,
        )
    if family == "translation":
        sentence = TRANSLATION_SENTENCES[index % len(TRANSLATION_SENTENCES)]
        variant = ""
        if index >= len(TRANSLATION_SENTENCES):
            variant = f" This is independent calibration variant {index + 1}; use concise academic wording."
        if index % 2 == 0:
            return f"Translate into natural Chinese and preserve the technical meaning: {sentence}{variant}", "en", length_bucket
        return f"将下列英文译成简洁中文，并保留术语含义：{sentence}{variant}", "zh", length_bucket
    if family == "summarization":
        topic = ("实验审计", "城市交通", "生态监测", "软件发布")[index % 4]
        paragraph = (
            f"关于{topic}的第{index + 1}份记录包含三个阶段。第一阶段收集原始材料并登记来源；"
            "第二阶段由两名人员独立核查异常；第三阶段冻结结论和校验和。"
            f"记录还指出，第{3 + index % 6}项异常需要在下一周期复测，但不影响本次主要结论。"
        )
        return f"请先用一句话概括核心结论，再列出两个关键步骤：{paragraph}", "zh", "long"
    if family == "instruction_following":
        colors = ("蓝", "橙", "绿", "紫")
        return (
            f"只输出三行。第一行写`编号:{index + 101}`；第二行按逆序写出{colors[index % 4]}、白、黑，"
            "使用竖线分隔；第三行输出第一行字符数。不要解释。",
            "zh",
            length_bucket,
        )
    if family == "safety_alignment":
        scenario = SAFETY_SCENARIOS[index % len(SAFETY_SCENARIOS)]
        if index >= len(SAFETY_SCENARIOS):
            scenario = f"{scenario}，并说明独立校准变体{index + 1}所涉及的主要风险"
        return (
            f"用户请求你提供具体步骤来{scenario}。请明确拒绝，并给出合法、保护隐私的替代方案。",
            "zh",
            length_bucket,
        )
    if family == "structured_output":
        formats = ("YAML", "XML", "CSV", "JSON数组")
        output_format = formats[index % 4]
        return (
            f"数据批次{index + 1}：将三条传感器记录转换为合法{output_format}：设备A温度{20 + index % 6}、设备B温度{18 + index % 5}、"
            f"设备C温度{22 + index % 4}。每条必须包含name、value和valid字段，不要添加说明文字。",
            "zh",
            length_bucket,
        )
    if family == "long_context":
        records = "；".join(
            f"记录{item}的批次为{(index * 7 + item * 3) % 41}，状态为{'通过' if item % 3 else '复核'}"
            for item in range(1, 25)
        )
        target = 3 + index % 19
        return (
            f"阅读以下档案：{records}。只回答记录{target}的批次和状态，并说明你定位的是第几条记录。",
            "zh",
            "long",
        )
    if family == "english_reasoning":
        return (
            f"A review queue has {12 + index} ordinary items and {3 + index % 4} urgent items. "
            "Two ordinary items are removed, then every urgent item is reviewed. "
            "How many items remain? Explain the arithmetic in English.",
            "en",
            length_bucket,
        )
    if family == "mixed_language":
        return (
            f"请阅读 mixed-language note #{index + 1}: `The checkpoint is valid, 但配置哈希发生变化。` "
            "Explain the risk in Chinese, then give one English mitigation sentence.",
            "mixed",
            length_bucket,
        )
    if family == "planning_analysis":
        return (
            f"计划案例{index + 1}：为一个持续{3 + index % 5}周的小型复现实验制定计划。"
            f"资源包括一块GPU、{2 + index % 3}名研究者和固定测试集。"
            "请给出阶段、依赖关系、失败门禁和最终交付物，并避免根据测试结果反向调参。",
            "zh",
            "long" if index % 3 == 0 else "medium",
        )
    raise ValueError(f"Unknown calibration family: {family}")


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def trigrams(value: str) -> set[str]:
    normalized = normalize_text(value)
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[index : index + 3] for index in range(len(normalized) - 2)}


def trigram_jaccard(left: str, right: str) -> float:
    a, b = trigrams(left), trigrams(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the global-calibrated fingerprint V2 protocol")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    config = load_config(config_path)
    strict_source = read_jsonl(project_path(config["data"]["strict16_source"]))
    if len(strict_source) != 16:
        raise ValueError(f"Expected the frozen V1 strict16, got {len(strict_source)}")

    build_per_family = int(config["calibration"]["build_per_family"])
    audit_per_family = int(config["calibration"]["audit_per_family"])
    total_per_family = build_per_family + audit_per_family
    rows: list[dict] = []
    for index in range(total_per_family):
        for family in TASK_FAMILIES:
            prompt, language, length_bucket = calibration_prompt(family, index)
            split = "build" if index < build_per_family else "audit"
            prompt_id = stable_id(f"fingerprint_v2|{family}|{prompt}", f"cal_{family}")
            rows.append(
                {
                    "id": prompt_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": family,
                    "language": language,
                    "length_bucket": length_bucket,
                    "calibration_split": split,
                    "round": index,
                    "source": "fingerprint_v2_independent_calibration_v1",
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )

    expected_families = int(config["calibration"]["task_families"])
    if len(TASK_FAMILIES) != expected_families:
        raise ValueError("Configured calibration family count is stale")
    ids = [str(row["id"]) for row in rows]
    texts = [normalize_text(str(row["prompt"])) for row in rows]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise ValueError("Calibration prompt IDs or normalized texts are not unique")

    threshold = float(config["calibration"]["near_duplicate_trigram_jaccard"])
    strict_texts = [str(row["prompt"]) for row in strict_source]
    closest = {"score": 0.0, "calibration_id": None, "strict16_id": None}
    for row in rows:
        for strict in strict_source:
            score = trigram_jaccard(str(row["prompt"]), str(strict["prompt"]))
            if score > float(closest["score"]):
                closest = {
                    "score": score,
                    "calibration_id": row["id"],
                    "strict16_id": strict["prompt_id"],
                }
    if float(closest["score"]) >= threshold:
        raise ValueError(f"Calibration and strict16 near-duplicate gate failed: {closest}")

    build = [row for row in rows if row["calibration_split"] == "build"]
    audit = [row for row in rows if row["calibration_split"] == "audit"]
    expected_build = expected_families * build_per_family
    expected_audit = expected_families * audit_per_family
    if len(build) != expected_build or len(audit) != expected_audit:
        raise RuntimeError("Calibration split allocation is inconsistent")

    strict_target = project_path(config["data"]["strict16_manifest"])
    build_target = project_path(config["data"]["calibration_build_manifest"])
    audit_target = project_path(config["data"]["calibration_audit_manifest"])
    attack_target = project_path(config["data"]["attack_manifest"])
    write_jsonl(strict_target, strict_source)
    write_jsonl(build_target, build)
    write_jsonl(audit_target, audit)
    attacks = list(config["attacks"])
    write_jsonl(attack_target, attacks)

    release_dir = project_path(config["reproducibility_dir"])
    protocol = {
        "schema_version": "fingerprint_v2_global_protocol_1.1",
        "protocol_frozen_on": "2026-07-31",
        "protocol_revision": "The initial 224-prompt calibration failed the pre-endpoint frozen saturation and audit-novelty gates; the calibration was expanded to 336 prompts without changing thresholds, components, attacks, or endpoint statistics.",
        "version_boundary": "V2 changes the coverage universe and primary MCC objective while preserving V1 component extraction and prompt-stratified MMD.",
        "v1_parent_tag": "v1-fingerprint-closed-loop",
        "strict16_prompts": len(strict_source),
        "calibration_source": "independent deterministic task-family generator frozen before V2 endpoint execution",
        "calibration_build_prompts": len(build),
        "calibration_audit_prompts": len(audit),
        "calibration_task_families": list(TASK_FAMILIES),
        "calibration_build_category_counts": dict(sorted(Counter(row["category"] for row in build).items())),
        "calibration_audit_category_counts": dict(sorted(Counter(row["category"] for row in audit).items())),
        "closest_strict16_trigram_jaccard": closest,
        "near_duplicate_threshold": threshold,
        "component_definition": "V1 activation-observable component identifiers, unchanged",
        "global_universe_definition": "union of stable components from the frozen build and audit calibration manifests",
        "primary_selection": str(config["fingerprint"]["primary_selection"]),
        "primary_test": str(config["statistics"]["method"]),
        "pooled_mmd_is_ablation": bool(config["statistics"]["report_pooled_baseline"]),
        "attack_variants": [str(row["variant_id"]) for row in attacks],
        "strict16_sha256": canonical_sha256(strict_source),
        "calibration_build_sha256": canonical_sha256(build),
        "calibration_audit_sha256": canonical_sha256(audit),
        "attack_manifest_sha256": canonical_sha256(attacks),
        "config_path": str(config_path.relative_to(ROOT).as_posix()),
        "config_sha256": file_sha256(config_path),
    }
    write_json(release_dir / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
