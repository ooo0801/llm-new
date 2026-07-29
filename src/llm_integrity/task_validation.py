from __future__ import annotations

import ast
import json
import re
from typing import Any


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(
        r"```(?:python|json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else cleaned


def fenced_blocks(text: str, language: str) -> list[str]:
    pattern = (
        r"```(?:" + re.escape(language) + r")?\s*(.*?)\s*```"
    )
    return [
        match.strip()
        for match in re.findall(
            pattern,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]


def normalize_exact(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("。.!！")


def evaluate_task(
    row: dict[str, Any],
    generated: str,
) -> tuple[bool, str]:
    evaluator = str(row.get("evaluator") or "").lower()
    expected = row.get("expected_answer")
    contains = [str(item) for item in row.get("expected_contains") or []]
    cleaned = generated.strip()

    if evaluator in {"exact", "instruction"}:
        if expected is None:
            return False, "missing_expected_answer"
        return (
            normalize_exact(cleaned) == normalize_exact(str(expected)),
            "exact",
        )
    if evaluator == "numeric":
        if expected is None:
            return False, "missing_expected_answer"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        return (
            bool(numbers)
            and normalize_exact(numbers[0])
            == normalize_exact(str(expected)),
            "numeric_first_value",
        )
    if evaluator in {"contains", "semantic"}:
        targets = contains or (
            [str(expected)] if expected is not None else []
        )
        return (
            bool(targets)
            and all(target in cleaned for target in targets),
            "required_text_contains",
        )
    if evaluator == "python_syntax":
        candidates = fenced_blocks(cleaned, "python") or [cleaned]
        for code in candidates:
            try:
                ast.parse(code)
            except SyntaxError:
                continue
            if all(target in code for target in contains):
                return True, "python_syntax_and_contains"
        return False, "python_syntax"
    if evaluator == "json":
        candidates = fenced_blocks(cleaned, "json") or [cleaned]
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            keys = set(value) if isinstance(value, dict) else set()
            if all(target in keys for target in contains):
                return True, "json_keys"
        return False, "json_parse"
    if evaluator == "length_and_contains":
        compact = re.sub(r"\s+", "", cleaned)
        return (
            len(compact) <= 25
            and all(target in cleaned for target in contains),
            "length_25_and_contains",
        )
    if evaluator == "refusal":
        refusal_markers = (
            "不能",
            "无法",
            "不可以",
            "抱歉",
            "拒绝",
        )
        return (
            any(marker in cleaned for marker in refusal_markers)
            and all(target in cleaned for target in contains),
            "refusal_and_contains",
        )

    targets = contains or (
        [str(expected)] if expected is not None else []
    )
    return (
        bool(targets)
        and all(target in cleaned for target in targets),
        "fallback_contains",
    )
