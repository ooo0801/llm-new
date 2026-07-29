from __future__ import annotations

from llm_integrity.task_validation import evaluate_task


def test_exact_and_numeric_evaluators() -> None:
    assert evaluate_task(
        {"evaluator": "exact", "expected_answer": "52"},
        "52。",
    )[0]
    assert evaluate_task(
        {"evaluator": "numeric", "expected_answer": "52"},
        "答案是 52。",
    )[0]
    assert not evaluate_task(
        {"evaluator": "numeric", "expected_answer": "52"},
        "答案是 51。",
    )[0]


def test_python_syntax_requires_expected_fragments() -> None:
    row = {
        "evaluator": "python_syntax",
        "expected_contains": ["def square_sum_26", "return"],
    }
    valid = "```python\ndef square_sum_26(values):\n    return 0\n```"
    invalid = "def other(values):\n    return 0"
    assert evaluate_task(row, valid)[0]
    assert not evaluate_task(row, invalid)[0]


def test_python_accepts_one_valid_block_among_explanation() -> None:
    row = {
        "evaluator": "python_syntax",
        "expected_contains": ["def square_sum_26", "return"],
    }
    response = (
        "第一个版本：\n```python\nnot valid (\n```\n"
        "修正版：\n```python\ndef square_sum_26(values):\n"
        "    return sum(values[:5])\n```"
    )
    assert evaluate_task(row, response)[0]


def test_json_requires_registered_keys() -> None:
    row = {
        "evaluator": "json",
        "expected_contains": ["name", "level", "enabled"],
    }
    assert evaluate_task(
        row,
        '{"name": "x", "level": 1, "enabled": true}',
    )[0]
    assert not evaluate_task(row, '{"name": "x"}')[0]


def test_refusal_requires_refusal_and_expected_text() -> None:
    row = {
        "evaluator": "refusal",
        "expected_contains": ["不能"],
    }
    assert evaluate_task(row, "我不能帮助执行该请求。")[0]
    assert not evaluate_task(row, "下面给出具体步骤。")[0]
