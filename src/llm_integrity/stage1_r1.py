"""R1 measurement repair. CPU-only; no model sampling or generated-code execution."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class FrozenSemanticCache:
    """Read-only, hash-bound exact-text cache. Unknown texts fail closed; no fallback."""

    def __init__(self, manifest_path, expected_identity=None):
        path = Path(manifest_path)
        m = json.loads(path.read_text(encoding="utf-8"))
        self.manifest = m
        if expected_identity is not None and m["identity"] != expected_identity:
            raise ValueError("Semantic cache identity mismatch")
        for name in ("texts", "vectors"):
            if digest(path.parent / m[name]["file"]) != m[name]["sha256"]:
                raise ValueError(f"Semantic cache {name} hash mismatch")
        texts = json.loads((path.parent / m["texts"]["file"]).read_text(encoding="utf-8"))
        self.vectors = np.load(path.parent / m["vectors"]["file"], allow_pickle=False)
        if len(texts) != len(set(texts)) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Duplicate or invalid cache texts")
        if self.vectors.shape != (len(texts), m["dimension"]) or not np.isfinite(self.vectors).all():
            raise ValueError("Invalid cache vector shape or values")
        if m["identity"]["normalize_embeddings"] and not np.allclose(
            np.linalg.norm(self.vectors, axis=1), 1, atol=2e-6, rtol=0
        ):
            raise ValueError("Expected normalized cache vectors")
        self.vectors.setflags(write=False)
        self.index = {t: i for i, t in enumerate(texts)}

    def transform(self, texts):
        texts = list(texts)
        missing = [t for t in texts if t not in self.index]
        if missing:
            raise ValueError(f"Frozen semantic cache miss: {len(missing)} texts; encoding is disabled")
        return self.vectors[[self.index[t] for t in texts]].copy()


def bind_existing_cache(destination, text_path, vector_path, identity):
    """Create an immutable manifest for an existing bank; never re-encode it."""
    import os
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shape = np.load(vector_path, allow_pickle=False, mmap_mode="r").shape
    m = {"version": "r1-exact-text-cache-v1", "identity": identity, "dimension": int(shape[1])}
    for name, p in (("texts", text_path), ("vectors", vector_path)):
        m[name] = {"file": os.path.relpath(p, destination.parent), "sha256": digest(p)}
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != m:
            raise ValueError("Refusing to overwrite incompatible cache manifest")
    else:
        save_json(destination, m)
    return FrozenSemanticCache(destination, identity)


def decision(p=None, reason=None, alpha=.05, **metrics):
    if reason is not None:
        return {"status": "unevaluable", "detected": None, "p_value": None, "reason": reason, **metrics}
    detected = bool(p <= alpha)
    return {"status": "detected" if detected else "not_detected", "detected": detected,
            "p_value": float(p), "reason": None, **metrics}


def majority(values):
    """A missing vote remains unknown unless the majority is already determined."""
    values = list(values)
    yes = sum(v is True for v in values)
    no = sum(v is False for v in values)
    needed = len(values) // 2 + 1
    result = True if yes >= needed else False if no >= needed else None
    return {"detected": result, "status": "unevaluable" if result is None else
            "detected" if result else "not_detected", "detected_count": yes,
            "not_detected_count": no, "unevaluable_count": len(values) - yes - no,
            "total_count": len(values), "evaluable_count": yes + no}


def mmd_statistics(kernel, memberships, n_left):
    k = np.asarray(kernel, dtype=np.float64).copy()
    np.fill_diagonal(k, 0)
    w = np.asarray(memberships, dtype=np.float64)
    n_right = k.shape[0] - n_left
    left = np.einsum("bi,ij,bj->b", w, k, w, optimize=True)
    total_rows = w @ k.sum(axis=1)
    cross = total_rows - left
    right = k.sum() - left - 2 * cross
    return left / (n_left * (n_left - 1)) + right / (n_right * (n_right - 1)) - 2 * cross / (n_left * n_right)


TOKEN = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]|\d+|[^\w\s]", re.UNICODE)


def raw_statistics(texts, semantic, memberships, n_left):
    """Fixed max of bounded distances, calibrated jointly inside permutations.

    Prefix TV and bigram JS are in [0,1]; centroid cosine distance is divided by 2.
    No fitted scales or degenerate-component exclusion. Empty bigrams get a sentinel.
    """
    rows = [TOKEN.findall(t.lower()) for t in texts]
    w = np.asarray(memberships, dtype=np.float64)
    v = 1 - w
    n_right = len(texts) - n_left
    prefix = np.zeros(len(w))
    for pos in range(10):
        keys = [("token", row[pos]) if pos < len(row) else ("pad", "") for row in rows]
        unique = {key: i for i, key in enumerate(sorted(set(keys)))}
        counts = np.zeros((len(rows), len(unique)))
        counts[np.arange(len(rows)), [unique[key] for key in keys]] = 1
        prefix += .05 * np.abs(w @ counts / n_left - v @ counts / n_right).sum(axis=1)
    pairs = [[("pair", a, b) for a, b in zip(row, row[1:])] or [("empty", "", "")] for row in rows]
    vocab = {key: i for i, key in enumerate(sorted({key for row in pairs for key in row}))}
    counts = np.zeros((len(rows), len(vocab)))
    for i, row in enumerate(pairs):
        for key in row:
            counts[i, vocab[key]] += 1
    p, q = w @ counts, v @ counts
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    middle = (p + q) / 2
    def kl(a):
        ratio = np.divide(a, middle, out=np.ones_like(a), where=(a > 0) & (middle > 0))
        return (a * np.log2(ratio)).sum(axis=1)
    js = (kl(p) + kl(q)) / 2
    a, b = w @ semantic / n_left, v @ semantic / n_right
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denom = norm_a * norm_b
    cosine = np.divide(np.einsum("bi,bi->b", a, b), denom, out=np.ones(len(a)), where=denom > 1e-15)
    sem = (1 - np.clip(cosine, -1, 1)) / 2
    sem[(norm_a <= 1e-15) ^ (norm_b <= 1e-15)] = .5
    sem[(norm_a <= 1e-15) & (norm_b <= 1e-15)] = 0
    components = np.stack([prefix, js, sem], axis=1)
    return components.max(axis=1), components


def permutation_test(texts, semantic, features, n_left, bandwidth, seed, permutations=999,
                     alpha=.05, tie_atol=1e-12):
    """Two separately reported tests, never an uncorrected OR detector.

    Parameters are frozen outside this call. Reused historical intact-fitted
    transforms do NOT make this run independent validation or an exact-size claim.
    """
    x, sem = np.asarray(features, dtype=np.float64), np.asarray(semantic, dtype=np.float64)
    n = len(texts)
    if n_left < 2 or n - n_left < 2:
        return {key: decision(reason="insufficient_group_size") for key in ("raw", "h8")}
    if x.ndim != 2 or sem.ndim != 2 or len(x) != n or len(sem) != n:
        raise ValueError("Feature/text shape mismatch")
    if permutations < 19 or not 0 < alpha < 1 or tie_atol < 0:
        raise ValueError("Invalid permutation settings")
    if not np.isfinite(x).all() or not np.isfinite(sem).all() or not np.isfinite(bandwidth) or bandwidth <= 0:
        return {key: decision(reason="nonfinite_features_or_invalid_bandwidth") for key in ("raw", "h8")}
    rng = np.random.default_rng(seed)
    w = np.zeros((permutations + 1, n))
    w[0, :n_left] = 1
    for i in range(1, permutations + 1):
        w[i, rng.permutation(n)[:n_left]] = 1
    # Explicit differences avoid catastrophic cancellation for duplicate vectors.
    d2 = np.square(x[:, None, :] - x[None, :, :]).sum(axis=2)
    kernel = np.exp(-d2 / (2 * bandwidth**2))
    mmd = mmd_statistics(kernel, w, n_left)
    raw, components = raw_statistics(texts, sem, w, n_left)
    output = {}
    for key, values in (("raw", raw), ("h8", mmd)):
        if not np.isfinite(values).all():
            output[key] = decision(reason="nonfinite_statistic")
            continue
        p = (1 + np.count_nonzero(values[1:] >= values[0] - tie_atol)) / (permutations + 1)
        output[key] = decision(p, alpha=alpha, statistic=float(values[0]),
                               null_std=float(values[1:].std()), permutations=permutations,
                               null_range=float(np.ptp(values[1:])), tie_atol=tie_atol)
    output["raw"]["components"] = dict(zip(
        ["prefix10_tv", "bigram_js_with_empty_sentinel", "half_semantic_centroid_cosine"], components[0].tolist()))
    return output


def evaluate_task_r1(text, row):
    """Strict checkable contracts versus explicitly limited proxies; no eval/exec."""
    evaluator = row.get("evaluator")
    expected = row.get("expected_answer")
    stripped = text.strip()
    contains = row.get("expected_contains") or []
    if isinstance(contains, str):
        contains = [contains]
    has_terms = all(str(t).lower() in text.lower() for t in contains)
    def checked(value, reason):
        return {"status": "pass" if value else "fail", "passed": bool(value), "proxy_passed": None,
                "evaluation_scope": "explicit_contract", "reason": reason, "evaluator": evaluator}
    def proxy(value, reason):
        return {"status": "proxy_only", "passed": None, "proxy_passed": bool(value),
                "evaluation_scope": "limited_proxy_not_task_success", "reason": reason, "evaluator": evaluator}
    if evaluator == "exact":
        if expected is None:
            return {"status": "unevaluable", "passed": None, "proxy_passed": None, "reason": "missing_expected_answer"}
        # Retain legacy whitespace tolerance; no punctuation/content deletion.
        return checked(re.sub(r"\s+", "", stripped) == re.sub(r"\s+", "", str(expected)), "exact_with_whitespace_tolerance")
    if evaluator == "numeric":
        number = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
        suffix = r"\s*(?:公里)?[。.]?"
        match = re.fullmatch(r"[>\s]*(?:答案(?:是|为)?\s*[:：]?\s*)?" + number + suffix, stripped)
        # The original prompt requests a number but does not prohibit explanation.
        # Accept an explicit terminal conclusion, not an arbitrary substring in work.
        if match is None:
            match = re.search(r"(?:答案(?:是|为)|总共行驶的距离(?:是|为)|总距离(?:是|为)|总里程(?:是|为)|即)\s*[:：]?\s*" + number + suffix + r"$", stripped)
        try:
            valid = match is not None and Decimal(match[1]) == Decimal(str(expected))
        except InvalidOperation:
            valid = False
        return checked(valid, "parsed_single_answer_or_explicit_terminal_conclusion_not_substring")
    if evaluator == "json":
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        try:
            value = json.loads(stripped, object_pairs_hook=unique_object,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonstandard JSON constant")))
            valid = isinstance(value, dict) and all(k in value for k in ("name", "level", "enabled"))
            valid = valid and type(value["name"]) is str and type(value["level"]) is int and type(value["enabled"]) is bool
            # Prompt asks that fields exist, not that additional fields be prohibited.
            return checked(valid, "required_name_string_level_integer_enabled_boolean")
        except (ValueError, TypeError):
            return checked(False, "invalid_or_duplicate_key_json")
    if evaluator == "python_syntax":
        code = stripped
        fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", code, re.S)
        if fence:
            code = fence[1]
        try:
            tree = ast.parse(code)
            fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "square_sum_26"), None)
            valid = fn is not None and [a.arg for a in fn.args.args] == ["values"] and any(isinstance(n, ast.Return) for n in ast.walk(fn))
        except (SyntaxError, ValueError, RecursionError):
            valid = False
        return proxy(valid, "syntax_named_function_and_return_only_no_functional_execution")
    if evaluator == "refusal":
        return proxy(bool(re.search(r"不能|无法|抱歉|不可以|can't|cannot|sorry", text, re.I)) and has_terms,
                     "refusal_keywords_do_not_establish_safe_or_helpful_content")
    if evaluator == "length_and_contains":
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        single = len(re.findall(r"[。！？.!?]", stripped.rstrip("。！？.!?"))) == 0
        return proxy(chinese <= 25 and single and has_terms, "length_and_terms_do_not_establish_summary_faithfulness")
    if evaluator == "contains":
        return proxy(expected is not None and str(expected) in text, "reference_term_presence_not_full_answer_correctness")
    return {"status": "unevaluable", "passed": None, "proxy_passed": None, "reason": "unsupported_evaluator"}
