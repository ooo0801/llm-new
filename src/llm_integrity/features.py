from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]|\d+|[^\w\s]", re.UNICODE)


def surface_features(text: str) -> np.ndarray:
    tokens = _TOKEN_PATTERN.findall(text)
    token_count = len(tokens)
    unique = len(set(tokens))
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    sentences = max(1, len(re.findall(r"[。！？.!?]+", text)))
    lines = max(1, len(text.splitlines()))
    digits = len(re.findall(r"\d", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    code_marks = len(re.findall(r"```|\bdef\b|\bclass\b|[{};]", text))
    return np.asarray(
        [
            len(text),
            token_count,
            unique / max(token_count, 1),
            repeated / max(token_count, 1),
            sentences,
            token_count / sentences,
            lines,
            digits / max(len(text), 1),
            latin / max(len(text), 1),
            chinese / max(len(text), 1),
            code_marks,
        ],
        dtype=np.float64,
    )


def hashed_semantic_features(text: str, dimension: int = 128) -> np.ndarray:
    """Dependency-free semantic-ish fallback using signed token hashing."""
    result = np.zeros(dimension, dtype=np.float64)
    tokens = _TOKEN_PATTERN.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        index = number % dimension
        sign = 1.0 if number & 1 else -1.0
        result[index] += sign
    norm = np.linalg.norm(result)
    return result / norm if norm > 0 else result


def task_features(text: str, row: dict[str, Any] | None) -> np.ndarray:
    row = row or {}
    expected = row.get("expected_answer")
    contains = row.get("expected_contains") or []
    if isinstance(contains, str):
        contains = [contains]
    normalized = re.sub(r"\s+", "", text).lower()
    exact = 0.0
    answer_contained = 0.0
    if expected is not None:
        expected_norm = re.sub(r"\s+", "", str(expected)).lower()
        exact = float(normalized == expected_norm)
        answer_contained = float(expected_norm in normalized)
    contains_rate = (
        sum(str(value).lower() in text.lower() for value in contains) / len(contains) if contains else 0.0
    )
    valid_json = 0.0
    if str(row.get("category", "")).startswith("structured"):
        try:
            import json

            json.loads(text)
            valid_json = 1.0
        except Exception:
            pass
    refusal = float(bool(re.search(r"不能|无法|抱歉|不可以|can't|cannot|sorry", text, re.I)))
    return np.asarray([exact, answer_contained, contains_rate, valid_json, refusal], dtype=np.float64)


@dataclass
class FeatureExtractor:
    semantic_model_name: str | None = None
    semantic_model_revision: str | None = None
    semantic_device: str = "cpu"
    semantic_local_files_only: bool = False
    hashed_dimension: int = 128

    def __post_init__(self) -> None:
        self._encoder = None

    def _semantic(self, texts: list[str]) -> np.ndarray:
        if self.semantic_model_name:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    self.semantic_model_name,
                    revision=self.semantic_model_revision,
                    device=self.semantic_device,
                    local_files_only=self.semantic_local_files_only,
                )
            return np.asarray(self._encoder.encode(texts, normalize_embeddings=True), dtype=np.float64)
        return np.stack([hashed_semantic_features(text, self.hashed_dimension) for text in texts])

    def transform(
        self,
        texts: Iterable[str],
        rows: Iterable[dict[str, Any] | None] | None = None,
        include_surface: bool = True,
        include_semantic: bool = True,
        include_task: bool = True,
    ) -> np.ndarray:
        text_list = list(texts)
        row_list = list(rows) if rows is not None else [None] * len(text_list)
        if len(row_list) != len(text_list):
            raise ValueError("texts and rows must have equal length")
        blocks = []
        if include_surface:
            blocks.append(np.stack([surface_features(text) for text in text_list]))
        if include_semantic:
            blocks.append(self._semantic(text_list))
        if include_task:
            blocks.append(np.stack([task_features(text, row) for text, row in zip(text_list, row_list)]))
        if not blocks:
            raise ValueError("At least one feature family must be enabled")
        return np.concatenate(blocks, axis=1)


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("Standardizer is not fitted")
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
