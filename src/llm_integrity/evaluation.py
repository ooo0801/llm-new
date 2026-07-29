from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def binary_metrics(labels: Iterable[int | bool], predictions: Iterable[int | bool]) -> BinaryMetrics:
    y = np.asarray(list(labels), dtype=bool)
    p = np.asarray(list(predictions), dtype=bool)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and predictions must be non-empty and equal length")
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    tn = int(np.sum(~y & ~p))
    fn = int(np.sum(y & ~p))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return BinaryMetrics(
        accuracy=(tp + tn) / len(y),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-12),
        false_positive_rate=fp / max(fp + tn, 1),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )
