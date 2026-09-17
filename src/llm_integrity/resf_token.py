"""RESF-inspired token detector with finite-grid nuisance and Monte Carlo tests.

Not a literal reproduction: no sparse chi-square approximation, no ambiguous tail
power transform, and no claim to cover temperatures between grid points. Every
candidate null is a full post-decoding token distribution. Query samples must be
iid at one fixed member of this family. Alpha covers prompts, rules and looks.
"""
from __future__ import annotations

import math
import numpy as np


def validate_probabilities(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Expected finite nonnegative [temperature, category] probabilities")
    if not np.allclose(q.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Each full category distribution must sum to one")
    return q


def early_window(p_out, alpha, maximum):
    if not 0 <= p_out <= 1 or not 0 < alpha < 1:
        raise ValueError("Invalid probability or alpha")
    if p_out == 0:
        return maximum
    if p_out == 1:
        return 0
    n = min(maximum, max(0, math.floor(math.log1p(-alpha) / math.log1p(-p_out))))
    # Protect against floating-point rounding at the boundary.
    while n and -math.expm1(n * math.log1p(-p_out)) > alpha:
        n -= 1
    return n


def fitted_deviance(counts, q):
    """Minimum multinomial deviance over the frozen nuisance grid."""
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    q = validate_probabilities(q)
    n = counts.sum(axis=1)
    if (counts < 0).any() or (n <= 0).any() or counts.shape[1] != q.shape[1]:
        raise ValueError("Invalid counts")
    empirical = counts / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        saturated = np.where(counts > 0, counts * np.log(empirical), 0).sum(axis=1)
    scores = []
    for row in q:
        impossible = (counts[:, row == 0] > 0).any(axis=1)
        safe_log = np.log(np.where(row > 0, row, 1))
        value = np.maximum(0, 2 * (saturated - counts @ safe_log))
        value[impossible] = np.inf
        scores.append(value)
    scores = np.stack(scores, axis=1)
    return scores.min(axis=1), scores.argmin(axis=1)


def calibration(q, looks=(30, 60, 100), simulations=10000, seed=918):
    """Generate independent simple-null distributions for each frozen look."""
    q = validate_probabilities(q)
    rng = np.random.default_rng(seed)
    result = {}
    for n in looks:
        result[n] = np.stack([np.sort(fitted_deviance(rng.multinomial(n, row, simulations), q)[0])
                              for row in q])
    return result


def detect(tokens, token_ids, q, calibrated, *, panel_size, panel_alpha=.05,
           e_fraction=.2, looks=(30, 60, 100)):
    q = validate_probabilities(q)
    if len(token_ids) != q.shape[1] or len(set(token_ids)) != len(token_ids):
        raise ValueError("Token category mapping mismatch")
    if panel_size < 1 or not 0 < e_fraction < 1 or not 0 < panel_alpha < 1:
        raise ValueError("Invalid alpha allocation")
    if not looks or sorted(set(looks)) != list(looks) or looks[0] < 1:
        raise ValueError("Looks must be positive, unique and ordered")
    if len(tokens) != looks[-1]:
        raise ValueError("Require exactly the frozen maximum number of observations")
    index = {int(token): i for i, token in enumerate(token_ids)}
    active = q.max(axis=0) > 0
    # E uses the UNION of exact post-decoding supports: tail mass is exactly zero
    # for every declared null member. P retains every active token (no tail merge).
    support = {int(token_ids[i]) for i in np.flatnonzero(active)}
    alpha_e = panel_alpha * e_fraction / panel_size
    alpha_p_look = panel_alpha * (1 - e_fraction) / panel_size / len(looks)
    e_window = early_window(0., alpha_e, looks[-1])
    counts = np.zeros(q.shape[1], dtype=np.int64)
    traces = []
    for t, token in enumerate(tokens, 1):
        if int(token) not in support:
            if t <= e_window:
                return dict(alarm=True, rule="E", query=t, traces=traces,
                            alpha_e=alpha_e, alpha_p_per_look=alpha_p_look)
            continue
        counts[index[int(token)]] += 1
        if t in looks:
            statistic, fitted = fitted_deviance(counts, q)
            statistic = float(statistic[0])
            null = calibrated[t]
            b = null.shape[1]
            # Monte Carlo rank p-values, taking the supremum over null members.
            p_values = [(1 + b - np.searchsorted(row, statistic - 1e-12, side="left")) / (b + 1)
                        for row in null]
            p = float(max(p_values))
            traces.append(dict(query=t, statistic=statistic if math.isfinite(statistic) else None,
                               statistic_is_infinite=math.isinf(statistic), p_value=p,
                               grid_index=int(fitted[0]), null_p_values=p_values))
            if p <= alpha_p_look:
                return dict(alarm=True, rule="P", query=t, traces=traces,
                            alpha_e=alpha_e, alpha_p_per_look=alpha_p_look)
    return dict(alarm=False, rule=None, query=len(tokens), traces=traces,
                alpha_e=alpha_e, alpha_p_per_look=alpha_p_look)
