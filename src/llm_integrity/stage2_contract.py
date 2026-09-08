"""Pure Stage2 definitions; no GPU imports and no implicit statistical tuning."""
import hashlib
import numpy as np

BLOCK_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROXIES = ("raw_logit_l2", "centered_logit_l2", "probability_l2", "js", "topk_change", "top1_flip")


def balanced_block_schedule(block_types, layers, rounds, prompt_id):
    if not block_types or not layers or rounds < 1:
        raise ValueError("Empty or invalid coverage schedule")
    if len(set(block_types)) != len(block_types) or len(set(layers)) != len(layers):
        raise ValueError("Duplicate blocks/layers")
    offset = int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16) % len(layers)
    # Successive passes shift the layer for EVERY module. Full cycle covers Cartesian product.
    return [(block_types[r % len(block_types)],
             layers[(r % len(block_types) + r // len(block_types) + offset) % len(layers)])
            for r in range(rounds)]


def internal_metrics(intact, attack, top_k=10):
    a, b = np.asarray(intact, dtype=np.float64), np.asarray(attack, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected paired finite vocabulary logits")
    if not 1 <= top_k <= len(a):
        raise ValueError("Invalid Top-K")
    def logsoftmax(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    la, lb = logsoftmax(a), logsoftmax(b)
    p, q = np.exp(la), np.exp(lb)
    lm = np.logaddexp(la, lb) - np.log(2.)
    # Stable sorting gives deterministic token-id tie breaking.
    ta, tb = set(np.argsort(-a, kind="stable")[:top_k]), set(np.argsort(-b, kind="stable")[:top_k])
    return {"raw_logit_l2": float(np.linalg.norm(b-a)),
            "centered_logit_l2": float(np.linalg.norm((b-b.mean())-(a-a.mean()))),
            "probability_l2": float(np.linalg.norm(q-p)),
            "js": float(max(0., .5*(p@(la-lm) + q@(lb-lm)))),
            "kl_intact_to_attack": float(max(0., p@(la-lb))),
            "topk_change": float(1-len(ta & tb)/top_k),
            "top1_flip": int(np.argmax(a) != np.argmax(b))}
