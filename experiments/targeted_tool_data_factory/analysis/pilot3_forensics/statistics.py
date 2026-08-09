"""Deterministic statistics for Pilot3 forensics."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def shannon_entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def effective_n(counter: Counter) -> float:
    h = shannon_entropy(counter)
    return float(2 ** h) if h > 0 else (1.0 if sum(counter.values()) else 0.0)


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def jensen_shannon(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av > 0 and bv > 0:
                s += av * math.log2(av / bv)
            elif av > 0 and bv == 0:
                return float("inf")
        return s

    kl_pm = _kl(p, m)
    kl_qm = _kl(q, m)
    if math.isinf(kl_pm) or math.isinf(kl_qm):
        # smooth
        eps = 1e-12
        keys2 = set(p) | set(q)
        p2 = {k: p.get(k, 0.0) + eps for k in keys2}
        q2 = {k: q.get(k, 0.0) + eps for k in keys2}
        sp = sum(p2.values())
        sq = sum(q2.values())
        p2 = {k: v / sp for k, v in p2.items()}
        q2 = {k: v / sq for k, v in q2.items()}
        m2 = {k: 0.5 * (p2[k] + q2[k]) for k in keys2}
        return 0.5 * (_kl(p2, m2) + _kl(q2, m2))
    return 0.5 * (kl_pm + kl_qm)


def normalize_counts(counter: Counter) -> Dict[str, float]:
    n = sum(counter.values()) or 1
    return {str(k): v / n for k, v in counter.items()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar via Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def binom_cdf(k_max: int, n_: int) -> float:
        total = 0.0
        c_i = 1.0
        for i in range(0, k_max + 1):
            if i > 0:
                c_i *= (n_ - i + 1) / i
            total += c_i
        return total / (2 ** n_)

    return min(1.0, 2 * binom_cdf(k, n))


def paired_bootstrap_delta(
    wins_a: Sequence[bool],
    wins_b: Sequence[bool],
    *,
    n_boot: int = 20000,
    seed: int = 42,
    strata: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Paired bootstrap of 100*(mean_b - mean_a) in percentage points.

    If ``strata`` is provided, resampling is done within each stratum
    (stratified paired bootstrap for diagnostic macro).
    """
    n = len(wins_a)
    if n == 0 or n != len(wins_b):
        raise ValueError("wins_a/wins_b length mismatch or empty")
    rng = random.Random(seed)
    point = 100.0 * (sum(wins_b) / n - sum(wins_a) / n)
    deltas: List[float] = []

    if strata is None:
        idx = list(range(n))
        for _ in range(n_boot):
            sample = [rng.choice(idx) for _ in range(n)]
            wa = sum(wins_a[i] for i in sample) / n
            wb = sum(wins_b[i] for i in sample) / n
            deltas.append(100.0 * (wb - wa))
    else:
        if len(strata) != n:
            raise ValueError("strata length mismatch")
        by: Dict[str, List[int]] = {}
        for i, s in enumerate(strata):
            by.setdefault(str(s), []).append(i)
        for _ in range(n_boot):
            sample: List[int] = []
            for group in by.values():
                sample.extend(rng.choice(group) for _ in range(len(group)))
            wa = sum(wins_a[i] for i in sample) / n
            wb = sum(wins_b[i] for i in sample) / n
            deltas.append(100.0 * (wb - wa))

    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "point_pp": point,
        "ci95_lo_pp": lo,
        "ci95_hi_pp": hi,
        "n_boot": float(n_boot),
        "seed": float(seed),
    }


def benjamini_hochberg(pvals: Sequence[Tuple[str, float]]) -> List[Dict[str, float]]:
    """Return BH-adjusted q-values for named p-values."""
    items = [(name, p) for name, p in pvals if p is not None and not math.isnan(p)]
    m = len(items)
    if m == 0:
        return []
    ranked = sorted(items, key=lambda x: x[1])
    q: Dict[str, float] = {}
    prev = 1.0
    for i in range(m, 0, -1):
        name, p = ranked[i - 1]
        adj = min(prev, p * m / i)
        q[name] = adj
        prev = adj
    return [{"name": name, "p_raw": p, "q_bh": q[name]} for name, p in items]


def odds_ratio_2x2(a: int, b: int, c: int, d: int, haldane: float = 0.5) -> Dict[str, float]:
    """OR for [[a,b],[c,d]] with Haldane-Anscombe correction if any zero."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + haldane, b + haldane, c + haldane, d + haldane
    or_ = (a / b) / (c / d) if b and c and d else float("nan")
    # Woolf SE
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(or_) - 1.96 * se) if or_ > 0 else float("nan")
    hi = math.exp(math.log(or_) + 1.96 * se) if or_ > 0 else float("nan")
    return {"odds_ratio": or_, "ci95_lo": lo, "ci95_hi": hi, "n": a + b + c + d}


def call_bucket(n: int) -> str:
    if n >= 6:
        return "6+"
    return str(n)


def standardized_mean_diff(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    ma, mb = mean(list(a)), mean(list(b))
    if ma is None or mb is None:
        return None
    va = sum((x - ma) ** 2 for x in a) / max(1, len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / max(1, len(b) - 1)
    pooled = math.sqrt((va + vb) / 2) if (va + vb) > 0 else 0.0
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def counter_top_share(counter: Counter, k: int) -> float:
    n = sum(counter.values()) or 1
    return sum(c for _, c in counter.most_common(k)) / n
