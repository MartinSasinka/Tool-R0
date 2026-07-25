from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(rank(xs), rank(ys))


def cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return None
    return dot / (na * nb)


def sign_agreement(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None

    def sgn(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    agree = sum(1 for x, y in zip(a, b) if sgn(x) == sgn(y))
    return agree / len(a)


def ranking_inversion_rate(order_a: Sequence[int], order_b: Sequence[int]) -> Optional[float]:
    """Fraction of pairs ordered differently (Kendall-like)."""
    n = len(order_a)
    if n != len(order_b) or n < 2:
        return None
    inv = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            da = order_a[i] - order_a[j]
            db = order_b[i] - order_b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) != (db > 0):
                inv += 1
    return inv / pairs if pairs else None


def top_bottom_agreement(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[Optional[bool], Optional[bool]]:
    if len(xs) != len(ys) or not xs:
        return None, None
    top_x = max(range(len(xs)), key=lambda i: xs[i])
    top_y = max(range(len(ys)), key=lambda i: ys[i])
    bot_x = min(range(len(xs)), key=lambda i: xs[i])
    bot_y = min(range(len(ys)), key=lambda i: ys[i])
    return top_x == top_y, bot_x == bot_y


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def quartile_indices(n: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    q = max(1, n // 4)
    return [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)]
