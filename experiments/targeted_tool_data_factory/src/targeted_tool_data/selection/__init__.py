"""Selection: hard gates → greedy deficit matching → coverage; distribution
metrics (JSD / Wasserstein / classifier two-sample AUC); leakage-free splits.
No opaque weighted score (D12) — every decision is traced.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..profile import featurize_row


# ── distribution metrics ──────────────────────────────────────────────────
def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    a = a / (a.sum() or 1)
    b = b / (b.sum() or 1)
    m = (a + b) / 2

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return round(0.5 * _kl(a, m) + 0.5 * _kl(b, m), 6)


def _dist_of(vals: List[str]) -> Dict[str, float]:
    c = Counter(vals)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def wasserstein(a: List[float], b: List[float]) -> float:
    try:
        from scipy.stats import wasserstein_distance
        return round(float(wasserstein_distance(a, b)), 6)
    except ImportError:                                # pragma: no cover
        aa, bb = np.sort(a), np.sort(b)
        n = max(len(aa), len(bb))
        qs = np.linspace(0, 1, n)
        return round(float(np.mean(np.abs(
            np.quantile(aa, qs) - np.quantile(bb, qs)))), 6)


def _feature_matrix(feats: List[Dict[str, Any]], motifs: List[str]) -> np.ndarray:
    rows = []
    for f in feats:
        row = [f["call_count"], f["depth"], f["ref_share"],
               f["numeric_string_share"], f["n_tools"], f["q_len"] / 100.0]
        row += [1.0 if f["motif"] == m else 0.0 for m in motifs]
        rows.append(row)
    return np.array(rows, dtype=float)


def two_sample_auc(feats_a: List[Dict[str, Any]],
                   feats_b: List[Dict[str, Any]], seed: int = 0) -> float:
    """Cross-validated AUC of a classifier separating A from B.
    0.5 = structurally indistinguishable."""
    motifs = sorted({f["motif"] for f in feats_a + feats_b})
    X = np.vstack([_feature_matrix(feats_a, motifs), _feature_matrix(feats_b, motifs)])
    y = np.array([0] * len(feats_a) + [1] * len(feats_b))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        proba = cross_val_predict(clf, Xs, y, cv=5, method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, proba)), 4)
    except ImportError:                                # pragma: no cover
        return float("nan")


def profile_match_report(feats_set: List[Dict[str, Any]],
                         feats_target: List[Dict[str, Any]],
                         label: str, seed: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label, "n": len(feats_set)}
    for key in ("call_bucket", "motif", "answer_type"):
        out[f"jsd_{key}"] = jsd(_dist_of([str(f[key]) for f in feats_set]),
                                _dist_of([str(f[key]) for f in feats_target]))
    arg_a: Counter = Counter()
    arg_b: Counter = Counter()
    for f in feats_set:
        arg_a.update(f["arg_types"])
    for f in feats_target:
        arg_b.update(f["arg_types"])
    na, nb = sum(arg_a.values()) or 1, sum(arg_b.values()) or 1
    out["jsd_arg_types"] = jsd({k: v / na for k, v in arg_a.items()},
                               {k: v / nb for k, v in arg_b.items()})
    for key in ("n_tools", "q_len", "depth", "ref_share"):
        out[f"wass_{key}"] = wasserstein([float(f[key]) for f in feats_set],
                                         [float(f[key]) for f in feats_target])
    out["auc_two_sample"] = two_sample_auc(feats_set, feats_target, seed=seed)
    return out


# ── greedy deficit-matching selection ─────────────────────────────────────
def select_records(records: List[Dict[str, Any]], cells: List[Dict[str, Any]],
                   n_select: int, seed: int,
                   paraphrase_target: Optional[float] = None
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Hard gates already applied upstream. Fill cell quotas greedily with
    novelty tie-breaks; then fill remainder by cell deficit order.

    `paraphrase_target` (pilot2) mixes LLM-paraphrased and deterministic
    template surfaces to the requested ratio, per cell, so neither surface
    generator dominates the trained distribution."""
    rng = random.Random(seed)
    by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cell[r["generation_cell_id"]].append(r)
    quotas = {c["generation_cell_id"]: c["quota_weight"] * n_select for c in cells}
    fam_used: Counter = Counter()
    tc_used: Counter = Counter()
    tmpl_used: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    chosen_ids = set()

    def _novelty_key(r: Dict[str, Any]) -> Tuple:
        return (fam_used[r["semantic_program_family"]],
                tc_used[r["tool_combination_hash"]],
                tmpl_used[r["template_id"]],
                r["task_id"])

    def _take(r: Dict[str, Any], why: str) -> None:
        selected.append(r)
        chosen_ids.add(r["task_id"])
        fam_used[r["semantic_program_family"]] += 1
        tc_used[r["tool_combination_hash"]] += 1
        tmpl_used[r["template_id"]] += 1
        trace.append({"task_id": r["task_id"], "cell": r["generation_cell_id"],
                      "decision": "select", "why": why})

    def _is_para(r: Dict[str, Any]) -> bool:
        return r.get("query_source") == "openrouter_paraphrase"

    # pass 1: integer quotas per cell
    for cell_id in sorted(quotas, key=lambda c: -quotas[c]):
        want = int(quotas[cell_id])
        pool = sorted(by_cell.get(cell_id, []), key=_novelty_key)
        if paraphrase_target is None:
            take = pool[:want]
        else:
            para = [r for r in pool if _is_para(r)]
            tmpl = [r for r in pool if not _is_para(r)]
            n_para = min(len(para), int(round(want * paraphrase_target)))
            take = para[:n_para] + tmpl[:want - n_para]
            if len(take) < want:                       # one side exhausted
                rest = [r for r in pool if r not in take]
                take += rest[:want - len(take)]
        for r in take:
            if len(selected) >= n_select:
                break
            _take(r, f"cell quota ({want})")
    # pass 2: fill remainder by largest fractional deficit
    while len(selected) < n_select:
        deficits = []
        got = Counter(r["generation_cell_id"] for r in selected)
        for cell_id, q in quotas.items():
            remaining = [r for r in by_cell.get(cell_id, [])
                         if r["task_id"] not in chosen_ids]
            if remaining:
                deficits.append((q - got[cell_id], cell_id, remaining))
        if not deficits:
            break
        deficits.sort(key=lambda x: (-x[0], x[1]))
        _dq, cell_id, remaining = deficits[0]
        if paraphrase_target is not None:
            share = sum(1 for r in selected if _is_para(r)) / max(len(selected), 1)
            want_para = share < paraphrase_target
            side = [r for r in remaining if _is_para(r) == want_para] or remaining
        else:
            side = remaining
        r = sorted(side, key=_novelty_key)[0]
        _take(r, "deficit fill")
    rng.shuffle(selected)
    return selected, trace


# ── leakage-free splits ───────────────────────────────────────────────────
GROUP_KEYS = ["semantic_program_family", "graph_template_id", "tool_combination",
              "paraphrase_family", "argument_skeleton", "value_seed"]


def _groups_of(r: Dict[str, Any]) -> Dict[str, str]:
    return {
        "semantic_program_family": r["semantic_program_family"],
        "graph_template_id": r["graph_template_id"],
        "tool_combination": r["tool_combination_hash"],
        "paraphrase_family": r["paraphrase_family"],
        "argument_skeleton": r["argument_skeleton_hash"],
        "value_seed": str(r["value_seed"]),
    }


def split_records(records: List[Dict[str, Any]], sizes: Dict[str, int],
                  seed: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Union-find over all group keys; whole components assigned greedily."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for r in records:
        g = _groups_of(r)
        anchor = f"task::{r['task_id']}"
        for k, v in g.items():
            union(anchor, f"{k}::{v}")

    comps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        comps[find(f"task::{r['task_id']}")].append(r)

    order = sorted(comps.values(), key=lambda rs: (-len(rs), rs[0]["task_id"]))
    rng = random.Random(seed)
    names = list(sizes.keys())
    filled = {k: [] for k in names}

    for comp in order:
        # assign to the split with the largest relative remaining deficit
        def _deficit(k: str) -> float:
            return (sizes[k] - len(filled[k])) / max(sizes[k], 1)
        best = max(names, key=lambda k: (_deficit(k), -len(filled[k]), k))
        if len(filled[best]) + len(comp) > sizes[best] * 1.15:
            alts = [k for k in names
                    if len(filled[k]) + len(comp) <= sizes[k] * 1.15]
            if alts:
                best = max(alts, key=lambda k: (_deficit(k), k))
        filled[best].extend(comp)

    # trim overshoot deterministically into 'reserve' if present
    for k in names:
        rng.shuffle(filled[k])
    audit = leakage_audit(filled)
    return filled, audit


def leakage_audit(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    seen: Dict[str, Dict[str, str]] = defaultdict(dict)   # key -> value -> split
    collisions: List[Dict[str, str]] = []
    for split_name, rows in splits.items():
        for r in rows:
            for k, v in _groups_of(r).items():
                prev = seen[k].get(v)
                if prev and prev != split_name:
                    collisions.append({"key": k, "value": v,
                                       "splits": f"{prev}|{split_name}"})
                seen[k][v] = split_name
    return {"leakage_collisions": collisions, "leaked": bool(collisions),
            "group_counts": {k: len(v) for k, v in seen.items()},
            "split_sizes": {k: len(v) for k, v in splits.items()}}
