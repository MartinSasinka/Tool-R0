"""Joint distribution / OOD and train-300 vs rest-300 audits."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_features import graph_features
from .statistics import (
    benjamini_hochberg,
    jensen_shannon,
    mean,
    normalize_counts,
    odds_ratio_2x2,
    standardized_mean_diff,
    total_variation,
)
from .surface_features import distractor_hardness, track_of_row


def featurize_train_row(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    calls = row.get("gold_calls") or []
    gf = graph_features(calls, sample_id=str(row.get("sample_id")), source=source)
    tools = row.get("tools") or []
    gold_as_tools = [{"name": c.get("name")} for c in calls]
    dh = distractor_hardness(gold_as_tools, tools if isinstance(tools, list) else [])
    prov = row.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}
    return {
        "sample_id": str(row.get("sample_id")),
        "source": source,
        "call_count": gf["n_nodes"],
        "call_bucket": gf["call_bucket"],
        "motif": row.get("motif_type") or gf["motif"],
        "topology_hash": gf["topology_hash"],
        "depth": gf["depth"],
        "reference_density": gf["reference_density"],
        "answer_type": row.get("answer_type") or type(row.get("gold_answer")).__name__,
        "offered_tool_count": len(tools) if isinstance(tools, list) else 0,
        "distractor_count": max(0, (len(tools) if isinstance(tools, list) else 0) - len(calls)),
        "distractor_hardness": dh["mean_distractor_hardness_proxy"],
        "track": prov.get("track") or track_of_row(row),
        "generation_cell": prov.get("generation_cell_id") or "",
        "target_skill": prov.get("target_skill") or "",
        "target_failure_mode": prov.get("target_failure_mode") or "",
        "semantic_program_family": prov.get("semantic_program_family") or "",
        "graph_template_id": prov.get("graph_template_id") or "",
        "paraphrase_status": prov.get("paraphrase_status") or ("paraphrased" if "paraphrase" in str(row.get("source", "")).lower() else "unknown"),
    }


def featurize_diag_row(row: Dict[str, Any], source: str = "diagnostic") -> Dict[str, Any]:
    calls = row.get("output") or []
    gf = graph_features(calls, sample_id=str(row.get("sample_id")), source=source)
    tools = row.get("tools") or []
    gold_as_tools = [{"name": c.get("name")} for c in calls]
    dh = distractor_hardness(gold_as_tools, tools if isinstance(tools, list) else [])
    return {
        "sample_id": str(row.get("sample_id")),
        "source": source,
        "call_count": gf["n_nodes"],
        "call_bucket": gf["call_bucket"],
        "motif": gf["motif"],
        "topology_hash": gf["topology_hash"],
        "depth": gf["depth"],
        "reference_density": gf["reference_density"],
        "answer_type": type(row.get("gold_answer")).__name__,
        "offered_tool_count": len(tools) if isinstance(tools, list) else 0,
        "distractor_count": max(0, (len(tools) if isinstance(tools, list) else 0) - len(calls)),
        "distractor_hardness": dh["mean_distractor_hardness_proxy"],
        "track": "diagnostic",
        "generation_cell": "",
        "target_skill": "",
        "target_failure_mode": "",
        "semantic_program_family": "",
        "graph_template_id": "",
        "paraphrase_status": "na",
    }


def joint_cell_key(feat: Dict[str, Any]) -> str:
    return "|".join([
        str(feat.get("call_bucket")),
        str(feat.get("motif")),
        str(feat.get("topology_hash")),
        str(feat.get("answer_type")),
        str(feat.get("track")),
    ])


def gower_distance(a: Dict[str, Any], b: Dict[str, Any], numeric_keys: Sequence[str], cat_keys: Sequence[str]) -> float:
    parts = []
    for k in numeric_keys:
        av, bv = float(a.get(k) or 0.0), float(b.get(k) or 0.0)
        # assume features roughly in [0, max]; use abs diff capped
        parts.append(min(1.0, abs(av - bv)))
    for k in cat_keys:
        parts.append(0.0 if a.get(k) == b.get(k) else 1.0)
    return sum(parts) / max(1, len(parts))


def ood_analysis(
    train_feats: Sequence[Dict[str, Any]],
    diag_feats: Sequence[Dict[str, Any]],
    outcomes: Dict[str, Dict[str, Any]],
    *,
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    numeric = ["call_count", "depth", "reference_density", "offered_tool_count", "distractor_hardness"]
    cats = ["call_bucket", "motif", "answer_type", "track"]
    train_cells = Counter(joint_cell_key(f) for f in train_feats)
    diag_cells = Counter(joint_cell_key(f) for f in diag_feats)
    joint_rows = []
    for cell, cnt in (train_cells | diag_cells).items():
        joint_rows.append({
            "cell": cell,
            "train_count": train_cells.get(cell, 0),
            "diagnostic_count": diag_cells.get(cell, 0),
            "unseen_in_train": int(train_cells.get(cell, 0) == 0 and diag_cells.get(cell, 0) > 0),
        })

    ood_rows = []
    for df in diag_feats:
        dists = sorted(
            (gower_distance(df, tf, numeric, cats), tf["sample_id"]) for tf in train_feats
        )
        nearest = dists[0][0] if dists else 1.0
        knn = mean([d for d, _ in dists[:k]]) if dists else 1.0
        cell = joint_cell_key(df)
        sid = df["sample_id"]
        oc = outcomes.get(sid, {})
        ood_rows.append({
            "sample_id": sid,
            "joint_cell": cell,
            "train_cell_count": train_cells.get(cell, 0),
            "nearest_train_distance": round(nearest, 4),
            "knn_distance": round(knn or 0.0, 4),
            "ood_score": round(0.6 * nearest + 0.4 * (knn or 0.0), 4),
            "c0_win": int(oc.get("c0_win", 0)),
            "d1_win": int(oc.get("d1_win", 0)),
            "outcome": oc.get("outcome", ""),
            "delta_win": int(oc.get("d1_win", 0)) - int(oc.get("c0_win", 0)),
        })

    # deciles
    scores = sorted(r["ood_score"] for r in ood_rows)
    def decile(x: float) -> int:
        if not scores:
            return 0
        rank = sum(1 for s in scores if s <= x)
        return min(9, max(0, int(10 * (rank - 1) / max(1, len(scores))) ))

    for r in ood_rows:
        r["ood_decile"] = decile(r["ood_score"])

    summary = {
        "n_train": len(train_feats),
        "n_diagnostic": len(diag_feats),
        "n_joint_cells_train": len(train_cells),
        "n_joint_cells_diagnostic": len(diag_cells),
        "unseen_combination_rate": sum(1 for r in ood_rows if r["train_cell_count"] == 0) / max(1, len(ood_rows)),
        "rare_combination_rate": sum(1 for r in ood_rows if r["train_cell_count"] <= 1) / max(1, len(ood_rows)),
        "by_ood_decile": [],
    }
    by_dec = defaultdict(list)
    for r in ood_rows:
        by_dec[r["ood_decile"]].append(r)
    for d in range(10):
        rows = by_dec.get(d, [])
        n = len(rows)
        summary["by_ood_decile"].append({
            "decile": d,
            "n": n,
            "c0_win_rate": sum(r["c0_win"] for r in rows) / n if n else None,
            "d1_win_rate": sum(r["d1_win"] for r in rows) / n if n else None,
            "net_gain": sum(1 for r in rows if r["outcome"] == "loss_to_win") - sum(1 for r in rows if r["outcome"] == "win_to_loss"),
        })
    return joint_rows, ood_rows, summary


def feature_associations(
    diag_feats: Sequence[Dict[str, Any]],
    outcomes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Univariate associations for gain vs not-gain; BH-adjusted."""
    rows = []
    for f in diag_feats:
        oc = outcomes.get(f["sample_id"], {})
        rows.append({**f, **oc})
    tests = []
    # categorical: motif, call_bucket
    for key in ("motif", "call_bucket", "answer_type"):
        levels = sorted({str(r.get(key)) for r in rows})
        for level in levels:
            a = sum(1 for r in rows if str(r.get(key)) == level and r.get("outcome") == "loss_to_win")
            b = sum(1 for r in rows if str(r.get(key)) == level and r.get("outcome") != "loss_to_win")
            c = sum(1 for r in rows if str(r.get(key)) != level and r.get("outcome") == "loss_to_win")
            d = sum(1 for r in rows if str(r.get(key)) != level and r.get("outcome") != "loss_to_win")
            # Fisher-ish via OR only; p from chi-square approx
            orstats = odds_ratio_2x2(a, b, c, d)
            n = a + b + c + d
            # simple chi-square p approx
            if n == 0:
                continue
            row1, row2 = a + b, c + d
            col1, col2 = a + c, b + d
            exp = lambda rr, cc: rr * cc / n
            chi = 0.0
            for obs, e in ((a, exp(row1, col1)), (b, exp(row1, col2)), (c, exp(row2, col1)), (d, exp(row2, col2))):
                if e > 0:
                    chi += (obs - e) ** 2 / e
            # chi1 p approx via survival; rough
            p = math.exp(-0.5 * chi)  # very rough; marked as approximate
            tests.append((f"gain~{key}={level}", p, {
                "feature": key,
                "level": level,
                "a_gain_in": a,
                "b_nongain_in": b,
                "c_gain_out": c,
                "d_nongain_out": d,
                "odds_ratio": orstats["odds_ratio"],
                "n": n,
                "effect": "gain_association",
            }))
    bh = {x["name"]: x["q_bh"] for x in benjamini_hochberg([(n, p) for n, p, _ in tests])}
    out = []
    for name, p, meta in tests:
        out.append({
            "test": name,
            "p_raw_approx": p,
            "q_bh": bh.get(name),
            **meta,
            "note": "p-values are approximate chi-square proxies; interpret with effect size and n",
        })
    return out


def compare_subset_distributions(
    first: Sequence[Dict[str, Any]],
    rest: Sequence[Dict[str, Any]],
    cat_keys: Sequence[str],
    num_keys: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    summary_cats = {}
    for key in cat_keys:
        c1 = Counter(str(r.get(key) or "") for r in first)
        c2 = Counter(str(r.get(key) or "") for r in rest)
        p1, p2 = normalize_counts(c1), normalize_counts(c2)
        tv = total_variation(p1, p2)
        js = jensen_shannon(p1, p2)
        summary_cats[key] = {"tv": tv, "jsd": js, "n_levels_first": len(c1), "n_levels_rest": len(c2)}
        all_levels = sorted(set(c1) | set(c2))
        for lev in all_levels:
            rows.append({
                "feature": key,
                "level": lev,
                "count_first300": c1.get(lev, 0),
                "count_rest300": c2.get(lev, 0),
                "share_first300": p1.get(lev, 0.0),
                "share_rest300": p2.get(lev, 0.0),
                "share_diff": p1.get(lev, 0.0) - p2.get(lev, 0.0),
                "tv_feature": tv,
                "jsd_feature": js,
            })
    num_summary = {}
    for key in num_keys:
        a = [float(r.get(key) or 0.0) for r in first]
        b = [float(r.get(key) or 0.0) for r in rest]
        smd = standardized_mean_diff(a, b)
        num_summary[key] = {
            "mean_first300": mean(a),
            "mean_rest300": mean(b),
            "smd_first_minus_rest": smd,
        }
        rows.append({
            "feature": key,
            "level": "__numeric__",
            "count_first300": len(a),
            "count_rest300": len(b),
            "share_first300": mean(a),
            "share_rest300": mean(b),
            "share_diff": (mean(a) or 0) - (mean(b) or 0),
            "tv_feature": None,
            "jsd_feature": None,
            "smd": smd,
        })

    # concentration / missing cells
    cells_first = Counter(str(r.get("generation_cell") or "") for r in first)
    cells_rest = Counter(str(r.get("generation_cell") or "") for r in rest)
    missing_in_first = sorted([k for k in cells_rest if k and cells_first.get(k, 0) == 0])
    over_first = cells_first.most_common(10)

    # shuffle heuristic: monotonic blocks of generation_cell
    cell_seq = [str(r.get("generation_cell") or "") for r in first]
    switches = sum(1 for i in range(1, len(cell_seq)) if cell_seq[i] != cell_seq[i - 1])
    unique = len([c for c in cells_first if c])
    # if few switches relative to unique, likely blocked not shuffled
    blocked_score = 1.0 - (switches / max(1, len(cell_seq) - 1))

    summary = {
        "categorical": summary_cats,
        "numeric": num_summary,
        "missing_generation_cells_in_first300": missing_in_first[:50],
        "n_missing_cells_in_first300": len(missing_in_first),
        "overrepresented_cells_first300": [{"cell": k, "count": v} for k, v in over_first],
        "cell_sequence_switches_first300": switches,
        "blocked_sequence_score": blocked_score,
        "shuffle_interpretation": (
            "likely_contiguous_cell_blocks"
            if blocked_score > 0.7 and unique > 3
            else "likely_interleaved_or_shuffled"
        ),
    }
    return rows, summary
