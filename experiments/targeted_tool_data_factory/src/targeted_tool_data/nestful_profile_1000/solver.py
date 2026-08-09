"""CP-SAT selection for NESTFUL_PROFILE_1000."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

from . import CALL_HARD, N_TRAIN, SURFACE_DESIGN


def _tv(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _shares(counts: Mapping[str, int], n: int) -> Dict[str, float]:
    n = max(1, n)
    return {k: v / n for k, v in counts.items()}


def solve(candidates: Sequence[Mapping[str, Any]],
          quotas: Mapping[str, Any],
          *,
          time_limit_s: float = 120.0,
          seed: int = 20260809) -> Dict[str, Any]:
    """Select exactly N_TRAIN tasks with hard call-count quotas.

    Soft objective: weighted L1 deviations from TargetProfile conditionals,
    plus concentration penalties.
    """
    model = cp_model.CpModel()
    n = len(candidates)
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]

    # Hard: exact total
    model.Add(sum(x) == N_TRAIN)

    # Hard: call-count
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_bucket[c["call_bucket"]].append(i)
    for bucket, need in CALL_HARD.items():
        idxs = by_bucket.get(bucket, [])
        if len(idxs) < need:
            return {
                "status": "infeasible",
                "reason": f"call_bucket {bucket}: available {len(idxs)} < need {need}",
                "selected_indices": [],
                "deficits": {bucket: need - len(idxs)},
            }
        model.Add(sum(x[i] for i in idxs) == need)

    # Index helpers
    def group(key: str) -> Dict[Any, List[int]]:
        g: Dict[Any, List[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            g[c[key]].append(i)
        return g

    # Soft: conditional quotas — for each (bucket, label) match hamilton count
    # Using abs-deviation integer vars scaled into objective.
    obj_terms: List[cp_model.LinearExpr] = []
    WEIGHTS = {
        "answer_type": 10,
        "query_mode": 10,
        "tool_band": 8,
        "depth_bucket": 8,
        "motif": 8,
        "join_bucket": 7,
        "ref_band": 6,
        "schema_complexity": 5,
    }
    QUOTA_KEYS = {
        "answer_type": "P(answer_type|call_count)",
        "query_mode": "P(query_mode|call_count)",
        "tool_band": "P(offered_tool_count|call_count)",
        "depth_bucket": "P(depth|call_count)",
        "motif": "P(motif|call_count)",
        "join_bucket": "P(join_count|call_count)",
        "ref_band": "P(reference_density|call_count)",
        "schema_complexity": "P(schema_complexity|call_count)",
    }

    for feat, qname in QUOTA_KEYS.items():
        w = WEIGHTS[feat]
        qmap = quotas.get(qname) or {}
        for bucket, want_dist in qmap.items():
            idxs_b = by_bucket.get(bucket, [])
            if not idxs_b:
                continue
            # group by feature within bucket
            by_lab: Dict[str, List[int]] = defaultdict(list)
            for i in idxs_b:
                by_lab[str(candidates[i][feat])].append(i)
            labels = sorted(set(want_dist) | set(by_lab))
            for lab in labels:
                target = int(want_dist.get(lab, 0))
                vars_lab = by_lab.get(lab, [])
                count = sum(x[i] for i in vars_lab) if vars_lab else 0
                # abs(count - target)
                diff = model.NewIntVar(-N_TRAIN, N_TRAIN, f"d_{feat}_{bucket}_{lab}")
                model.Add(diff == count - target)
                absd = model.NewIntVar(0, N_TRAIN, f"ad_{feat}_{bucket}_{lab}")
                model.AddAbsEquality(absd, diff)
                obj_terms.append(w * absd)

    # Soft surface design 70/30
    by_surf = group("surface_track")
    for surf, share in SURFACE_DESIGN.items():
        target = int(round(share * N_TRAIN))
        vars_s = by_surf.get(surf, [])
        count = sum(x[i] for i in vars_s) if vars_s else 0
        diff = model.NewIntVar(-N_TRAIN, N_TRAIN, f"d_surf_{surf}")
        model.Add(diff == count - target)
        absd = model.NewIntVar(0, N_TRAIN, f"ad_surf_{surf}")
        model.AddAbsEquality(absd, diff)
        obj_terms.append(3 * absd)

    # Soft: discourage over-concentration of workflows / templates / sequences
    # Cap: workflow top share soft via per-workflow max 50 (5%)
    by_wf = group("workflow_id")
    for wf, idxs in by_wf.items():
        if len(idxs) <= 1:
            continue
        # max 50 selected from one workflow
        model.Add(sum(x[i] for i in idxs) <= 50)

    by_intent = group("intent_fingerprint")
    for intent, idxs in by_intent.items():
        if not intent or len(idxs) <= 1:
            continue
        model.Add(sum(x[i] for i in idxs) <= 10)  # skeleton concentration

    by_seq = group("primitive_sequence")
    for seq, idxs in by_seq.items():
        if not seq or len(idxs) <= 1:
            continue
        model.Add(sum(x[i] for i in idxs) <= 30)  # <=3%

    # Soft: prefer tasks that already have full export (cheaper / already audited)
    for i, c in enumerate(candidates):
        if not c.get("has_full_export"):
            obj_terms.append(x[i])  # tiny penalty weight 1

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    status_name = {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "feasible",
        cp_model.INFEASIBLE: "infeasible",
        cp_model.MODEL_INVALID: "model_invalid",
        cp_model.UNKNOWN: "unknown",
    }.get(status, str(status))

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "reason": "solver did not find a feasible selection",
            "selected_indices": [],
            "objective": None,
        }

    selected = [i for i in range(n) if solver.Value(x[i]) == 1]
    return {
        "status": status_name,
        "objective": solver.ObjectiveValue(),
        "selected_indices": selected,
        "wall_time_s": solver.WallTime(),
        "n_selected": len(selected),
    }


def achieved_distributions(selected: Sequence[Mapping[str, Any]],
                           quotas: Mapping[str, Any]) -> Dict[str, Any]:
    n = len(selected)
    call = Counter(c["call_bucket"] for c in selected)

    def cond(feat: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for bucket in CALL_HARD:
            rows = [c for c in selected if c["call_bucket"] == bucket]
            cnt = Counter(str(c[feat]) for c in rows)
            denom = max(1, len(rows))
            out[bucket] = {k: round(v / denom, 5) for k, v in sorted(cnt.items())}
        return out

    def tv_cond(feat: str, qname: str) -> Dict[str, Any]:
        target = quotas.get(qname) or {}
        achieved_counts: Dict[str, Counter] = {}
        tvs = {}
        max_dev = {}
        for bucket, need in CALL_HARD.items():
            rows = [c for c in selected if c["call_bucket"] == bucket]
            cnt = Counter(str(c[feat]) for c in rows)
            achieved_counts[bucket] = cnt
            tgt = target.get(bucket) or {}
            # shares
            p = {k: v / need for k, v in tgt.items()}
            q = {k: cnt.get(k, 0) / max(1, need) for k in set(p) | set(cnt)}
            tvs[bucket] = round(_tv(p, q), 5)
            max_dev[bucket] = max(
                (abs(cnt.get(k, 0) - int(tgt.get(k, 0))) for k in set(p) | set(cnt)),
                default=0)
        return {"tv_by_bucket": tvs, "max_abs_count_dev": max_dev,
                "mean_tv": round(sum(tvs.values()) / max(1, len(tvs)), 5)}

    return {
        "n": n,
        "call_count": dict(call),
        "call_count_exact": dict(call) == dict(CALL_HARD),
        "P(answer_type|call_count)": cond("answer_type"),
        "P(query_mode|call_count)": cond("query_mode"),
        "P(offered_tool_count|call_count)": cond("tool_band"),
        "P(depth|call_count)": cond("depth_bucket"),
        "P(join_count|call_count)": cond("join_bucket"),
        "P(reference_density|call_count)": cond("ref_band"),
        "P(motif|call_count)": cond("motif"),
        "distances": {
            "answer_type": tv_cond("answer_type", "P(answer_type|call_count)"),
            "query_mode": tv_cond("query_mode", "P(query_mode|call_count)"),
            "offered_tools": tv_cond("tool_band", "P(offered_tool_count|call_count)"),
            "depth": tv_cond("depth_bucket", "P(depth|call_count)"),
            "motif": tv_cond("motif", "P(motif|call_count)"),
            "join_count": tv_cond("join_bucket", "P(join_count|call_count)"),
            "reference_density": tv_cond("ref_band", "P(reference_density|call_count)"),
        },
        "surface": dict(Counter(c["surface_track"] for c in selected)),
        "six_plus_lengths": dict(Counter(
            c["call_count"] for c in selected if c["call_bucket"] == "6+")),
    }
