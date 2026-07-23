"""Aggregate annotations and build high-priority handoff."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from weak_audit.io_utils import read_jsonl, write_json


def cluster_counts(
    annotations: Dict[str, dict],
    packets: Dict[str, dict],
) -> List[dict]:
    rows: List[dict] = []
    by_root = Counter(a.get("root_cause", "unclear") for a in annotations.values())
    by_fix = Counter(a.get("recommended_fix", "unclear") for a in annotations.values())
    by_cohort: Dict[str, Counter] = defaultdict(Counter)
    for tid, ann in annotations.items():
        for c in packets.get(tid, {}).get("cohorts") or []:
            by_cohort[c][ann.get("root_cause", "unclear")] += 1
    for root, n in by_root.most_common():
        tids = [t for t, a in annotations.items() if a.get("root_cause") == root]
        rows.append({
            "cluster": root,
            "count": n,
            "recommended_fix_top": by_fix.most_common(1)[0][0] if by_fix else "",
            "representative_task_ids": tids[:3],
        })
    return rows


def write_cluster_csv(path: Path, annotations: Dict[str, dict], packets: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cohort", "root_cause", "recommended_fix", "responsible_reward_component",
        "first_divergence_turn", "count",
    ]
    ctr: Counter = Counter()
    for tid, ann in annotations.items():
        for c in packets.get(tid, {}).get("cohorts") or ["unknown"]:
            key = (
                c,
                ann.get("root_cause"),
                ann.get("recommended_fix"),
                ann.get("responsible_reward_component"),
                ann.get("first_divergence_turn"),
            )
            ctr[key] += 1
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (cohort, rc, rf, rr, fdt), n in ctr.most_common():
            w.writerow({
                "cohort": cohort,
                "root_cause": rc,
                "recommended_fix": rf,
                "responsible_reward_component": rr,
                "first_divergence_turn": fdt,
                "count": n,
            })


def priority_score(
    tid: str,
    packet: dict,
    ann_a: dict,
    ann_b: dict,
    agree: dict,
) -> int:
    score = 0
    cohorts = packet.get("cohorts") or []
    flags = packet.get("deterministic_flags") or {}
    if "c0_win_e2_loss" in cohorts:
        score += 3
    if flags.get("reward_prefers_E2_over_C0"):
        score += 3
    if ann_a.get("root_cause") != ann_b.get("root_cause"):
        score += 2
    if ann_a.get("reward_ordering_correct") != ann_b.get("reward_ordering_correct"):
        score += 2
    if "e2_executable_wrong_other" in cohorts or (
        (packet.get("E2") or {}).get("failure_class") == "executable trajectory ending wrong result"
    ):
        score += 2
    if float(ann_a.get("confidence") or 1) < 0.75 or float(ann_b.get("confidence") or 1) < 0.75:
        score += 1
    if "official_win_reward_too_few" in cohorts:
        score += 1
    if (packet.get("E2") or {}).get("reward_total", 0) >= 0.52 and not (packet.get("E2") or {}).get("official_win"):
        score += 1
    if ann_a.get("root_cause") == "evaluator_or_data_inconsistency" or ann_b.get("root_cause") == "evaluator_or_data_inconsistency":
        score += 2
    return score


def select_high_priority(
    packets: List[dict],
    ann_a: Dict[str, dict],
    ann_b: Dict[str, dict],
    agree: dict,
    *,
    max_n: int = 80,
) -> List[dict]:
    pkt_map = {p["task_id"]: p for p in packets}
    scored = []
    for tid in sorted(set(ann_a) & set(pkt_map)):
        b = ann_b.get(tid, ann_a[tid])
        s = priority_score(tid, pkt_map[tid], ann_a[tid], b, agree)
        if s <= 0:
            continue
        scored.append((s, tid))
    scored.sort(reverse=True)
    out = []
    for s, tid in scored[:max_n]:
        p = pkt_map[tid]
        out.append({
            "task_id": tid,
            "priority_score": s,
            "cohorts": p.get("cohorts"),
            "deterministic_flags": p.get("deterministic_flags"),
            "annotation_pass_a": ann_a.get(tid),
            "annotation_pass_b": ann_b.get(tid),
            "packet": p,
        })
    return out
