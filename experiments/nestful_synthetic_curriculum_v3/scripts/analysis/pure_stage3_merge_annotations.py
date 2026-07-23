#!/usr/bin/env python3
"""Merge dual weak-model annotations and flag escalation candidates.

Usage:
  python pure_stage3_merge_annotations.py \\
    --named-dir reports/.../annotation_inputs/named_annotations \\
    --anon-dir reports/.../annotation_inputs/anonymized_annotations \\
    --out reports/.../diagnostic_annotations.jsonl

Each input file should be the model JSON response (one file per task_id.json).
Escalation flags written to escalation_queue.jsonl when:
  - named vs anonymized disagree on root_cause or reward_ordering_correct
  - confidence < 0.75 in either pass
  - root_cause claims evaluator/scorer error
  - cohort is C0_win_E2_loss
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_dir(d: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        obj = json.loads(p.read_text(encoding="utf-8"))
        tid = obj.get("task_id") or p.stem
        out[str(tid)] = obj
    return out


def _escalate(named: dict, anon: dict, cohort: str) -> Optional[str]:
    reasons: List[str] = []
    if named.get("root_cause") != anon.get("root_cause"):
        reasons.append("root_cause_disagreement")
    if named.get("reward_ordering_correct") != anon.get("reward_ordering_correct"):
        reasons.append("reward_ordering_disagreement")
    for ann in (named, anon):
        if float(ann.get("confidence") or 0) < 0.75:
            reasons.append("low_confidence")
            break
    rc = str(named.get("root_cause") or "") + str(anon.get("root_cause") or "")
    if "evaluator" in rc or "scorer" in rc:
        reasons.append("claims_evaluator_error")
    if cohort == "C0_win_E2_loss":
        reasons.append("C0_win_E2_loss_cohort")
    return ";".join(dict.fromkeys(reasons)) if reasons else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--named-dir", type=Path, required=True)
    ap.add_argument("--anon-dir", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True,
                    help="diagnostic_cases.jsonl for cohort lookup")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--escalation-out", type=Path, default=None)
    args = ap.parse_args()

    cohort = {}
    with open(args.cases, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                cohort[r["task_id"]] = r.get("cohort", "")

    named = _load_dir(args.named_dir)
    anon = _load_dir(args.anon_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    esc_path = args.escalation_out or args.out.parent / "escalation_queue.jsonl"
    merged: List[dict] = []
    escalations: List[dict] = []

    for tid in sorted(set(named) | set(anon)):
        n = named.get(tid, {})
        a = anon.get(tid, {})
        row = {
            "task_id": tid,
            "cohort": cohort.get(tid, ""),
            "named": n,
            "anonymized": a,
            "agreement": {
                "root_cause": n.get("root_cause") == a.get("root_cause"),
                "reward_ordering_correct": (
                    n.get("reward_ordering_correct") == a.get("reward_ordering_correct")
                ),
            },
        }
        reason = _escalate(n, a, cohort.get(tid, ""))
        if reason:
            row["escalate"] = True
            row["escalation_reasons"] = reason.split(";")
            escalations.append(row)
        else:
            row["escalate"] = False
        merged.append(row)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(esc_path, "w", encoding="utf-8") as fh:
        for row in escalations:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged={len(merged)} escalate={len(escalations)} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
