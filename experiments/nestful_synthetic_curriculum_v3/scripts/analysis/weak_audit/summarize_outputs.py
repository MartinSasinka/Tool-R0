"""Extended agreement metrics and summarize outputs."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from weak_audit.agreement import compare_passes
from weak_audit.io_utils import read_jsonl, write_json, write_jsonl
from weak_audit.summary import select_high_priority, write_cluster_csv


def _suffix_tag(suffix: str) -> str:
    return f"_{suffix}" if suffix else ""


def _provider_index(out_dir: Path) -> Dict[tuple, str]:
    idx: Dict[tuple, str] = {}
    for label in ("A", "B"):
        for src in (
            out_dir / f"pass_{label.lower()}_annotations_raw.jsonl",
            out_dir / "retry_invalid_raw.jsonl",
        ):
            if not src.is_file():
                continue
            for row in read_jsonl(src):
                if row.get("task_id"):
                    idx[(row["task_id"], label)] = row.get("provider") or "unknown"
    return idx


def extended_agreement(
    ann_a: Dict[str, dict],
    ann_b: Dict[str, dict],
    pkt_map: Dict[str, dict],
    provider_idx: Dict[tuple, str],
) -> dict:
    base = compare_passes(ann_a, ann_b)
    ids = sorted(set(ann_a) & set(ann_b))
    same_prov: List[bool] = []
    diff_prov: List[bool] = []
    cohort_agree: Dict[str, List[bool]] = defaultdict(list)
    conf_bucket: Dict[str, List[bool]] = defaultdict(list)
    fields = list(base["per_field_agreement"].keys())

    for tid in ids:
        a, b = ann_a[tid], ann_b[tid]
        exact = all(a.get(f) == b.get(f) for f in fields)
        pa = provider_idx.get((tid, "A"), "unknown")
        pb = provider_idx.get((tid, "B"), "unknown")
        (same_prov if pa == pb else diff_prov).append(exact)
        min_conf = min(float(a.get("confidence") or 0), float(b.get("confidence") or 0))
        conf_bucket["low" if min_conf < 0.75 else "high"].append(exact)
        for c in pkt_map.get(tid, {}).get("cohorts") or ["unknown"]:
            cohort_agree[c].append(exact)

    def _rate(vals: List[bool]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    base.update({
        "provider_same_exact_rate": _rate(same_prov),
        "provider_different_exact_rate": _rate(diff_prov),
        "provider_same_n": len(same_prov),
        "provider_different_n": len(diff_prov),
        "cohort_exact_agreement": {c: _rate(v) for c, v in cohort_agree.items()},
        "confidence_bucket_exact_agreement": {b: _rate(v) for b, v in conf_bucket.items()},
    })
    return base


def write_summarize_outputs(out_dir: Path, *, suffix: str = "") -> dict:
    tag = _suffix_tag(suffix)
    packets = read_jsonl(out_dir / "case_packets.jsonl")
    pkt_map = {p["task_id"]: p for p in packets}
    ann_a = {r["task_id"]: r for r in read_jsonl(out_dir / f"pass_a_annotations{tag}.jsonl")}
    ann_b = {r["task_id"]: r for r in read_jsonl(out_dir / f"pass_b_annotations{tag}.jsonl")}
    agree = extended_agreement(ann_a, ann_b, pkt_map, _provider_index(out_dir))

    fields = list(agree["per_field_agreement"].keys())
    rows = []
    for tid in sorted(set(ann_a) & set(ann_b)):
        a, b = ann_a[tid], ann_b[tid]
        rows.append({
            "task_id": tid,
            "exact_match": all(a.get(f) == b.get(f) for f in fields),
            **{f"{f}_a": a.get(f) for f in fields},
            **{f"{f}_b": b.get(f) for f in fields},
            "confidence_diff": abs(float(a.get("confidence") or 0) - float(b.get("confidence") or 0)),
        })
    csv_path = out_dir / f"annotation_agreement{tag}.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md_lines = [
        f"# Annotation agreement (Pass A vs Pass B){tag.replace('_', ' ').title()}",
        "",
        "Weak-model annotator stability only — not ground truth.",
        "",
        f"- Tasks compared: {agree['n_tasks']}",
        f"- Exact agreement rate: {agree.get('exact_agreement_rate')}",
        f"- Root cause Cohen kappa: {agree.get('root_cause_kappa')}",
        f"- Root cause changed rate: {agree.get('root_cause_changed_rate')}",
        f"- Reward ordering changed rate: {agree.get('reward_ordering_changed_rate')}",
        f"- Provider-matched exact rate: {agree.get('provider_same_exact_rate')}",
        f"- Provider-different exact rate: {agree.get('provider_different_exact_rate')}",
        "",
        "## Per-field agreement",
        "",
    ]
    for f, stats in (agree.get("per_field_agreement") or {}).items():
        md_lines.append(f"- {f}: {stats.get('agreement_rate')}")
    (out_dir / (f"ANNOTATION_AGREEMENT_{suffix.upper()}.md" if suffix else "ANNOTATION_AGREEMENT.md")).write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    write_cluster_csv(out_dir / f"cluster_counts{tag}.csv", ann_a, pkt_map)
    clusters = defaultdict(list)
    for tid, a in ann_a.items():
        clusters[a.get("root_cause", "unclear")].append(tid)
    examples = {
        rc: {
            "representative_task_ids": tids[:3],
            "high_confidence": sorted(tids, key=lambda t: -(ann_a[t].get("confidence") or 0))[:3],
            "disagreement_pass_b": [t for t in tids if ann_b.get(t, {}).get("root_cause") != rc][:3],
        }
        for rc, tids in clusters.items()
    }
    write_json(out_dir / f"cluster_examples{tag}.json", examples)

    rc_ctr = Counter(a.get("root_cause") for a in ann_a.values())
    summary = [
        f"# Weak model summary{tag.replace('_', ' ')}",
        "",
        f"- Pass A: {len(ann_a)}, Pass B: {len(ann_b)}, both: {agree['n_tasks']}",
        "",
        "## Root causes (Pass A)",
        "",
    ]
    for rc, n in rc_ctr.most_common(10):
        summary.append(f"- {rc}: {n}")
    summary += [
        "",
        "## Limitations",
        "",
        "- Weak annotations are hypotheses, not ground truth.",
        "- first_divergence_turn is relatively more stable than root_cause.",
    ]
    (out_dir / (f"WEAK_MODEL_SUMMARY_{suffix.upper()}.md" if suffix else "WEAK_MODEL_SUMMARY.md")).write_text(
        "\n".join(summary), encoding="utf-8"
    )

    hp = select_high_priority(list(pkt_map.values()), ann_a, ann_b, agree, max_n=80)
    hp_json = out_dir / (f"HIGH_PRIORITY_CASES_{suffix.upper()}.jsonl" if suffix else "HIGH_PRIORITY_CASES.jsonl")
    write_jsonl(hp_json, hp)
    hp_md = out_dir / (f"HIGH_PRIORITY_CASES_{suffix.upper()}.md" if suffix else "HIGH_PRIORITY_CASES.md")
    hp_md_lines = [f"# High-priority cases{tag}", "", f"**Count:** {len(hp)}", ""]
    for row in hp[:30]:
        hp_md_lines.append(f"- {row['task_id']} (score={row['priority_score']})")
    hp_md.write_text("\n".join(hp_md_lines), encoding="utf-8")

    return {
        "agreement": agree,
        "n_ann_a": len(ann_a),
        "n_ann_b": len(ann_b),
        "n_both": agree["n_tasks"],
        "high_priority_count": len(hp),
    }
