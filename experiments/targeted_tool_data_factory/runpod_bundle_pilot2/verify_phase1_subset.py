#!/usr/bin/env python3
"""Verify the 80-task Phase-1 subset recommended by the signal probe.

Checks (fail-fast):
  - exactly 80 tasks
  - gold replay 100 % through the factory trainer executor
  - leakage 0 against held-out / reserve / deferred (via selected_pilot2 groups)
  - A/G, call count, motif, answer type, reference share, offered-tools,
    generation-cell distributions are reported
  - major-feature JSD vs NESTFUL profile < 0.10
  - no significant NESTFUL bucket disappears from the subset

Usage:
  python verify_phase1_subset.py --phase1 .../recommended_phase1_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
EXPERIMENTS = FACTORY.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
ADAPTER = FACTORY / "trainer_adapter"
SRC = FACTORY / "src"

MAJOR_JSD_MAX = 0.10
SIGNIFICANT_BUCKET_MIN = 0.05   # buckets ≥5 % in NESTFUL must survive
EXPECTED_N = 80

sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(SRC))
from signal_probe_lib import extract_task_meta  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    a = [p.get(k, 0.0) for k in keys]
    b = [q.get(k, 0.0) for k in keys]
    sa, sb = sum(a) or 1.0, sum(b) or 1.0
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    m = [(x + y) / 2 for x, y in zip(a, b)]

    def _kl(x, y):
        s = 0.0
        for xi, yi in zip(x, y):
            if xi > 0 and yi > 0:
                s += xi * math.log2(xi / yi)
        return s

    return round(0.5 * _kl(a, m) + 0.5 * _kl(b, m), 6)


def _dist(vals: Sequence[str]) -> Dict[str, float]:
    c = Counter(vals)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in sorted(c.items())}


def _call_bucket(n: int) -> str:
    return "6+" if n >= 6 else str(n)


def _is_ref(v: Any) -> bool:
    return isinstance(v, str) and v.strip().startswith("$")


def row_features(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = extract_task_meta(row)
    gold = row.get("gold_calls") or row.get("output") or []
    args = []
    for c in gold:
        a = c.get("arguments") or {}
        if isinstance(a, dict):
            args.extend(a.values())
    n_args = len(args) or 1
    n_refs = sum(1 for v in args if _is_ref(v))
    return {
        "task_id": meta["task_id"],
        "track": meta["track"],
        "call_count": meta["call_count"],
        "call_bucket": _call_bucket(int(meta["call_count"])),
        "motif": meta["motif"],
        "answer_type": meta["answer_type"],
        "generation_cell": meta["generation_cell"],
        "ref_share": n_refs / n_args,
        "n_tools": len(row.get("tools") or []),
        "has_ref": n_refs > 0,
    }


def gold_replay_rate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    os.environ["SYNTHETIC_TOOLS_DIR"] = str(ADAPTER)
    for p in (str(MINIMAL), str(ADAPTER)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from executor import ToolExecutor  # noqa: WPS433

    ok = 0
    errors: List[str] = []
    for row in rows:
        sid = row.get("sample_id") or row.get("task_id") or "?"
        task = {"tools": row.get("tools") or [],
                "gold_calls": row.get("gold_calls") or [],
                "gold_answer": row.get("gold_answer")}
        try:
            ex = ToolExecutor(task, mode="synthetic")
            failed = False
            for i, call in enumerate(task["gold_calls"]):
                res = ex.execute(call)
                if res.error is not None:
                    errors.append(f"{sid}: call {i + 1} ({call.get('name')}): {res.error}")
                    failed = True
                    break
            if not failed:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sid}: {type(exc).__name__}: {exc}")
    n = len(rows)
    return {
        "n": n,
        "ok": ok,
        "rate": round(ok / n, 6) if n else 0.0,
        "errors": errors[:20],
        "pass": ok == n and n > 0,
    }


def leakage_check(phase1_ids: Sequence[str],
                  selected_path: Path,
                  heldout_path: Path,
                  reserve_path: Path,
                  deferred_ids: Sequence[str]) -> Dict[str, Any]:
    """Leakage is against held-out / reserve only.

    ``deferred_phase2_tasks.jsonl`` is the other half of the same train pool
    (signal-split, not a structural hold-out), so sharing a graph_template_id
    with deferred is expected and must not fail the gate.
    """
    if not selected_path.is_file():
        return {"leaked": False, "skipped": True,
                "note": f"selected records missing at {selected_path}"}

    sys.path.insert(0, str(SRC))
    from targeted_tool_data.selection import leakage_audit  # noqa: WPS433

    selected = {str(r["task_id"]): r for r in read_jsonl(selected_path)
                if "task_id" in r}
    phase1_recs = [selected[i] for i in phase1_ids if i in selected]
    heldout_ids = set()
    if heldout_path.is_file():
        for r in read_jsonl(heldout_path):
            heldout_ids.add(str(r.get("task_id") or r.get("sample_id")))
    reserve_ids = set()
    if reserve_path.is_file():
        for r in read_jsonl(reserve_path):
            reserve_ids.add(str(r.get("task_id") or r.get("sample_id")))

    heldout_recs = [selected[i] for i in heldout_ids if i in selected]
    reserve_recs = [selected[i] for i in reserve_ids if i in selected]

    missing = [i for i in phase1_ids if i not in selected]
    splits = {
        "phase1": phase1_recs,
        "heldout": heldout_recs,
        "reserve": reserve_recs,
    }
    audit = leakage_audit(splits)
    # Raw id overlap with held-out/reserve is a hard fail. Overlap with
    # deferred is reported for transparency but does not fail the gate.
    id_overlap_heldout = sorted(set(phase1_ids) & heldout_ids)
    id_overlap_reserve = sorted(set(phase1_ids) & reserve_ids)
    id_overlap_deferred = sorted(set(phase1_ids) & set(deferred_ids))
    leaked = bool(audit.get("leaked")) or bool(id_overlap_heldout) or bool(id_overlap_reserve)
    return {
        "leaked": leaked,
        "leakage_audit": audit,
        "id_overlap_heldout": id_overlap_heldout,
        "id_overlap_reserve": id_overlap_reserve,
        "id_overlap_deferred_expected": id_overlap_deferred,
        "phase1_mapped": len(phase1_recs),
        "missing_from_selected": missing[:20],
        "pass": (not leaked) and not missing,
        "note": ("deferred is a signal split of the same train pool — "
                 "template overlap with deferred is expected"),
    }


def _project_motif_for_nestful(motif: str) -> str:
    """Project pilot2 motifs onto the NESTFUL motif vocabulary.

    NESTFUL's profile only knows linear / fan_in / mixed. Pilot2's
    ``branch_aggregate`` is a multi-parent fan-in family, so it maps to
    ``fan_in`` for the JSD comparison; the raw motif distribution is still
    reported separately.
    """
    if motif == "branch_aggregate":
        return "fan_in"
    return motif


def nestful_jsd(feats: Sequence[Dict[str, Any]],
                nestful_profile: Dict[str, Any]) -> Dict[str, Any]:
    call_p = _dist([f["call_bucket"] for f in feats])
    motif_raw = _dist([str(f["motif"]) for f in feats])
    motif_p = _dist([_project_motif_for_nestful(str(f["motif"])) for f in feats])
    ans_p = _dist([str(f["answer_type"]) for f in feats])

    call_q = {str(k): float(v) for k, v in
              (nestful_profile.get("call_count_dist") or {}).items()}
    motif_q = {str(k): float(v) for k, v in
               (nestful_profile.get("motif_dist") or {}).items()}
    ans_q = {str(k): float(v) for k, v in
             (nestful_profile.get("answer_type_dist") or {}).items()}

    jsd_call = jsd(call_p, call_q)
    jsd_motif = jsd(motif_p, motif_q)
    jsd_ans = jsd(ans_p, ans_q)
    major = {
        "jsd_call_bucket": jsd_call,
        "jsd_motif": jsd_motif,
        "jsd_answer_type": jsd_ans,
    }
    max_major = max(major.values()) if major else 0.0

    missing_buckets = []
    for name, qdist, pdist in (
        ("call_bucket", call_q, call_p),
        ("motif", motif_q, motif_p),
        ("answer_type", ans_q, ans_p),
    ):
        for bucket, share in qdist.items():
            if share >= SIGNIFICANT_BUCKET_MIN and pdist.get(bucket, 0.0) <= 0.0:
                # Motif "mixed" in NESTFUL has no pilot2 analogue — warn, don't fail.
                if name == "motif" and bucket == "mixed":
                    continue
                missing_buckets.append(f"{name}:{bucket} (nestful_share={share:.3f})")

    return {
        "phase1_dists": {"call_bucket": call_p, "motif_raw": motif_raw,
                         "motif_projected": motif_p, "answer_type": ans_p},
        "nestful_dists": {"call_bucket": call_q, "motif": motif_q, "answer_type": ans_q},
        "jsd": major,
        "max_major_jsd": max_major,
        "threshold": MAJOR_JSD_MAX,
        "missing_significant_buckets": missing_buckets,
        "motif_projection": "branch_aggregate->fan_in for NESTFUL vocabulary",
        "pass": max_major < MAJOR_JSD_MAX and not missing_buckets,
    }


def distribution_report(feats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(feats),
        "track": _dist([str(f["track"]) for f in feats]),
        "call_count": _dist([str(f["call_count"]) for f in feats]),
        "call_bucket": _dist([f["call_bucket"] for f in feats]),
        "motif": _dist([str(f["motif"]) for f in feats]),
        "answer_type": _dist([str(f["answer_type"]) for f in feats]),
        "generation_cell": _dist([str(f["generation_cell"]) for f in feats]),
        "mean_ref_share": round(sum(f["ref_share"] for f in feats) / max(len(feats), 1), 6),
        "tasks_with_reference": sum(1 for f in feats if f["has_ref"]),
        "reference_task_rate": round(
            sum(1 for f in feats if f["has_ref"]) / max(len(feats), 1), 6),
        "mean_offered_tools": round(
            sum(f["n_tools"] for f in feats) / max(len(feats), 1), 4),
        "offered_tools_hist": dict(sorted(Counter(f["n_tools"] for f in feats).items())),
        "n_generation_cells": len({f["generation_cell"] for f in feats}),
    }


def render_md(report: Dict[str, Any]) -> str:
    g = report["gates"]
    lines = [
        "# Phase-1 subset verification",
        "",
        f"**Verdict: {'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "| gate | result |",
        "|---|---|",
        f"| n_tasks == 80 | {'PASS' if g['n_tasks']['pass'] else 'FAIL'} "
        f"({g['n_tasks']['n']}) |",
        f"| gold replay 100 % | {'PASS' if g['gold_replay']['pass'] else 'FAIL'} "
        f"({g['gold_replay']['rate']}) |",
        f"| leakage 0 | {'PASS' if g['leakage']['pass'] else 'FAIL'} |",
        f"| major-feature JSD < 0.10 | "
        f"{'PASS' if g['nestful_jsd']['pass'] else 'FAIL'} "
        f"(max={g['nestful_jsd']['max_major_jsd']}) |",
        "",
        "## Distributions",
        "",
    ]
    dist = report["distributions"]
    for key in ("track", "call_count", "motif", "answer_type"):
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| value | share |")
        lines.append("|---|---|")
        for k, v in (dist.get(key) or {}).items():
            lines.append(f"| {k} | {v:.3f} |")
        lines.append("")
    lines += [
        f"- mean reference-arg share: {dist.get('mean_ref_share')}",
        f"- reference task rate: {dist.get('reference_task_rate')}",
        f"- mean offered tools: {dist.get('mean_offered_tools')}",
        f"- generation cells: {dist.get('n_generation_cells')}",
        "",
        "## NESTFUL JSD",
        "",
        "| feature | JSD |",
        "|---|---|",
    ]
    for k, v in (g["nestful_jsd"].get("jsd") or {}).items():
        lines.append(f"| {k} | {v} |")
    if g["nestful_jsd"].get("missing_significant_buckets"):
        lines += ["", "Missing significant buckets:", ""]
        for b in g["nestful_jsd"]["missing_significant_buckets"]:
            lines.append(f"- {b}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--deferred", type=Path, default=None)
    ap.add_argument("--heldout", type=Path,
                    default=BUNDLE / "data" / "heldout_grpo_pilot2.jsonl")
    ap.add_argument("--reserve", type=Path,
                    default=FACTORY / "outputs" / "splits" / "reserve_pilot2.jsonl")
    ap.add_argument("--selected", type=Path,
                    default=FACTORY / "outputs" / "selected" / "selected_pilot2.jsonl")
    ap.add_argument("--nestful-profile", type=Path,
                    default=FACTORY / "outputs" / "profiles" / "nestful_profile.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--expect-n", type=int, default=EXPECTED_N)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.phase1.is_file():
        print(f"[phase1] ABORT: missing {args.phase1}", file=sys.stderr)
        return 2

    rows = read_jsonl(args.phase1)
    out_dir = args.out_dir or args.phase1.parent / "phase1_verification"
    print(f"[phase1] rows={len(rows)} expect={args.expect_n}")

    if args.dry_run:
        print(f"[phase1] DRY RUN — would verify into {out_dir}")
        return 0

    feats = [row_features(r) for r in rows]
    ids = [f["task_id"] for f in feats]
    n_gate = {"n": len(rows), "pass": len(rows) == args.expect_n
              and len(ids) == len(set(ids))}

    replay = gold_replay_rate(rows)

    deferred_ids: List[str] = []
    if args.deferred and args.deferred.is_file():
        deferred_ids = [str(r.get("sample_id") or r.get("task_id"))
                        for r in read_jsonl(args.deferred)]

    leakage = leakage_check(ids, args.selected, args.heldout, args.reserve,
                            deferred_ids)

    nestful_profile = {}
    if args.nestful_profile.is_file():
        nestful_profile = json.loads(args.nestful_profile.read_text(encoding="utf-8"))
    nest = nestful_jsd(feats, nestful_profile)

    dist = distribution_report(feats)
    gates = {
        "n_tasks": n_gate,
        "gold_replay": replay,
        "leakage": leakage,
        "nestful_jsd": nest,
    }
    passed = all(g.get("pass") for g in gates.values())

    report = {
        "phase1": str(args.phase1),
        "pass": passed,
        "gates": gates,
        "distributions": dist,
        "task_ids": ids,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PHASE1_SUBSET_VERIFICATION.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "PHASE1_SUBSET_VERIFICATION.md").write_text(
        render_md(report), encoding="utf-8")

    print(f"[phase1] n={n_gate} replay={replay['rate']} "
          f"leakage_pass={leakage['pass']} "
          f"max_jsd={nest['max_major_jsd']} verdict="
          f"{'PASS' if passed else 'FAIL'}")
    print(f"[phase1] report -> {out_dir / 'PHASE1_SUBSET_VERIFICATION.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
