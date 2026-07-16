#!/usr/bin/env python3
"""Merge per-GPU-worker v5 agentic dataset shards into one deduped,
renumbered canonical dataset.

v5 counterpart of ``merge_agentic_workers.py`` — identical merge logic
(cross-worker dedup by question/trace hash, sequential ``sample_id``
renumbering, diversity report), but imports hashing helpers from
``lib.agentic_data.exec_bridge`` (the v5 ``lib/synthetic_tools.py`` registry)
instead of the legacy ``lib.nestful_like_generator``, and renumbers IDs as
``agentic_v5_<short_stage>_NNNNNN``.

Each worker launched by ``launch_multi_gpu_workers_v5.sh`` runs the FULL
orchestrator pipeline independently (best-of-N candidate selection, real
``executor.mode=synthetic`` execution, the GRPO-signal rollout probe, hard
trace + semantic validation, the LLM judge) against its own ``--output-dir``,
so in-corpus dedup / diversity caps are only enforced WITHIN a single
worker. This script closes that gap AFTER all workers finish. It does NOT
re-run any solver/judge/execution gates — every row it merges was ALREADY
accepted by its worker's full orchestrator pipeline.

Usage (repo root):
    python experiments/nestful_synthetic_curriculum_v3/scripts/data/merge_agentic_workers_v5.py \\
        --workers-glob "experiments/nestful_synthetic_curriculum_v3/data/agentic_v5_workers/gpu*" \\
        --output-dir experiments/nestful_synthetic_curriculum_v3/data/curriculum_v5_agentic_synthetic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)

from lib.agentic_data.distribution import corpus_stats  # noqa: E402
from lib.agentic_data.exec_bridge import question_hash, trace_hash  # noqa: E402
from lib.agentic_data.quality import tier_quotas_for_merge  # noqa: E402
from lib.agentic_data.schema import STAGE_FILES, STAGES  # noqa: E402


def _row_tier(row: Dict[str, Any]) -> str:
    return ((row.get("quality") or {}).get("quality_tier")
            or (row.get("rollout_signal") or {}).get("quality_tier")
            or "unknown")


def apply_merge_tier_quotas(rows: List[Dict[str, Any]],
                            *, min_frontier: float,
                            max_partial: float,
                            max_easy: float) -> List[Dict[str, Any]]:
    """Greedy stratified pick toward global tier targets (post-dedup merge)."""
    if not rows:
        return rows
    target_n = len(rows)
    min_frontier_n = int(round(min_frontier * target_n))
    max_partial_n = int(round(max_partial * target_n))
    max_easy_n = int(round(max_easy * target_n))
    buckets = {"frontier": [], "partial_frontier": [], "easy_anchor": [], "other": []}
    for row in rows:
        t = _row_tier(row)
        buckets.get(t, buckets["other"]).append(row)
    picked: List[Dict[str, Any]] = []
    counts = {"frontier": 0, "partial_frontier": 0, "easy_anchor": 0}

    def _take(pool: List[Dict[str, Any]], tier: str, limit: int) -> None:
        while pool and len(picked) < target_n and counts[tier] < limit:
            picked.append(pool.pop(0))
            counts[tier] += 1

    _take(buckets["frontier"], "frontier", target_n)
    while counts["frontier"] < min_frontier_n and buckets["frontier"]:
        picked.append(buckets["frontier"].pop(0))
        counts["frontier"] += 1
    _take(buckets["partial_frontier"], "partial_frontier", max_partial_n)
    _take(buckets["easy_anchor"], "easy_anchor", max_easy_n)
    for pool in (buckets["frontier"], buckets["partial_frontier"],
                 buckets["easy_anchor"], buckets["other"]):
        while pool and len(picked) < target_n:
            picked.append(pool.pop(0))
    return picked[:target_n]


def _short_stage(stage: str) -> str:
    return stage.split("_", 1)[0]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_stage(stage: str, worker_dirs: List[str], *, max_rows: int = None
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge one stage across workers. Returns (merged_rows, report_dict)."""
    stage_file = STAGE_FILES[stage]
    seen_q: set = set()
    seen_t: set = set()
    merged: List[Dict[str, Any]] = []
    per_worker: Dict[str, Dict[str, int]] = {}
    dup_examples: List[Dict[str, str]] = []

    for wdir in worker_dirs:
        path = os.path.join(wdir, "filtered", stage_file)
        rows = _load_jsonl(path)
        kept = 0
        dropped = 0
        for row in rows:
            q = row.get("question") or ""
            gc = row.get("gold_calls")
            if not q or not isinstance(gc, list):
                dropped += 1
                continue
            qh = question_hash(q)
            try:
                th = trace_hash(gc)
            except (KeyError, TypeError):
                th = None
            is_dup = qh in seen_q or (th is not None and th in seen_t)
            if is_dup:
                dropped += 1
                dup_examples.append({
                    "worker": os.path.basename(wdir.rstrip("/\\")),
                    "sample_id": row.get("sample_id", "?"),
                    "question": q[:120],
                })
                continue
            seen_q.add(qh)
            if th is not None:
                seen_t.add(th)
            merged.append(row)
            kept += 1
        per_worker[os.path.basename(wdir.rstrip("/\\"))] = {
            "loaded": len(rows), "kept": kept, "dropped_as_duplicate": dropped,
        }

    if max_rows is not None:
        merged = merged[:max_rows]

    # renumber sample_id sequentially — worker-local ids may collide across GPUs;
    # merge dedups by question/trace hash (not by sample_id). provenance.worker_id
    # / run_id are preserved for audit.
    short = _short_stage(stage)
    for i, row in enumerate(merged):
        row["sample_id"] = f"agentic_v5_{short}_{i + 1:06d}"

    report = {
        "stage": stage,
        "workers": per_worker,
        "total_loaded": sum(w["loaded"] for w in per_worker.values()),
        "total_kept": len(merged),
        "total_dropped_as_duplicate": sum(
            w["dropped_as_duplicate"] for w in per_worker.values()),
        "duplicate_examples_sample": dup_examples[:10],
    }
    return merged, report


def write_merge_report(out_root: str, per_stage_reports: List[Dict[str, Any]],
                       merged_by_stage: Dict[str, List[Dict[str, Any]]]) -> None:
    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    stats_by_stage = {
        stage: corpus_stats(rows) for stage, rows in merged_by_stage.items() if rows
    }

    with open(os.path.join(reports_dir, "MERGE_REPORT_V5.json"), "w",
             encoding="utf-8") as fh:
        json.dump({"per_stage": per_stage_reports,
                   "corpus_stats_by_stage": stats_by_stage}, fh, indent=2,
                  ensure_ascii=False)

    lines = ["# MERGE_REPORT_V5 — multi-GPU v5 agentic worker merge", ""]
    for rep in per_stage_reports:
        stage = rep["stage"]
        lines += [f"## {stage}", "",
                 f"- total rows loaded across workers: {rep['total_loaded']}",
                 f"- kept after cross-worker dedup: {rep['total_kept']}",
                 f"- dropped as cross-worker duplicate: "
                 f"{rep['total_dropped_as_duplicate']}", "",
                 "| worker | loaded | kept | dropped_dup |",
                 "|---|---|---|---|"]
        for wname, w in rep["workers"].items():
            lines.append(f"| {wname} | {w['loaded']} | {w['kept']} | "
                         f"{w['dropped_as_duplicate']} |")
        lines.append("")
        st = stats_by_stage.get(stage)
        if st:
            lines += [
                "**Merged diversity (post-dedup):**", "",
                f"- motif dominance: {st.get('dominance', {}).get('motif')}",
                f"- answer_type dominance: "
                f"{st.get('dominance', {}).get('answer_type')}",
                f"- tool_family dominance: "
                f"{st.get('dominance', {}).get('tool_family')}",
                f"- question_template dominance: "
                f"{st.get('dominance', {}).get('question_template')}",
                "",
            ]
        if rep["duplicate_examples_sample"]:
            lines.append("**Sample of dropped cross-worker duplicates:**")
            lines.append("")
            for d in rep["duplicate_examples_sample"]:
                lines.append(f"- [{d['worker']}] `{d['sample_id']}`: "
                             f"{d['question']}...")
            lines.append("")

    with open(os.path.join(reports_dir, "MERGE_REPORT_V5.md"), "w",
             encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="*", default=None,
                    help="explicit list of worker output dirs")
    ap.add_argument("--workers-glob", default=None,
                    help="glob pattern matching worker output dirs "
                         "(e.g. '.../agentic_v5_workers/gpu*')")
    ap.add_argument("--output-dir", required=True,
                    help="canonical output dir; filtered/*.jsonl + "
                         "reports/MERGE_REPORT_V5.* are written here")
    ap.add_argument("--stages", nargs="*", default=list(STAGES.keys()),
                    choices=list(STAGES.keys()))
    ap.add_argument("--max-rows-per-stage", type=int, default=None,
                    help="optional cap on merged rows per stage (keeps the "
                         "first N in worker order after dedup)")
    ap.add_argument("--apply-merge-tier-quotas", action="store_true",
                    help="stratified trim toward global tier targets "
                         "(TIER_QUOTA_MERGE_*) after dedup")
    args = ap.parse_args()

    worker_dirs: List[str] = []
    if args.workers_glob:
        worker_dirs.extend(sorted(glob.glob(args.workers_glob)))
    if args.workers:
        worker_dirs.extend(args.workers)
    worker_dirs = sorted({os.path.abspath(w) for w in worker_dirs
                          if os.path.isdir(w)})
    if not worker_dirs:
        print("[merge-v5] ERROR: no worker directories found "
              f"(--workers={args.workers} --workers-glob={args.workers_glob})",
              file=sys.stderr)
        return 2

    print(f"[merge-v5] merging {len(worker_dirs)} worker(s):")
    for w in worker_dirs:
        print(f"  - {w}")

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(out_root, "filtered"), exist_ok=True)

    per_stage_reports = []
    merged_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for stage in args.stages:
        merged, report = merge_stage(stage, worker_dirs,
                                     max_rows=args.max_rows_per_stage)
        if args.apply_merge_tier_quotas and merged:
            before = len(merged)
            mf, mp, me = tier_quotas_for_merge()
            merged = apply_merge_tier_quotas(merged, min_frontier=mf,
                                             max_partial=mp, max_easy=me)
            report["tier_quota_applied"] = True
            report["tier_quota_targets"] = {
                "min_frontier": mf, "max_partial": mp, "max_easy": me}
            report["rows_before_tier_quota"] = before
            report["rows_after_tier_quota"] = len(merged)
        merged_by_stage[stage] = merged
        per_stage_reports.append(report)
        out_path = os.path.join(out_root, "filtered", STAGE_FILES[stage])
        _write_jsonl(out_path, merged)
        print(f"[merge-v5] {stage}: {report['total_loaded']} loaded -> "
              f"{report['total_kept']} kept "
              f"(-{report['total_dropped_as_duplicate']} cross-worker dup) "
              f"-> {out_path}")

    write_merge_report(out_root, per_stage_reports, merged_by_stage)
    print(f"[merge-v5] wrote reports/MERGE_REPORT_V5.md and .json under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
