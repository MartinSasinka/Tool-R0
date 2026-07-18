#!/usr/bin/env python3
"""Build a canonical Stage 3 v5 agentic training dataset from worker shards.

Merges accepted rows from multiple RunPod worker trees under
``data/agentic_workers/`` (and optional pilot dirs), deduplicates by
question/trace hash, replays every gold trace through the real v5 synthetic
executor, renumbers sample_ids, and writes a manifest.

Excluded by default: ``stage3_win1`` (win=1 gate), legacy v4 root ``gpu*``
shards, empty/broken runs.

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_stage3_training_dataset.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))

from paths import sha256_file  # noqa: E402
from lib.agentic_data.distribution import corpus_stats  # noqa: E402
from lib.agentic_data.exec_bridge import (  # noqa: E402
    REGISTRY_VERSION, registry_hash, replay_task,
)
from lib.agentic_data.schema import STAGE_FILES  # noqa: E402

# Import merge helpers from sibling script.
sys.path.insert(0, HERE)
from merge_agentic_workers_v5 import (  # noqa: E402
    apply_merge_tier_quotas, merge_stage, write_merge_report, _write_jsonl,
)

STAGE = "stage3_3call_agentic_openrouter"
DEFAULT_OUT = os.path.join(V3_ROOT, "data", "curriculum_v5_stage3_training")

# v5 production runs only (not win1, not legacy v4 gpu0..3 at agentic_workers/gpu*).
DEFAULT_WORKER_GLOBS = [
    os.path.join(V3_ROOT, "data", "agentic_workers",
                 "agentic_v5_stage3_full", "agentic_v5_stage3_full", "gpu*"),
    os.path.join(V3_ROOT, "data", "agentic_workers",
                 "agentic_v5_stage3_full_second_run(1)",
                 "agentic_v5_stage3_full_second_run", "gpu*"),
    os.path.join(V3_ROOT, "data", "agentic_workers",
                 "agentic_v5_stage3_full_third_run",
                 "agentic_v5_stage3_full_third_run", "gpu*"),
    os.path.join(V3_ROOT, "data", "agentic_v5_stage3_pilot_v2",
                 "agentic_v5_stage3_pilot_v2", "gpu*"),
    os.path.join(V3_ROOT, "data", "agentic_workers",
                 "agentic_v5_stage3_loose", "agentic_v5_stage3_loose", "gpu*"),
]

_GPU_DIR_RE = re.compile(r"^gpu\d+$")


def collect_worker_dirs(globs: List[str]) -> List[str]:
    out: List[str] = []
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            if not os.path.isdir(path):
                continue
            if not _GPU_DIR_RE.match(os.path.basename(path.rstrip("/\\"))):
                continue
            out.append(os.path.abspath(path))
    return sorted(set(out))


def _row_tier(row: Dict[str, Any]) -> str:
    q = row.get("quality") or {}
    rs = row.get("rollout_signal") or {}
    tier = q.get("quality_tier") or rs.get("quality_tier")
    if tier:
        return tier
    fsr = rs.get("full_success_rate") or 0
    if fsr >= 0.999:
        return "easy_anchor"
    if fsr > 0:
        return "frontier"
    return "partial_frontier"


def replay_filter(rows: List[Dict[str, Any]]
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    kept: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    for row in rows:
        ok, detail = replay_task(row)
        if ok:
            kept.append(row)
        else:
            failed.append({
                "sample_id": row.get("sample_id", "?"),
                "detail": str(detail)[:200],
            })
    return kept, failed


def write_manifest(out_root: str, *, n_rows: int, merge_report: Dict[str, Any],
                   replay_failed: List[Dict[str, str]],
                   tier_counts: Dict[str, int]) -> str:
    filtered_path = os.path.join(out_root, "filtered", STAGE_FILES[STAGE])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "curriculum_v5_stage3_training",
        "stage": STAGE,
        "executor_mode": "synthetic",
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "rows": n_rows,
        "training_candidate": True,
        "source_runs": DEFAULT_WORKER_GLOBS,
        "merge": merge_report,
        "replay_failures_dropped": len(replay_failed),
        "replay_failure_samples": replay_failed[:20],
        "tier_counts": tier_counts,
        "files": {
            STAGE_FILES[STAGE]: {
                "path": os.path.abspath(filtered_path),
                "rows": n_rows,
                "sha256": sha256_file(filtered_path),
            },
        },
    }
    man_dir = os.path.join(out_root, "manifests")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, "curriculum_v5_stage3_training_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return man_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers-glob", action="append", default=None,
                    help="extra worker-dir glob (repeatable)")
    ap.add_argument("--apply-merge-tier-quotas", action="store_true")
    ap.add_argument("--skip-replay-filter", action="store_true")
    args = ap.parse_args()

    globs = list(DEFAULT_WORKER_GLOBS)
    if args.workers_glob:
        globs.extend(args.workers_glob)
    workers = collect_worker_dirs(globs)
    if not workers:
        print("[stage3-train] ERROR: no worker dirs found", file=sys.stderr)
        return 2

    raw_loaded = 0
    for w in workers:
        path = os.path.join(w, "filtered", STAGE_FILES[STAGE])
        if os.path.isfile(path):
            with open(path, encoding="utf-8-sig") as fh:
                raw_loaded += sum(1 for line in fh if line.strip())

    print(f"[stage3-train] merging {len(workers)} worker shard(s) "
          f"({raw_loaded} raw rows)")
    for w in workers:
        print(f"  - {w}")

    merged, merge_report = merge_stage(STAGE, workers)
    print(f"[stage3-train] merge: {merge_report['total_loaded']} loaded -> "
          f"{merge_report['total_kept']} after dedup "
          f"(-{merge_report['total_dropped_as_duplicate']} dup)")

    replay_failed: List[Dict[str, str]] = []
    if not args.skip_replay_filter:
        before = len(merged)
        merged, replay_failed = replay_filter(merged)
        print(f"[stage3-train] replay: {before} -> {len(merged)} "
              f"(-{len(replay_failed)} gold replay failures)")

    if args.apply_merge_tier_quotas and merged:
        from lib.agentic_data.quality import tier_quotas_for_merge
        mf, mp, me = tier_quotas_for_merge()
        before = len(merged)
        merged = apply_merge_tier_quotas(merged, min_frontier=mf,
                                         max_partial=mp, max_easy=me)
        merge_report["tier_quota_applied"] = True
        merge_report["rows_before_tier_quota"] = before
        merge_report["rows_after_tier_quota"] = len(merged)
        print(f"[stage3-train] tier quota: {before} -> {len(merged)}")

    # Renumber after replay/tier trim (merge_stage already numbered; redo for final).
    for i, row in enumerate(merged):
        row["sample_id"] = f"agentic_v5_stage3_{i + 1:06d}"

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(out_root, "filtered"), exist_ok=True)
    out_path = os.path.join(out_root, "filtered", STAGE_FILES[STAGE])
    _write_jsonl(out_path, merged)

    tier_counts = dict(Counter(_row_tier(r) for r in merged))
    stats = corpus_stats(merged) if merged else {}
    man_path = write_manifest(out_root, n_rows=len(merged),
                              merge_report=merge_report,
                              replay_failed=replay_failed,
                              tier_counts=tier_counts)
    write_merge_report(out_root, [merge_report], {STAGE: merged})

    summary_path = os.path.join(out_root, "reports", "STAGE3_TRAINING_DATASET.md")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("# Stage 3 v5 training dataset\n\n")
        fh.write(f"- **Rows:** {len(merged)}\n")
        fh.write(f"- **Registry:** {REGISTRY_VERSION} `{registry_hash()[:16]}…`\n")
        fh.write(f"- **Output:** `{out_path}`\n")
        fh.write(f"- **Manifest:** `{man_path}`\n\n")
        fh.write("## Tier mix\n\n")
        for k, v in sorted(tier_counts.items()):
            pct = 100.0 * v / len(merged) if merged else 0
            fh.write(f"- {k}: {v} ({pct:.1f}%)\n")
        fh.write("\n## Source workers\n\n")
        for w in workers:
            fh.write(f"- `{w}`\n")
        if replay_failed:
            fh.write(f"\n## Replay failures dropped ({len(replay_failed)})\n\n")
            for rf in replay_failed[:10]:
                fh.write(f"- `{rf['sample_id']}`: {rf['detail']}\n")
        if stats:
            fh.write("\n## Diversity\n\n")
            fh.write(f"- motif dominance: {stats.get('dominance', {}).get('motif')}\n")
            fh.write(f"- tool_family dominance: "
                     f"{stats.get('dominance', {}).get('tool_family')}\n")

    print(f"[stage3-train] wrote {len(merged)} rows -> {out_path}")
    print(f"[stage3-train] manifest -> {man_path}")
    print(f"[stage3-train] summary -> {summary_path}")
    return 0 if merged else 1


if __name__ == "__main__":
    sys.exit(main())
