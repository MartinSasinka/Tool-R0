#!/usr/bin/env python3
"""Offline re-filter of agentic worker shards with the NEW multi-turn rollout gate.

Reads ``filtered/*.jsonl`` from each GPU worker (or any input glob), re-runs
``probe_rollout_signal()`` (``run_episode`` + training reward — same as the
fixed generation gate), and reports how many rows would still be accepted.

Does NOT re-run solver-gap, judge, trace validation, etc. — only replaces the
old single-shot ``rollout_signal`` verdict stored at generation time.

Usage (repo root, GPU + local weak solver required):

    export WEAK_SOLVER_BACKEND=local
    export LOCAL_WEAK_MODEL=Qwen/Qwen3-4B-Instruct-2507

    # quick estimate on 30 rows
    python experiments/nestful_synthetic_curriculum_v3/scripts/data/refilter_agentic_rollout_gate.py \\
        --workers-glob "experiments/nestful_synthetic_curriculum_v3/data/agentic_workers/gpu*" \\
        --max-rows 30

    # full worker sweep + deduped export
    python experiments/nestful_synthetic_curriculum_v3/scripts/data/refilter_agentic_rollout_gate.py \\
        --workers-glob "experiments/nestful_synthetic_curriculum_v3/data/agentic_workers/gpu*" \\
        --output-dir experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter/refilter_mt_gate_20260712

Outputs (in --output-dir):
    REFILTER_REPORT.md / REFILTER_REPORT.json
    kept/<stage_file>.jsonl          rows passing the new gate (deduped if --dedup)
    rejected/<stage_file>.jsonl      rows failing only the rollout gate
    checkpoint.jsonl                 resume state (one line per processed row)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)

from lib.agentic_data.rollout_signal import (  # noqa: E402
    ROLLOUT_N, ROLLOUT_TEMPERATURE, load_rollout_config, probe_rollout_signal,
    rollout_mode, summarize_rollouts, target_is_local)
from lib.agentic_data.schema import STAGE_FILES  # noqa: E402
from lib.nestful_like_generator import question_hash, trace_hash  # noqa: E402


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


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_worker_rows(workers: List[str],
                          stages: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Load all worker shards; tag each row with ``_worker`` and ``_source_path``."""
    out: List[Dict[str, Any]] = []
    for wdir in workers:
        worker = os.path.basename(os.path.normpath(wdir))
        for stage, fname in STAGE_FILES.items():
            if stages and stage not in stages:
                continue
            path = os.path.join(wdir, "filtered", fname)
            for row in _load_jsonl(path):
                row = dict(row)
                row["_worker"] = worker
                row["_source_path"] = path
                row["_stage_key"] = stage
                out.append(row)
    return out


def _dedup_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (question_hash(row.get("question", "")), trace_hash(row.get("gold_calls") or []))


def _old_rollout_positive(row: Dict[str, Any]) -> Optional[bool]:
    rs = row.get("rollout_signal") or {}
    if rs.get("skipped"):
        return None
    return bool(rs.get("grpo_signal_positive"))


def _row_probe_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": row["question"],
        "tools": row.get("tools") or [],
        "gold_calls": row.get("gold_calls") or [],
        "gold_observations": row.get("observations") or [],
        "gold_answer": row.get("gold_answer"),
        "stage": row.get("stage") or row.get("_stage_key"),
    }


def _load_checkpoint(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return done
    for row in _load_jsonl(path):
        rid = str(row.get("row_id") or "")
        if rid:
            done[rid] = row
    return done


def _row_id(row: Dict[str, Any], idx: int) -> str:
    return "|".join([
        str(row.get("_worker") or "row"),
        str(row.get("sample_id") or f"idx_{idx}"),
        str(row.get("_source_path") or ""),
    ])


def run_refilter(args) -> int:
    if not target_is_local():
        print("[refilter] ERROR: set WEAK_SOLVER_BACKEND=local", file=sys.stderr)
        return 2

    workers = sorted(
        p for p in (glob.glob(args.workers_glob) if args.workers_glob else list(args.workers or []))
        if os.path.isdir(p))
    if not workers:
        print("[refilter] ERROR: no worker dirs found", file=sys.stderr)
        return 1

    stages_filter = None if args.stages == ["all"] else args.stages
    rows = _discover_worker_rows(workers, stages_filter)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.output_dir or os.path.join(
        V3_ROOT, "data", "curriculum_v4_nestful_like_agentic_openrouter",
        f"refilter_mt_gate_{ts}"))
    ckpt_path = os.path.join(out_dir, "checkpoint.jsonl")
    done = _load_checkpoint(ckpt_path) if args.resume else {}

    print(f"[refilter] workers       : {len(workers)} ({', '.join(os.path.basename(w) for w in workers)})")
    print(f"[refilter] rows to probe : {len(rows)} (max_rows={args.max_rows})")
    print(f"[refilter] rollout mode  : {rollout_mode()}")
    print(f"[refilter] rollouts/row  : {ROLLOUT_N} @ T={ROLLOUT_TEMPERATURE}")
    print(f"[refilter] output dir    : {out_dir}")
    print(f"[refilter] resume        : {args.resume} ({len(done)} already done)")

    if args.dry_run:
        by_stage: Dict[str, int] = defaultdict(int)
        by_worker: Dict[str, int] = defaultdict(int)
        for r in rows:
            by_stage[str(r.get("_stage_key"))] += 1
            by_worker[str(r.get("_worker"))] += 1
        print("[refilter] DRY RUN — per-stage:", dict(by_stage))
        print("[refilter] DRY RUN — per-worker:", dict(by_worker))
        old_pos = sum(1 for r in rows if _old_rollout_positive(r) is True)
        print(f"[refilter] DRY RUN — old gate positive (stored): {old_pos}/{len(rows)}")
        return 0

    # Warm-load config per call-count (stage2=2, stage3=3, ...)
    call_counts = {len(r.get("gold_calls") or []) for r in rows}
    for nc in sorted(call_counts):
        if nc > 0:
            load_rollout_config(nc)
            print(f"[refilter] warmed rollout config for num_calls={nc}")

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        rid = _row_id(row, i)
        if rid in done:
            rec = done[rid]
            results.append(rec)
            continue

        fields = _row_probe_fields(row)
        seed = (args.seed + i) if args.seed is not None else None
        new_signal = probe_rollout_signal(**fields, n=ROLLOUT_N, seed=seed)
        old_pos = _old_rollout_positive(row)
        rec = {
            "row_id": rid,
            "sample_id": row.get("sample_id"),
            "worker": row.get("_worker"),
            "stage": fields["stage"],
            "motif_type": row.get("motif_type"),
            "num_calls": len(fields["gold_calls"]),
            "old_grpo_signal_positive": old_pos,
            "new_grpo_signal_positive": bool(new_signal.get("grpo_signal_positive")),
            "new_rollout_signal": new_signal,
            "new_unique_rewards": new_signal.get("unique_rewards"),
            "new_reward_variance": new_signal.get("reward_variance"),
            "new_reward_mean": new_signal.get("reward_mean"),
        }
        results.append(rec)
        _append_jsonl(ckpt_path, rec)

        if (i + 1) % max(1, args.log_every) == 0:
            elapsed = time.time() - t0
            new_pos = sum(1 for r in results if r["new_grpo_signal_positive"])
            print(f"[refilter] {i + 1}/{len(rows)} probed | new_pass={new_pos} "
                  f"({100 * new_pos / len(results):.1f}%) | {elapsed:.0f}s elapsed",
                  flush=True)

    # Join results back to full rows
    kept_by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejected_by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_dedup: set = set()
    dedup_dropped = 0

    for row, rec in zip(rows, results):
        stage_key = str(row.get("_stage_key") or row.get("stage"))
        fname = STAGE_FILES.get(stage_key, f"{stage_key}.jsonl")
        full = dict(row)
        full["rollout_signal_old"] = full.pop("rollout_signal", None)
        full["rollout_signal"] = rec["new_rollout_signal"]
        full["rollout_refilter"] = {
            "old_grpo_signal_positive": rec["old_grpo_signal_positive"],
            "new_grpo_signal_positive": rec["new_grpo_signal_positive"],
            "refilter_mode": rollout_mode(),
        }
        for k in ("_worker", "_source_path", "_stage_key"):
            full.pop(k, None)

        if args.dedup:
            key = _dedup_key(full)
            if key in seen_dedup:
                dedup_dropped += 1
                continue
            seen_dedup.add(key)

        if rec["new_grpo_signal_positive"]:
            kept_by_stage[fname].append(full)
        else:
            rejected_by_stage[fname].append(full)

    # Aggregate stats
    def _rate(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    n_total = len(results)
    old_pass = sum(1 for r in results if r["old_grpo_signal_positive"] is True)
    new_pass = sum(1 for r in results if r["new_grpo_signal_positive"])
    both_pass = sum(1 for r in results
                     if r["old_grpo_signal_positive"] is True and r["new_grpo_signal_positive"])
    old_only = sum(1 for r in results
                    if r["old_grpo_signal_positive"] is True and not r["new_grpo_signal_positive"])
    new_only = sum(1 for r in results
                    if r["old_grpo_signal_positive"] is not True and r["new_grpo_signal_positive"])

    per_worker: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_stage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        per_worker[r["worker"]]["total"] += 1
        per_stage[r["stage"]]["total"] += 1
        if r["old_grpo_signal_positive"] is True:
            per_worker[r["worker"]]["old_pass"] += 1
            per_stage[r["stage"]]["old_pass"] += 1
        if r["new_grpo_signal_positive"]:
            per_worker[r["worker"]]["new_pass"] += 1
            per_stage[r["stage"]]["new_pass"] += 1

    kept_total = sum(len(v) for v in kept_by_stage.values())
    rejected_total = sum(len(v) for v in rejected_by_stage.values())

    report = {
        "refilter_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workers": [os.path.basename(w) for w in workers],
        "rollout_mode": rollout_mode(),
        "rollout_n": ROLLOUT_N,
        "rollout_temperature": ROLLOUT_TEMPERATURE,
        "rows_probed": n_total,
        "old_gate_pass": old_pass,
        "new_gate_pass": new_pass,
        "old_gate_pass_rate": _rate(old_pass, n_total),
        "new_gate_pass_rate": _rate(new_pass, n_total),
        "both_pass": both_pass,
        "old_only_pass": old_only,
        "new_only_pass": new_only,
        "dedup_enabled": bool(args.dedup),
        "dedup_dropped": dedup_dropped,
        "kept_after_dedup": kept_total,
        "rejected_after_dedup": rejected_total,
        "per_worker": {w: dict(c) for w, c in sorted(per_worker.items())},
        "per_stage": {s: dict(c) for s, c in sorted(per_stage.items())},
        "kept_per_stage_file": {k: len(v) for k, v in kept_by_stage.items()},
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(out_dir, exist_ok=True)
    for fname, kept in kept_by_stage.items():
        _write_jsonl(os.path.join(out_dir, "kept", fname), kept)
    for fname, rej in rejected_by_stage.items():
        _write_jsonl(os.path.join(out_dir, "rejected", fname), rej)

    report_path = os.path.join(out_dir, "REFILTER_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    md_path = os.path.join(out_dir, "REFILTER_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Agentic rollout-gate refilter report\n\n")
        fh.write(f"- **Rows probed:** {n_total}\n")
        fh.write(f"- **Old gate pass (stored):** {old_pass} ({100 * old_pass / max(n_total, 1):.1f}%)\n")
        fh.write(f"- **New multi-turn gate pass:** {new_pass} ({100 * new_pass / max(n_total, 1):.1f}%)\n")
        fh.write(f"- **Both pass:** {both_pass}\n")
        fh.write(f"- **Old-only (would drop):** {old_only}\n")
        fh.write(f"- **New-only (would gain):** {new_only}\n")
        if args.dedup:
            fh.write(f"- **Dedup dropped:** {dedup_dropped}\n")
            fh.write(f"- **Kept after dedup:** {kept_total}\n")
            fh.write(f"- **Rejected after dedup:** {rejected_total}\n")
        fh.write(f"- **Rollout mode:** {rollout_mode()} | N={ROLLOUT_N} T={ROLLOUT_TEMPERATURE}\n")
        fh.write(f"- **Elapsed:** {report['elapsed_seconds']}s\n\n")
        fh.write("## Per worker\n\n| worker | total | old_pass | new_pass |\n")
        fh.write("|---|---:|---:|---:|\n")
        for w, c in sorted(per_worker.items()):
            fh.write(f"| {w} | {c['total']} | {c.get('old_pass', 0)} | {c.get('new_pass', 0)} |\n")
        fh.write("\n## Per stage\n\n| stage | total | old_pass | new_pass |\n")
        fh.write("|---|---:|---:|---:|\n")
        for s, c in sorted(per_stage.items()):
            fh.write(f"| {s} | {c['total']} | {c.get('old_pass', 0)} | {c.get('new_pass', 0)} |\n")

    print(f"\n[refilter] DONE")
    print(f"[refilter] old gate pass : {old_pass}/{n_total} ({100 * old_pass / max(n_total, 1):.1f}%)")
    print(f"[refilter] new gate pass : {new_pass}/{n_total} ({100 * new_pass / max(n_total, 1):.1f}%)")
    if args.dedup:
        print(f"[refilter] kept (deduped): {kept_total} | rejected: {rejected_total}")
    print(f"[refilter] report: {md_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers-glob", default=None,
                    help="glob of worker dirs (e.g. .../agentic_workers/gpu*)")
    ap.add_argument("--workers", nargs="*", default=None,
                    help="explicit worker dir paths")
    ap.add_argument("--stages", nargs="+",
                    default=["all"],
                    help="stage keys or 'all' (default: all)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows for a quick estimate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="skip rows already in checkpoint.jsonl")
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="cross-worker dedup on export (default: on)")
    ap.add_argument("--no-dedup", action="store_false", dest="dedup")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    return run_refilter(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
