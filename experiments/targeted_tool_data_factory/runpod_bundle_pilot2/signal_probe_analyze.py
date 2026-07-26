#!/usr/bin/env python3
"""Analysis stage of the Pilot2 signal probe — CPU only, no model, no training.

Two modes:

  ``--mode select-p3``  read the P2 shards, compute group metrics and write the
                        list of boundary tasks the P3 pass should re-probe at 8
                        rollouts (spec §4).

  ``--mode report``     read every shard, write ``rollouts.jsonl`` /
                        ``groups.jsonl`` / ``SIGNAL_PROBE_REPORT.{md,json}`` and
                        the recommended Phase-1 / deferred Phase-2 task files
                        (spec §5-§8).

Usage (repo root):
    python signal_probe_analyze.py --mode report \
        --probe-dir .../outputs/runpod_pilot2/signal_probe \
        --data .../data/train_grpo_pilot2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
from signal_probe_lib import (  # noqa: E402
    build_group, build_report, extract_task_meta, render_report_md,
    select_p3_tasks, select_phase1,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(probe_dir: Path, phases: List[str]) -> List[Dict[str, Any]]:
    """Merge every shard, de-duplicating on the content-hash cache key."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for phase in phases:
        for shard in sorted(probe_dir.glob(f"shard_{phase.lower()}_*.jsonl")):
            for rec in read_jsonl(shard):
                key = str(rec.get("cache_key") or
                          f"{rec.get('phase')}|{rec.get('task_id')}|{rec.get('rollout_idx')}")
                by_key[key] = rec
    return sorted(by_key.values(), key=lambda r: (str(r.get("phase")),
                                                  str(r.get("task_id")),
                                                  int(r.get("rollout_idx") or 0)))


def build_groups(records: List[Dict[str, Any]],
                 meta_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[(str(rec.get("phase")), str(rec.get("task_id")))].append(rec)
    groups: List[Dict[str, Any]] = []
    for (phase, task_id), recs in sorted(buckets.items()):
        meta = meta_by_id.get(task_id) or {
            "task_id": task_id,
            "track": recs[0].get("track"),
            "call_count": recs[0].get("call_count"),
            "motif": recs[0].get("motif"),
            "answer_type": recs[0].get("answer_type"),
            "generation_cell": recs[0].get("generation_cell"),
        }
        groups.append(build_group(meta, recs, phase=phase))
    return groups


def load_provenance(probe_dir: Path) -> Dict[str, Any]:
    """Reconstruct the run's provenance from the worker manifests."""
    manifests = sorted(probe_dir.glob("manifest_*.json"))
    if not manifests:
        return {}
    first = json.loads(manifests[0].read_text(encoding="utf-8"))
    sig = first.get("probe_signature") or {}
    per_phase_rollouts: Dict[str, Any] = {}
    for path in manifests:
        m = json.loads(path.read_text(encoding="utf-8"))
        per_phase_rollouts[str(m.get("phase"))] = m.get("rollouts_per_task")
    return {
        "worker_version": first.get("worker_version"),
        "model": sig.get("model"),
        "dtype": sig.get("dtype"),
        "lora_adapter": sig.get("lora_adapter"),
        "dataset": first.get("dataset"),
        "dataset_sha256": sig.get("dataset_sha256"),
        "reward_arm": sig.get("reward_arm"),
        "resolved_reward_policy": (first.get("reward") or {}).get("resolved_policy"),
        "executor_mode": first.get("executor_mode"),
        "synthetic_tools_dir": (first.get("overrides") and "trainer_adapter") or None,
        "registry_hash": sig.get("registry_hash"),
        "registry_version": sig.get("registry_version"),
        "temperature": sig.get("temperature"),
        "top_p": sig.get("top_p"),
        "seed": sig.get("seed"),
        "backend": first.get("backend"),
        "p2_rollouts": per_phase_rollouts.get("P2"),
        "p3_rollouts": per_phase_rollouts.get("P3"),
        "mt_grpo": first.get("mt_grpo"),
        "training_performed": False,
        "optimizer_steps": 0,
    }


def write_jsonl(path: Path, rows: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["select-p3", "report"], required=True)
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True,
                    help="frozen train_grpo_pilot2.jsonl (verbatim source rows)")
    ap.add_argument("--p3-limit", type=int, default=64)
    ap.add_argument("--phase1-target", type=int, default=100)
    ap.add_argument("--phase1-min", type=int, default=80)
    ap.add_argument("--phase1-max", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    meta_by_id = {}
    raw_by_id = {}
    for row in rows:
        meta = extract_task_meta(row)
        meta_by_id[meta["task_id"]] = meta
        raw_by_id[meta["task_id"]] = row

    if args.dry_run:
        print(f"[analyze] DRY RUN mode={args.mode} probe_dir={args.probe_dir} "
              f"tasks={len(rows)} — nothing written.")
        return 0

    args.probe_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "select-p3":
        records = load_records(args.probe_dir, ["P2"])
        if not records:
            print("[analyze] ABORT: no P2 rollouts found", file=sys.stderr)
            return 2
        groups = build_groups(records, meta_by_id)
        selection = select_p3_tasks(groups, limit=args.p3_limit)
        (args.probe_dir / "p3_task_ids.txt").write_text(
            "\n".join(selection["task_ids"]) + "\n", encoding="utf-8")
        (args.probe_dir / "p3_selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
        write_jsonl(args.probe_dir / "groups_p2.jsonl", groups)
        print(f"[analyze] P2 groups: {len(groups)}")
        print(f"[analyze] P3 selected: {len(selection['task_ids'])} tasks "
              f"{selection['selected_bucket_counts']}")
        print(f"[analyze] -> {args.probe_dir / 'p3_task_ids.txt'}")
        return 0

    # ── report ───────────────────────────────────────────────────────────
    records = load_records(args.probe_dir, ["P2", "P3"])
    if not records:
        print("[analyze] ABORT: no rollouts found", file=sys.stderr)
        return 2
    groups = build_groups(records, meta_by_id)

    write_jsonl(args.probe_dir / "rollouts.jsonl", records)
    write_jsonl(args.probe_dir / "groups.jsonl", groups)

    # Selection uses the DEEPEST evidence per task: a P3 group (8 rollouts)
    # supersedes the same task's P2 group (4 rollouts).
    decision: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        tid = str(g["task_id"])
        if tid not in decision or g.get("phase") == "P3":
            decision[tid] = g
    decision_groups = [decision[t] for t in sorted(decision)]

    phase1 = select_phase1(decision_groups, target=args.phase1_target,
                           min_size=args.phase1_min, max_size=args.phase1_max)
    selected_ids = [t for t in phase1["task_ids"] if t in raw_by_id]
    deferred_ids = [t for t in sorted(raw_by_id) if t not in set(selected_ids)]

    # Verbatim source rows so the subset can be hashed and trained directly.
    write_jsonl(args.probe_dir / "recommended_phase1_train.jsonl",
                [raw_by_id[t] for t in selected_ids])
    write_jsonl(args.probe_dir / "deferred_phase2_tasks.jsonl",
                [raw_by_id[t] for t in deferred_ids])

    p3_selection = {}
    p3_path = args.probe_dir / "p3_selection.json"
    if p3_path.is_file():
        p3_selection = json.loads(p3_path.read_text(encoding="utf-8"))

    provenance = load_provenance(args.probe_dir)
    provenance["synthetic_tools_dir"] = str(
        (BUNDLE.parent / "trainer_adapter").resolve())
    report = build_report(groups=groups, records=records, provenance=provenance,
                          phase1=phase1, p3_selection=p3_selection,
                          deferred_count=len(deferred_ids))
    report["files"] = {
        "rollouts": "rollouts.jsonl",
        "groups": "groups.jsonl",
        "recommended_phase1_train": "recommended_phase1_train.jsonl",
        "deferred_phase2_tasks": "deferred_phase2_tasks.jsonl",
    }
    report["phase1_task_ids"] = selected_ids
    report["deferred_task_ids"] = deferred_ids

    (args.probe_dir / "SIGNAL_PROBE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.probe_dir / "SIGNAL_PROBE_REPORT.md").write_text(
        render_report_md(report), encoding="utf-8")

    summary = report["summary"]
    verdict = report["verdict"]
    print(f"[analyze] groups={len(groups)} rollouts={len(records)} "
          f"decision_phase={report['decision_phase']}")
    print(f"[analyze] dead={summary.get('dead_group_rate')} "
          f"terminal_mixed={summary.get('terminal_mixed_rate')} "
          f"process_only_mixed={summary.get('process_only_mixed_rate')}")
    print(f"[analyze] reward_ordering_valid={verdict.get('ordering_valid')} "
          f"inversions={report['reward_ordering'].get('inversions')}")
    print(f"[analyze] phase1={len(selected_ids)} deferred={len(deferred_ids)}")
    print(f"[analyze] VERDICT: {verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"[analyze]   {reason}")
    print(f"[analyze] report -> {args.probe_dir / 'SIGNAL_PROBE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
