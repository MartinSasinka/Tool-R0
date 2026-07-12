#!/usr/bin/env python3
"""Merge per-GPU refilter shard outputs + cross-shard dedup + combined report."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)

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


def _dedup_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (question_hash(row.get("question", "")), trace_hash(row.get("gold_calls") or []))


def merge_shards(args) -> int:
    shards = sorted(
        p for p in glob.glob(args.shards_glob)
        if os.path.isdir(p))
    if not shards:
        print(f"[merge-refilter] ERROR: no shards match {args.shards_glob}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.output_dir)
    kept_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejected_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    shard_reports: List[Dict[str, Any]] = []

    for shard in shards:
        report_path = os.path.join(shard, "REFILTER_REPORT.json")
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as fh:
                shard_reports.append(json.load(fh))
        for sub in ("kept", "rejected"):
            for path in glob.glob(os.path.join(shard, sub, "*.jsonl")):
                fname = os.path.basename(path)
                target = kept_all if sub == "kept" else rejected_all
                target[fname].extend(_load_jsonl(path))

    seen: set = set()
    dedup_dropped = 0
    kept_deduped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fname, rows in kept_all.items():
        for row in rows:
            key = _dedup_key(row)
            if key in seen:
                dedup_dropped += 1
                continue
            seen.add(key)
            kept_deduped[fname].append(row)

    # Stats from shard reports
    rows_probed = sum(int(r.get("rows_probed") or 0) for r in shard_reports)
    old_pass = sum(int(r.get("old_gate_pass") or 0) for r in shard_reports)
    new_pass_raw = sum(int(r.get("new_gate_pass") or 0) for r in shard_reports)
    kept_total = sum(len(v) for v in kept_deduped.values())
    rejected_total = sum(len(v) for v in rejected_all.values())

    report = {
        "merge_version": 1,
        "shards": [os.path.basename(s) for s in shards],
        "rows_probed": rows_probed,
        "old_gate_pass_sum": old_pass,
        "new_gate_pass_sum": new_pass_raw,
        "dedup_dropped": dedup_dropped,
        "kept_after_dedup": kept_total,
        "rejected_total": rejected_total,
        "kept_per_file": {k: len(v) for k, v in kept_deduped.items()},
        "rejected_per_file": {k: len(v) for k, v in rejected_all.items()},
        "shard_reports": shard_reports,
    }

    os.makedirs(out_dir, exist_ok=True)
    for fname, rows in kept_deduped.items():
        _write_jsonl(os.path.join(out_dir, "kept", fname), rows)
    for fname, rows in rejected_all.items():
        _write_jsonl(os.path.join(out_dir, "rejected", fname), rows)

    json_path = os.path.join(out_dir, "MERGED_REFILTER_REPORT.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    md_path = os.path.join(out_dir, "MERGED_REFILTER_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Merged refilter report\n\n")
        fh.write(f"- **Shards:** {', '.join(report['shards'])}\n")
        fh.write(f"- **Rows probed (sum):** {rows_probed}\n")
        fh.write(f"- **Old gate pass (sum):** {old_pass}\n")
        fh.write(f"- **New gate pass (per-shard sum):** {new_pass_raw}\n")
        fh.write(f"- **Cross-shard dedup dropped:** {dedup_dropped}\n")
        fh.write(f"- **Kept after dedup:** {kept_total}\n")
        fh.write(f"- **Rejected (all shards):** {rejected_total}\n\n")
        fh.write("## Kept per file\n\n")
        for k, n in sorted(report["kept_per_file"].items()):
            fh.write(f"- `{k}`: {n}\n")

    print(f"[merge-refilter] shards={len(shards)} probed={rows_probed}")
    print(f"[merge-refilter] kept after dedup={kept_total} (dropped {dedup_dropped} dupes)")
    print(f"[merge-refilter] rejected={rejected_total}")
    print(f"[merge-refilter] report: {md_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-glob", required=True,
                    help="glob of per-GPU refilter dirs (e.g. .../refilter_mt_gate/refilter_gpu*)")
    ap.add_argument("--output-dir", required=True)
    return merge_shards(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
