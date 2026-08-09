"""Deterministic-render the subset named in render_allocation_deterministic.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()

    from targeted_tool_data.pilot43.pipeline import iter_jsonl, write_jsonl
    from targeted_tool_data.pilot43.qstage import (
        DETERMINISTIC, contract_seed, llm_records, render_task, selectable_rows)
    from targeted_tool_data.pilot43.resume import SELECTABLE_FINAL

    out = Path(args.out_dir)
    alloc = {r["task_id"]: r for r in iter_jsonl(
        out / "render_allocation_deterministic.jsonl")}
    if not alloc:
        print(json.dumps({"written": 0, "reason": "empty allocation"}))
        return 0

    by_id = {r["task_id"]: r for r in selectable_rows(out)}
    # Prefer the frozen selectable file when present (authoritative).
    if (out / SELECTABLE_FINAL).exists():
        by_id.update({r["task_id"]: r for r in iter_jsonl(out / SELECTABLE_FINAL)})

    llm = llm_records(out)
    done = {r["task_id"] for r in iter_jsonl(out / DETERMINISTIC)} if (
        out / DETERMINISTIC).exists() else set()

    written = skipped = missing = 0
    for tid, plan in sorted(alloc.items()):
        if tid in llm or tid in done:
            skipped += 1
            continue
        row = by_id.get(tid)
        if row is None:
            missing += 1
            continue
        mode = plan.get("planned_mode") or "OPERATION_EXPLICIT_GRAPH_IMPLICIT"
        rec = render_task(row, mode, seed=contract_seed(tid, args.seed))
        if rec.get("dropped"):
            missing += 1
            continue
        write_jsonl(out / DETERMINISTIC, [rec], append=True)
        written += 1

    report = {"written": written, "skipped_already_present": skipped,
              "missing_or_dropped": missing, "allocated": len(alloc)}
    print(json.dumps(report, indent=2))
    (out / "deterministic_allocated_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
