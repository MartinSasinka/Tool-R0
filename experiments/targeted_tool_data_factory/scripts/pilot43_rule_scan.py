"""Determinability of the verified pool: self-evident, rule-bearing, unstatable.

Run before rendering: it says how much of the pool can be asked for implicitly,
how much needs its rule stated, and how much has to be dropped, per call bucket.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from targeted_tool_data.pilot43 import determinability as det, qstage
from targeted_tool_data.pilot43.pipeline import SHORTLIST, VERIFIED, iter_jsonl
from targeted_tool_data.pilot43.queries import build_contract
from targeted_tool_data.pilot43.tasks import rebuild


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--show-unstatable", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    selectable = {r["task_id"] for r in iter_jsonl(out_dir / VERIFIED)
                  if r.get("selectable")}

    by_bucket: dict[str, Counter] = defaultdict(Counter)
    lengths: list[int] = []
    shown = 0
    scanned = 0
    for row in iter_jsonl(out_dir / SHORTLIST):
        tid = row["task_id"]
        if tid not in selectable:
            continue
        scanned += 1
        if scanned > args.limit:
            break
        inst, bp, plan = rebuild(row)
        contract = build_contract(inst, bp, plan, mode="SEMI_IMPLICIT",
                                  task_id=tid,
                                  seed=qstage.contract_seed(tid, args.seed))
        by_bucket[row["call_bucket"]][contract.determinability] += 1
        lengths.extend(len(r) for r in contract.specification)
        if contract.determinability == det.NOT_STATABLE and shown < args.show_unstatable:
            shown += 1
            print(f"\n[unstatable] {tid} {row['workflow_id']}/{row['plan_id']}")
            for rule in contract.specification:
                print(f"  {rule[:400]}")

    print(f"\nscanned {min(scanned, args.limit)} selectable tasks")
    grand: Counter = Counter()
    for bucket in sorted(by_bucket, key=lambda b: (b == "6+", b)):
        counts = by_bucket[bucket]
        grand.update(counts)
        total = sum(counts.values())
        print(f"  {bucket:<3} n={total:<5} "
              f"self_evident={counts[det.SELF_EVIDENT]:<5} "
              f"needs_rule={counts[det.NEEDS_RULE]:<4} "
              f"unstatable={counts[det.NOT_STATABLE]:<4}")
    total = sum(grand.values()) or 1
    print(f"  ALL n={total} self_evident={grand[det.SELF_EVIDENT]} "
          f"({grand[det.SELF_EVIDENT] / total:.3f}) "
          f"needs_rule={grand[det.NEEDS_RULE]} "
          f"({grand[det.NEEDS_RULE] / total:.3f}) "
          f"unstatable={grand[det.NOT_STATABLE]} "
          f"({grand[det.NOT_STATABLE] / total:.3f})")
    if lengths:
        lengths.sort()
        mid = lengths[len(lengths) // 2]
        print(f"  rule length: median={mid} max={lengths[-1]}")


if __name__ == "__main__":
    main()
