"""Print the derived rule sentences for tasks that need one.

The rule text is generated from the program, so it has to be read before it is
sent to a writer: an unreadable rule produces an unanswerable query just as surely
as a missing one.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from targeted_tool_data.pilot43 import qstage
from targeted_tool_data.pilot43.pipeline import SHORTLIST, VERIFIED, iter_jsonl
from targeted_tool_data.pilot43.queries import build_contract, render_deterministic
from targeted_tool_data.pilot43.tasks import rebuild


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_ids", nargs="*")
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wanted = set(args.task_ids)
    selectable = {r["task_id"] for r in iter_jsonl(out_dir / VERIFIED)
                  if r.get("selectable")}

    shown = scanned = with_rule = 0
    reasons: Counter = Counter()
    for row in iter_jsonl(out_dir / SHORTLIST):
        tid = row["task_id"]
        if wanted:
            if tid not in wanted:
                continue
        elif tid not in selectable or scanned >= args.limit:
            if scanned >= args.limit:
                break
            continue
        scanned += 1
        inst, bp, plan = rebuild(row)
        contract = build_contract(inst, bp, plan, mode="SEMI_IMPLICIT",
                                  task_id=tid,
                                  seed=qstage.contract_seed(tid, args.seed))
        if not contract.specification:
            continue
        with_rule += 1
        reasons[len(contract.specification)] += 1
        if shown >= args.show and not wanted:
            continue
        shown += 1
        print(f"\n=== {tid} {row['workflow_id']}/{row['plan_id']} "
              f"({row['call_count']} calls, {row['answer_type']})")
        print(f"target: {contract.target_phrase}")
        for rule in contract.specification:
            print(f"  RULE: {rule}")
        print("  rendered: "
              + render_deterministic(contract, "SEMI_IMPLICIT", seed=1)["query"])

    print(f"\nscanned {scanned}, needing a rule {with_rule}, "
          f"rules per task {dict(reasons)}")


if __name__ == "__main__":
    main()
