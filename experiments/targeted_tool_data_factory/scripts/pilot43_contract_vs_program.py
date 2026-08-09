"""Show, for one task, every program literal next to the facts the query states.

A node the query cannot determine is unanswerable however well it is written, so
this is the first thing to check when a critic says a node is not required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from targeted_tool_data.pilot43 import qstage
from targeted_tool_data.pilot43.pipeline import SHORTLIST, iter_jsonl
from targeted_tool_data.pilot43.program import gold_calls
from targeted_tool_data.pilot43.queries import build_contract
from targeted_tool_data.pilot43.tasks import rebuild


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_ids", nargs="+")
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wanted = set(args.task_ids)
    rendered = {}
    with (out_dir / "llm_rendered.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["task_id"] in wanted:
                rendered[rec["task_id"]] = rec

    for row in iter_jsonl(out_dir / SHORTLIST):
        tid = row["task_id"]
        if tid not in wanted:
            continue
        inst, bp, plan = rebuild(row)
        contract = build_contract(inst, bp, plan,
                                  mode=rendered.get(tid, {}).get(
                                      "requested_mode", "GOAL_BASED_IMPLICIT"),
                                  task_id=tid,
                                  seed=qstage.contract_seed(tid, args.seed))
        print(f"\n================ {tid}  {row['workflow_id']} / {plan.plan_id}")
        print(f"blueprint goal : {bp.natural_user_goal}")
        print(f"target phrase  : {contract.target_phrase}")
        print(f"sink purpose   : {plan.steps[-1].purpose}  answer={inst.answer!r}")
        print("\nstated facts:")
        for f in contract.facts:
            print(f"  {f.role:<26} {f.description:<46} = {f.rendered!r}")
        print("\nprogram:")
        for call in gold_calls(inst.program, inst.track, inst.observations):
            print(f"  {call['node_id']:<4} {call['capability']:<28} "
                  f"{json.dumps(call['arguments'])}  -> {call['observation']!r}")
        stated = {f.rendered for f in contract.facts}
        stated |= {str(f.value) for f in contract.facts}
        literals = []
        for call in gold_calls(inst.program, inst.track, inst.observations):
            for key, value in call["arguments"].items():
                text = str(value)
                if text.startswith("$"):
                    continue
                if text not in stated and all(text not in s for s in stated):
                    literals.append((call["node_id"], key, value))
        print(f"\nliterals the query never states: {literals}")
        query = rendered.get(tid, {}).get("query")
        if query:
            print(f"\nquery: {query}")


if __name__ == "__main__":
    main()
