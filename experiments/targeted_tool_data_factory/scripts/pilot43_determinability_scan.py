"""Classify every plan by whether an implicit query can determine it.

Two plan shapes cannot be asked for without stating the recipe:

* a *computed criterion* -- a threshold, minimum or pad width produced by another
  step, so the rule ("above the mean plus the spread") exists only in the graph;
* an *opaque composite sink* -- a formatting or concatenation step that welds
  unrelated quantities together, so the shape of the answer is a private
  convention rather than a quantity anyone would name.

Both are answerable when the query states the rule, which is what the
semi-implicit and operation-explicit modes are for. The scan reports how much of
the pool that would move, per call-count bucket, before any routing is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from targeted_tool_data.pilot43.blueprints import all_blueprints
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.pipeline import SHORTLIST, VERIFIED, iter_jsonl

CRITERION_PARAMS = {"threshold", "minimum", "cutoff", "limit", "tolerance"}
CRITERION_WIDTH_OPS = {"format.pad"}
#: capabilities that only *present* a value; welding two computed values together
#: with one of these produces an answer shape nobody would ask for by name
COMPOSING = {"format.tag", "string.concat", "format.pad", "format.join_fields",
             "string.join", "list.combine_append", "list.combine_concat"}


def plan_flags(ops):
    out = {}
    for bp in all_blueprints():
        for plan in bp.plans:
            criterion, composite = set(), ""
            for step in plan.steps:
                op = ops.get(step.capability)
                if op is None:
                    continue
                for param, arg in zip(op.params, step.args):
                    if not arg.startswith("@"):
                        continue
                    if (param.name in CRITERION_PARAMS
                            or (param.name == "width"
                                and step.capability in CRITERION_WIDTH_OPS)):
                        criterion.add(f"{step.capability}.{param.name}")
            sink = plan.step(plan.sink)
            if (sink.capability in COMPOSING
                    and sum(1 for a in sink.args if a.startswith("@")) >= 2):
                composite = sink.capability
            out[(bp.workflow_id, plan.plan_id)] = (criterion, composite,
                                                   plan.call_count)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()

    ops = {op.capability: op for op in build_ops().values()}
    flags = plan_flags(ops)

    by_bucket_plans: dict[str, Counter] = defaultdict(Counter)
    for (_wf, _pid), (criterion, composite, calls) in flags.items():
        bucket = "6+" if calls >= 6 else str(calls)
        kind = ("criterion" if criterion else "") + ("+composite" if composite
                                                    else "")
        by_bucket_plans[bucket][kind or "determinable"] += 1
    print("plans by call bucket:")
    for bucket in sorted(by_bucket_plans, key=lambda b: (b == "6+", b)):
        counts = by_bucket_plans[bucket]
        total = sum(counts.values())
        det = counts["determinable"]
        print(f"  {bucket:<3} total={total:<4} determinable={det:<4} "
              f"({det / total:.2f})  {dict(counts)}")

    out_dir = Path(args.out_dir)
    if not (out_dir / VERIFIED).is_file():
        return
    selectable = {r["task_id"] for r in iter_jsonl(out_dir / VERIFIED)
                  if r.get("selectable")}
    tasks: dict[str, Counter] = defaultdict(Counter)
    for row in iter_jsonl(out_dir / SHORTLIST):
        if row["task_id"] not in selectable:
            continue
        criterion, composite, _calls = flags[(row["workflow_id"],
                                             row["plan_id"])]
        bucket = row["call_bucket"]
        kind = "determinable"
        if criterion and composite:
            kind = "both"
        elif criterion:
            kind = "criterion"
        elif composite:
            kind = "composite"
        tasks[bucket][kind] += 1

    print("\nselectable tasks by call bucket:")
    grand = Counter()
    for bucket in sorted(tasks, key=lambda b: (b == "6+", b)):
        counts = tasks[bucket]
        grand.update(counts)
        total = sum(counts.values())
        det = counts["determinable"]
        print(f"  {bucket:<3} total={total:<5} determinable={det:<5} "
              f"({det / total:.2f})  {dict(counts)}")
    total = sum(grand.values())
    print(f"  ALL total={total} determinable={grand['determinable']} "
          f"({grand['determinable'] / max(total, 1):.3f})  {dict(grand)}")


if __name__ == "__main__":
    main()
