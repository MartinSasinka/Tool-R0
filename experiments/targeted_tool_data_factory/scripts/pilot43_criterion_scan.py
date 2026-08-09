"""Count plans whose decision criterion is computed rather than stated.

A criterion parameter fed by another step ("count parts above <mean+stdev>") is a
convention the query cannot convey without naming it, so such a task is not
answerable in a fully implicit mode however well the query is written. This scans
the plan registry and weights each plan by how often the shortlist uses it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from targeted_tool_data.pilot43.blueprints import all_blueprints
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.pipeline import SHORTLIST, VERIFIED, iter_jsonl

#: parameter names whose value *is* a rule rather than a subject of the request
CRITERION_PARAMS = {
    "threshold", "limit", "reference", "tolerance", "cap", "bound", "upper",
    "lower", "minimum", "maximum", "target_value", "column_width", "width",
    "allowance", "budget", "quota", "cutoff", "band", "ceiling", "floor",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    ops = build_ops()
    by_capability = {op.capability: op for op in ops.values()}
    computed: dict[tuple[str, str], set[str]] = {}
    total_plans = 0
    for bp in all_blueprints():
        for plan in bp.plans:
            total_plans += 1
            hits: set[str] = set()
            for step in plan.steps:
                op = by_capability.get(step.capability)
                if op is None:
                    continue
                for param, arg in zip(op.params, step.args):
                    if param.name in CRITERION_PARAMS and arg.startswith("@"):
                        hits.add(f"{step.capability}.{param.name}")
            if hits:
                computed[(bp.workflow_id, plan.plan_id)] = hits

    print(f"plans in registry: {total_plans}")
    print(f"plans with a computed criterion: {len(computed)} "
          f"({len(computed) / max(total_plans, 1):.3f})")
    params = Counter(h for hits in computed.values() for h in hits)
    print(f"criterion parameters: {params.most_common(args.show)}")

    out_dir = Path(args.out_dir)
    verified = out_dir / VERIFIED
    if not verified.is_file():
        return
    selectable = {r["task_id"] for r in iter_jsonl(verified) if r.get("selectable")}
    n = affected = 0
    workflows: Counter = Counter()
    for row in iter_jsonl(out_dir / SHORTLIST):
        if row["task_id"] not in selectable:
            continue
        n += 1
        if (row["workflow_id"], row["plan_id"]) in computed:
            affected += 1
            workflows[row["workflow_id"]] += 1
    print(f"\nselectable tasks: {n}")
    print(f"tasks with a computed criterion: {affected} "
          f"({affected / max(n, 1):.3f})")
    print(f"top workflows: {workflows.most_common(args.show)}")


if __name__ == "__main__":
    main()
