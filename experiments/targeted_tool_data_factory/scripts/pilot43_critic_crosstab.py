"""Cross-tabulate critic verdicts against plan properties.

Tells apart "the writer wrote a poor query" from "the plan cannot be asked for in
this mode", which need completely different fixes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from targeted_tool_data.pilot43.blueprints import all_blueprints
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.pipeline import SHORTLIST, iter_jsonl

CRITERION_PARAMS = {"threshold", "minimum", "cutoff", "limit", "tolerance"}
#: ``format.pad.width`` is a layout rule; ``geometry.*.width`` is a measurement the
#: user states, so the parameter name alone is not enough to tell them apart
CRITERION_WIDTH_OPS = {"format.pad"}


def computed_criterion_plans() -> dict[tuple[str, str], set[str]]:
    ops = {op.capability: op for op in build_ops().values()}
    out: dict[tuple[str, str], set[str]] = {}
    for bp in all_blueprints():
        for plan in bp.plans:
            hits = set()
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
                        hits.add(f"{step.capability}.{param.name}")
            if hits:
                out[(bp.workflow_id, plan.plan_id)] = hits
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    criterion = computed_criterion_plans()
    rendered = {}
    with (out_dir / "llm_rendered.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            rendered[rec["task_id"]] = rec

    plans = {}
    for row in iter_jsonl(out_dir / SHORTLIST):
        if row["task_id"] in rendered:
            plans[row["task_id"]] = (row["workflow_id"], row["plan_id"])

    table: Counter = Counter()
    for tid, rec in rendered.items():
        critic = rec.get("critic") or {}
        nodes_ok = critic.get("all_program_nodes_required")
        has = plans.get(tid) in criterion
        table[(has, nodes_ok)] += 1
    print("(computed_criterion, all_program_nodes_required) -> count")
    for key, count in sorted(table.items(), key=lambda kv: str(kv[0])):
        print(f"  {key}: {count}")

    rejected = [tid for tid, rec in rendered.items()
                if (rec.get("critic") or {}).get(
                    "all_program_nodes_required") is False]
    with_criterion = [tid for tid in rejected if plans.get(tid) in criterion]
    print(f"\nrejected on nodes: {len(rejected)}; "
          f"of those with a computed criterion: {len(with_criterion)}")
    print("rejected without a computed criterion:")
    for tid in rejected:
        if tid in with_criterion:
            continue
        rec = rendered[tid]
        print(f"  {tid} {plans.get(tid)} {rec['requested_mode']}")


if __name__ == "__main__":
    main()
