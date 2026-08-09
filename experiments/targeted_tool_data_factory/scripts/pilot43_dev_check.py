"""Developer check for the Pilot4.3 workflow registry.

Builds every blueprint plan several times and reports what the plans actually
produce (call count, answer types, satisfied structural patterns, bound
primitives) plus every build error, so a blueprint that cannot instantiate is
visible immediately instead of silently shrinking the pool.

    python scripts/pilot43_dev_check.py            # summary per plan
    python scripts/pilot43_dev_check.py --verbose  # plus one rendered example
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from targeted_tool_data.pilot43 import blueprints as B  # noqa: E402
from targeted_tool_data.pilot43 import build as BD  # noqa: E402
from targeted_tool_data.pilot43 import ops as O  # noqa: E402
from targeted_tool_data.pilot43.program import gold_calls  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    ops = O.build_ops()
    print(f"ops: {len(ops)}  capabilities: {len(O.ops_by_capability(ops))} "
          f"families: {len(O.ops_by_family(ops))} "
          f"coding: {sum(1 for o in ops.values() if o.coding_like)}")

    bps = [bp for bp in B.all_blueprints()
           if not args.only or args.only in bp.workflow_id]
    print(f"workflows: {len(B.all_blueprints())} (checking {len(bps)})")

    call_counts: Counter = Counter()
    answer_types: Counter = Counter()
    patterns: Counter = Counter()
    primitives: set[str] = set()
    families: set[str] = set()
    failures = 0
    total_plans = 0
    for bp in bps:
        for plan in bp.plans:
            total_plans += 1
            d = BD.describe_plan(bp, plan, trials=args.trials)
            ok = d["instantiated"]
            call_counts[d["call_count"]] += ok
            for k, v in d["answer_types"].items():
                answer_types[k] += v
            for k, v in d["actual_primary_patterns"].items():
                patterns[k] += v
            primitives.update(d["primitives_bound"])
            families.update(d["capability_families"])
            flag = "  " if ok == args.trials else "!!"
            if ok < args.trials:
                failures += 1
            print(f"{flag} {bp.workflow_id}/{plan.plan_id} "
                  f"calls={d['call_count']} ok={ok}/{args.trials} "
                  f"answers={dict(d['answer_types'])} "
                  f"primary={dict(d['actual_primary_patterns'])}"
                  + (f" errors={d['build_errors']}" if d["build_errors"] else ""))
            if args.verbose and ok:
                inst = None
                for i in range(args.trials):
                    try:
                        inst = BD.instantiate(bp, plan, 991 + i * 7919,
                                              track="A_NATIVE")
                        break
                    except BD.BuildError:
                        continue
                if inst is not None:
                    print(json.dumps(gold_calls(inst.program, "A_NATIVE"),
                                     indent=1, default=str)[:1400])
                    print("   answer:", repr(inst.answer)[:120],
                          "patterns:", inst.actual_patterns)

    print("\n--- totals ---")
    print("plans:", total_plans, "plans with failures:", failures)
    print("call counts:", dict(sorted(call_counts.items())))
    print("answer types:", dict(sorted(answer_types.items())))
    print("primary patterns:", dict(sorted(patterns.items())))
    print("distinct primitives bound:", len(primitives))
    print("capability families:", len(families), sorted(families))
    coding = sorted(f for f in families if f in O.CODING_FAMILIES)
    print("coding families:", len(coding), coding)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
