"""Per-phase timing of the Pilot4.3 verification stage.

Verification is the pipeline's only expensive stage, and its cost decides whether a
12k render pool is a 20-minute or a 20-hour job, so it is measured rather than
guessed.

    python scripts/pilot43_profile_verify.py --n 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from targeted_tool_data.pilot43 import counterfactuals as CF  # noqa: E402
from targeted_tool_data.pilot43 import distractors as D  # noqa: E402
from targeted_tool_data.pilot43 import necessity as N  # noqa: E402
from targeted_tool_data.pilot43 import pipeline as P  # noqa: E402
from targeted_tool_data.pilot43 import tasks as T  # noqa: E402
from targeted_tool_data.pilot43 import v4 as V4  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/_p43_smoke")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    rows = P.read_jsonl(Path(args.dir) / P.SHORTLIST)[:args.n]
    totals = {"build": 0.0, "cf": 0.0, "dist": 0.0, "nec": 0.0, "v4": 0.0}
    for row in rows:
        t = time.perf_counter()
        inst, bp, plan = T.rebuild(row)
        t_build = time.perf_counter() - t

        t = time.perf_counter()
        cf_insts, cf_meta = CF.counterfactual_instances(
            bp, plan, answer_type=inst.answer_type, track=inst.track, seed=1)
        t_cf = time.perf_counter() - t
        progs = CF.as_programs(cf_insts)
        pairs = CF.as_fact_pairs(cf_insts)

        t = time.perf_counter()
        offered = D.build_offered_tools(inst.program, inst.answer,
                                        track=inst.track, target_count=10,
                                        seed=1, counterfactuals=progs[:6])
        t_dist = time.perf_counter() - t
        ids = [x["primitive_id"] for x in offered["tools"]]

        t = time.perf_counter()
        N.node_necessity(inst.program, allowed_ops=ids, check_alternatives=True,
                         counterfactuals=progs)
        t_nec = time.perf_counter() - t

        facts = {r.name: (inst.role_values[r.name], r.sem) for r in plan.roles}
        t = time.perf_counter()
        gate = V4.v4_gate(facts, ids, inst.answer, inst.call_count, pairs,
                          counterfactuals_mixed=cf_meta["mixed"])
        t_v4 = time.perf_counter() - t

        totals["build"] += t_build
        totals["cf"] += t_cf
        totals["dist"] += t_dist
        totals["nec"] += t_nec
        totals["v4"] += t_v4
        print(f"{inst.answer_type:9s} calls={inst.call_count:2d} "
              f"build={t_build:6.2f} cf={t_cf:6.2f}(n={cf_meta['built']}) "
              f"dist={t_dist:6.2f}(examined={offered['candidates_examined']}) "
              f"nec={t_nec:6.2f} v4={t_v4:6.2f}(exp={gate['expansions']})")
    n = max(1, len(rows))
    print("\nmean seconds per task:",
          {k: round(v / n, 2) for k, v in totals.items()},
          "total", round(sum(totals.values()) / n, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
