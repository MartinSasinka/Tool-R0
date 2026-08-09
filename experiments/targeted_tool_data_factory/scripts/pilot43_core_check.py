"""Smoke check for the Pilot4.3 validation and query layers.

Builds a few instances and runs necessity, V4, distractor selection, contract
rendering and deterministic query validation over them, printing timings so the
full-scale run can be sized.

    python scripts/pilot43_core_check.py --n 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from targeted_tool_data.pilot43 import blueprints as B  # noqa: E402
from targeted_tool_data.pilot43 import build as BD  # noqa: E402
from targeted_tool_data.pilot43 import counterfactuals as CF  # noqa: E402
from targeted_tool_data.pilot43 import distractors as D  # noqa: E402
from targeted_tool_data.pilot43 import necessity as N  # noqa: E402
from targeted_tool_data.pilot43 import profile as P  # noqa: E402
from targeted_tool_data.pilot43 import qvalidate as QV  # noqa: E402
from targeted_tool_data.pilot43 import queries as Q  # noqa: E402
from targeted_tool_data.pilot43 import v4 as V4  # noqa: E402
from targeted_tool_data.pilot43.program import gold_calls  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prof = P.build_profile_v3()
    print("profile:", prof["source_file"], "hash", prof["profile_hash"][:12],
          "n_rows", prof["source_n_rows"])
    print("  call targets:", {k: v["target_share"]
                              for k, v in prof["call_count"].items()})
    print("  answer target 6+:", prof["answer_type"]["target"]["6+"])
    print("  query modes:", prof["query_mode"]["overall"])
    print("  6+ minima:", {k: v for k, v in prof["long_horizon_minima"].items()
                           if k != "dev_observed"})

    bps = [bp for bp in B.all_blueprints()
           if not args.only or args.only in bp.workflow_id]
    queries = []
    for bp in bps:
        for plan in bp.plans[:args.n]:
            t0 = time.perf_counter()
            try:
                inst = BD.instantiate(bp, plan, 4242, track="A_NATIVE")
            except BD.BuildError as exc:
                print(f"!! {bp.workflow_id}/{plan.plan_id}: {exc}")
                continue
            t_build = time.perf_counter() - t0

            cf_insts, cf_meta = CF.counterfactual_instances(
                bp, plan, answer_type=inst.answer_type, track="A_NATIVE",
                seed=909091, n=6)
            cf_progs = CF.as_programs(cf_insts)
            cfs = CF.as_fact_pairs(cf_insts)

            t0 = time.perf_counter()
            nec = N.node_necessity(inst.program, allowed_ops=[],
                                   check_alternatives=False,
                                   counterfactuals=cf_progs)
            t_nec = time.perf_counter() - t0

            t0 = time.perf_counter()
            offered = D.build_offered_tools(inst.program, inst.answer,
                                            track=inst.track, target_count=11,
                                            seed=inst.seed,
                                            counterfactuals=cf_progs[:2])
            t_dist = time.perf_counter() - t0

            facts = {r: (inst.role_values[r],
                         next(rr.sem for rr in plan.roles if rr.name == r))
                     for r in inst.role_values}
            t0 = time.perf_counter()
            gate = V4.v4_gate(facts, [t["primitive_id"] for t in offered["tools"]],
                              inst.answer, inst.call_count, cfs,
                              counterfactuals_mixed=cf_meta["mixed"])
            t_v4 = time.perf_counter() - t0

            contract = Q.build_contract(inst, bp, plan, mode="SEMI_IMPLICIT",
                                        task_id=f"t_{plan.plan_id}", seed=7)
            payload = QV.contract_payload(
                contract, answer=inst.answer,
                gold_capabilities=[c["capability"] for c in
                                   gold_calls(inst.program, inst.track)],
                predicate_steps=sum(1 for s in plan.steps
                                    if s.capability.split(".")[0]
                                    in ("comparison", "boolean", "decision",
                                        "classification")))
            rows = []
            for mode in ("SEMI_IMPLICIT", "OPERATION_EXPLICIT_GRAPH_IMPLICIT",
                         "GRAPH_EXPLICIT", "DOMAIN_GROUNDED_IMPLICIT"):
                r = Q.render_deterministic(contract, mode, seed=11)
                payload["mode"] = mode
                v = QV.validate_query(r["query"], payload)
                rows.append((mode, v["passed"], v["failed_layers"],
                             v["classification"]["actual_query_mode"]))
                queries.append(r["query"])
                if args.verbose and not v["passed"]:
                    print("   ", mode, r["query"][:220])
                    print("     ", json.dumps({k: vv for k, vv in v["layers"].items()
                                               if not vv["passed"]})[:400])

            print(f"{bp.workflow_id}/{plan.plan_id} calls={inst.call_count} "
                  f"answer={inst.answer_type} nec_ok={N.all_nodes_necessary(nec)} "
                  f"v4_safe={gate['safe_for_core_train']} "
                  f"(shortcut={gate['has_shortcut']} resolved={gate['resolved']} "
                  f"depth={gate['search_space']['max_depth_complete']}"
                  f"/{gate['search_space']['max_depth_requested']} "
                  f"cand={gate['n_confirmed']}/{gate['n_coincidental']}) "
                  f"v4_exp={gate['expansions']} tools={offered['offered_tool_count']} "
                  f"hard={offered['hard_distractor_count']} "
                  f"cf={cf_meta['built']}/{cf_meta['distinct_answers']}distinct "
                  f"t(build/nec/dist/v4)={t_build:.2f}/{t_nec:.2f}/{t_dist:.2f}/{t_v4:.2f}s")
            if gate["confirmed_shortcuts"]:
                print("    shortcut:", gate["confirmed_shortcuts"][0]["rendered"][:180])
            if not N.all_nodes_necessary(nec):
                print("    unnecessary:", [r["node_id"] for r in nec
                                           if not r["necessary"]])
            for mode, ok, failed, actual in rows:
                print(f"    {mode:38s} pass={ok!s:5s} actual={actual:34s} {failed}")
            if args.verbose:
                print("    example:", Q.render_deterministic(
                    contract, "DOMAIN_GROUNDED_IMPLICIT", seed=3)["query"])

    div = QV.diversity_report(queries)
    print("\ndiversity:", json.dumps(div))
    print("gates:", QV.check_diversity(div))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
