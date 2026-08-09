"""Re-plan render allocation after a low-yield full pass.

Uses the observed clean yield, prefers never-attempted tasks, and pushes more
SEMI_IMPLICIT / explicit modes onto the deterministic channel so remaining
OpenRouter budget buys DOMAIN/GOAL coverage.
"""
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
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--clean-target", type=int, default=9000)
    ap.add_argument("--expected-yield", type=float, default=0.30)
    ap.add_argument("--render-ceiling", type=int, default=11500)
    args = ap.parse_args()

    from targeted_tool_data.pilot43 import resume as res

    out = Path(args.out_dir)
    # refresh reused from current llm_rendered before planning
    reused = res.export_reused_llm(out)
    plan = res.build_render_allocation(
        out,
        seed=args.seed,
        clean_target=args.clean_target,
        render_ceiling=args.render_ceiling,
        expected_yield=args.expected_yield,
    )
    print(json.dumps({
        "reused": reused,
        "n_allocated_llm": plan["n_allocated_llm"],
        "n_allocated_deterministic": plan["n_allocated_deterministic"],
        "projected_clean": plan["projected_clean"],
        "n_already_attempted": plan["n_already_attempted"],
        "llm_budget_remaining_attempts": plan["llm_budget_remaining_attempts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
