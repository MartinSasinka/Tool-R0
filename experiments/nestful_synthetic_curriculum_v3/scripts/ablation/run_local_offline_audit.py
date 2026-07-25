#!/usr/bin/env python3
"""Local offline audit of Round-1 reward ablation runs (no training / no inference)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_V3 = Path(__file__).resolve().parents[2]
if str(_V3) not in sys.path:
    sys.path.insert(0, str(_V3))

from lib.offline_audit import DEFAULT_SEED  # noqa: E402
from lib.offline_audit.adapters import adapter_analysis  # noqa: E402
from lib.offline_audit.coverage import coverage  # noqa: E402
from lib.offline_audit.credit import credit_audit  # noqa: E402
from lib.offline_audit.discovery import discover  # noqa: E402
from lib.offline_audit.eval_behavior import eval_behavior  # noqa: E402
from lib.offline_audit.groups import groups_inventory  # noqa: E402
from lib.offline_audit.heldout import prepare_heldout  # noqa: E402
from lib.offline_audit.on_policy import on_policy  # noqa: E402
from lib.offline_audit.optimizer import optimizer_audit  # noqa: E402
from lib.offline_audit.pairwise import counterfactual_and_pairwise  # noqa: E402
from lib.offline_audit.pipeline import run_all  # noqa: E402
from lib.offline_audit.progress import training_progress  # noqa: E402
from lib.offline_audit.verdict import summarize_reports  # noqa: E402


def _default_runs_root() -> Path:
    return _V3 / "outputs" / "runs" / "_local_round1_analysis"


def _default_reports() -> Path:
    return _V3 / "reports" / "reward_ablation" / "offline_audit"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Round-1 reward ablation offline audit")
    p.add_argument(
        "command",
        choices=[
            "discover",
            "coverage",
            "groups",
            "on-policy",
            "counterfactual",
            "credit",
            "optimizer",
            "adapters",
            "eval-behavior",
            "prepare-heldout",
            "summarize",
            "all",
        ],
    )
    p.add_argument("--runs-root", type=Path, default=_default_runs_root())
    p.add_argument("--reports-dir", type=Path, default=_default_reports())
    p.add_argument("--canonical-arm", default="A0_R0_CURRENT")
    p.add_argument("--seed", default=DEFAULT_SEED)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--resume", action="store_true", help="ignored; audit is fast/idempotent")
    p.add_argument("--skip-adapters", action="store_true")
    p.add_argument("--skip-credit", action="store_true")
    p.add_argument("--allow-partial", action="store_true", default=True)
    args = p.parse_args(argv)

    runs_root = args.runs_root
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "all":
        run_all(
            runs_root,
            reports_dir,
            seed=args.seed,
            canonical_arm=args.canonical_arm,
            strict=args.strict,
            skip_adapters=args.skip_adapters,
            skip_credit=args.skip_credit,
            allow_partial=args.allow_partial,
        )
        return 0

    ctx = {}
    if args.command == "discover":
        discover(runs_root, args.seed, reports_dir, strict=args.strict)
    elif args.command == "coverage":
        coverage(runs_root, args.seed, reports_dir)
    elif args.command == "groups":
        groups_inventory(runs_root, args.seed, reports_dir)
    elif args.command == "on-policy":
        on_policy(runs_root, args.seed, reports_dir)
    elif args.command == "counterfactual":
        counterfactual_and_pairwise(
            runs_root, args.seed, reports_dir, args.canonical_arm
        )
    elif args.command == "credit":
        credit_audit(runs_root, args.seed, reports_dir)
    elif args.command == "optimizer":
        optimizer_audit(runs_root, args.seed, reports_dir)
    elif args.command == "adapters":
        adapter_analysis(runs_root, args.seed, reports_dir, skip=False)
    elif args.command == "eval-behavior":
        eval_behavior(runs_root, args.seed, reports_dir)
    elif args.command == "prepare-heldout":
        prepare_heldout(reports_dir)
    elif args.command == "summarize":
        # minimal re-load from json artifacts if present
        summarize_reports(reports_dir, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
