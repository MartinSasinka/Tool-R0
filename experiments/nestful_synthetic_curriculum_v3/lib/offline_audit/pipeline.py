from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from lib.offline_audit import DEFAULT_SEED
from lib.offline_audit.adapters import adapter_analysis
from lib.offline_audit.coverage import coverage
from lib.offline_audit.credit import credit_audit
from lib.offline_audit.discovery import discover
from lib.offline_audit.eval_behavior import eval_behavior
from lib.offline_audit.groups import groups_inventory
from lib.offline_audit.heldout import prepare_heldout
from lib.offline_audit.on_policy import on_policy
from lib.offline_audit.optimizer import optimizer_audit
from lib.offline_audit.pairwise import counterfactual_and_pairwise
from lib.offline_audit.progress import training_progress
from lib.offline_audit.verdict import summarize_reports


def run_all(
    runs_root: Path,
    reports_dir: Path,
    *,
    seed: str = DEFAULT_SEED,
    canonical_arm: str = "A0_R0_CURRENT",
    strict: bool = False,
    skip_adapters: bool = False,
    skip_credit: bool = False,
    allow_partial: bool = True,
) -> Dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ctx: Dict[str, Any] = {}
    ctx["discovery"] = discover(runs_root, seed, reports_dir, strict=strict)
    if strict and ctx["discovery"].get("errors"):
        if not allow_partial:
            raise SystemExit(f"discovery strict failed: {ctx['discovery']['errors']}")
    ctx["coverage"] = coverage(runs_root, seed, reports_dir)
    ctx["groups"] = groups_inventory(runs_root, seed, reports_dir)
    ctx["on_policy"] = on_policy(runs_root, seed, reports_dir)
    ctx["progress"] = training_progress(runs_root, seed, reports_dir)
    ctx["pairwise"] = counterfactual_and_pairwise(
        runs_root, seed, reports_dir, canonical_arm
    )
    if not skip_credit:
        ctx["credit"] = credit_audit(runs_root, seed, reports_dir)
    ctx["optimizer"] = optimizer_audit(runs_root, seed, reports_dir)
    ctx["adapters"] = adapter_analysis(
        runs_root, seed, reports_dir, skip=skip_adapters
    )
    ctx["eval_behavior"] = eval_behavior(runs_root, seed, reports_dir)
    ctx["heldout"] = prepare_heldout(reports_dir)
    summarize_reports(reports_dir, ctx)
    return ctx
