"""Pilot4.2 coverage cells and hard support targets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from . import QUERY_MODES
from .workflows_v2 import get_workflows

CALL_BUCKETS = ("2", "3", "4", "5", "6+")
TIER_ORDER = ("CORE_PROFILE", "STRUCTURAL_ENRICHMENT",
              "CAPABILITY_ENRICHMENT", "CHALLENGE")


@dataclass(frozen=True)
class Cell:
    cell_id: str
    workflow_id: str
    tier: str
    structural_skill: str
    call_count_bucket: str
    query_mode: str
    target_count: int
    min_support: int
    track: str = "A_NATIVE"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bucket(n: int) -> str:
    if n >= 6:
        return "6+"
    return str(n)


def build_cells() -> List[Cell]:
    cells: List[Cell] = []
    workflows = get_workflows()
    modes = [m for m in QUERY_MODES if m != "GRAPH_EXPLICIT"] + ["GRAPH_EXPLICIT"]
    for i, workflow in enumerate(workflows):
        n = len(workflow.plan_template)
        bucket = _bucket(n)
        if workflow.coding_like:
            tier = "CAPABILITY_ENRICHMENT"
            min_support = 8
            target = 25
        elif n >= 5:
            tier = "STRUCTURAL_ENRICHMENT" if i % 3 else "CHALLENGE"
            min_support = 10 if tier != "CHALLENGE" else 5
            target = 30 if tier != "CHALLENGE" else 12
        elif i < max(40, int(0.65 * len(workflows))):
            tier = "CORE_PROFILE"
            min_support = 15
            target = 40
        else:
            tier = "STRUCTURAL_ENRICHMENT"
            min_support = 10
            target = 28
        pattern = workflow.allowed_structural_patterns[
            i % len(workflow.allowed_structural_patterns)]
        mode = modes[i % len(modes)]
        track = "A_NATIVE" if i % 5 < 3 else ("G_GENERAL_1" if i % 2 else "G_GENERAL_2")
        cells.append(Cell(
            cell_id=f"p42_cell_{i:03d}", workflow_id=workflow.workflow_id,
            tier=tier, structural_skill=pattern, call_count_bucket=bucket,
            query_mode=mode, target_count=target, min_support=min_support,
            track=track))
    return cells


def cells_summary(cells: List[Cell] | None = None) -> Dict[str, Any]:
    cells = cells or build_cells()
    tiers: Dict[str, int] = {}
    for cell in cells:
        tiers[cell.tier] = tiers.get(cell.tier, 0) + cell.target_count
    return {"n_cells": len(cells), "target_total": sum(c.target_count for c in cells),
            "tier_targets": tiers}
