"""Pilot4.2 implementation and data-quality reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def write_reports(out_dir: Path, metrics: Dict[str, Any],
                  selected: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = [
        "# PILOT42_DATA_QUALITY_REPORT", "",
        f"- semantic candidates: **{metrics.get('semantic_candidates', 0)}**",
        f"- hard validated: **{metrics.get('hard_validated', 0)}**",
        f"- V4 safe: **{metrics.get('v4_safe', 0)}**",
        f"- selected: **{len(selected)}**",
        f"- all hard constraints met: **{metrics.get('selection_all_hard_constraints_met', False)}**",
    ]
    (out_dir / "PILOT42_DATA_QUALITY_REPORT.md").write_text(
        "\n".join(quality) + "\n", encoding="utf-8")
    (out_dir / "PILOT42_IMPLEMENTATION_REPORT.md").write_text(
        "# PILOT42_IMPLEMENTATION_REPORT\n\n"
        "WorkflowBlueprint → WorkflowInstance → SemanticComputationPlan → "
        "PrimitiveBinding → typed DAG → oracle → query contract → hard gates → V4 → selection.\n",
        encoding="utf-8")
