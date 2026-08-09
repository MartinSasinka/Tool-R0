"""Pilot4.1 versus Pilot4.2 architectural audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def write_comparison(path: Path, metrics: Dict[str, Any] | None = None) -> Path:
    metrics = metrics or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# PILOT41_VS_PILOT42_AUDIT\n\n"
        "- Pilot4.1 attaches workflow labels after random DAG generation.\n"
        "- Pilot4.2 generates every instance and executable plan from a WorkflowBlueprint.\n"
        "- Pilot4.2 treats semantic alignment, replay, necessity, query checks, and V4 as "
        "hard pre-selection gates.\n"
        f"- Pilot4.2 selected rows: **{metrics.get('selected', 0)}**.\n",
        encoding="utf-8")
    return path
