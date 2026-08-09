"""Pilot4 data factory: structural, linguistic and schema diversity.

Layered on top of the pilot3 factory rather than replacing it. The pipeline is
split into three explicit layers so difficulty can be varied along one axis at
a time:

    SemanticProgram  ->  QueryRenderer  ->  ToolSurfaceRenderer

The semantic program owns the typed DAG, the constants and the oracle; it holds
no natural language and no surface tool names.
"""
from __future__ import annotations

PILOT4_VERSION = "ttdf-pilot4-0.1.0"
SCHEMA_VERSION = "ttdf.pilot4.task.v1"
