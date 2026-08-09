# Pilot4 language / graph-leak audit

Source: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_profile_safe\canonical.jsonl` (read-only).

Pilot4 GOAL_BASED_IMPLICIT frequently discloses the dependency graph via 'The stages are related as follows' and per-stage derives-from clauses; these are graph-explicit, not goal-implicit.

## Train split

- n: 600
- stages_related_phrase_rate: 0.845
- mean_graph_edge_coverage: 1.3643
- call_count_disclosed_rate: 0.845
- high_or_complete_rate: 0.8833
- class dist: {"COMPLETE": 476, "HIGH": 54, "NONE": 26, "MEDIUM": 31, "LOW": 13}

## Selected set

- n: 1000
- stages_related_phrase_rate: 0.848
- mean_graph_edge_coverage: 1.3642
- high_or_complete_rate: 0.888
