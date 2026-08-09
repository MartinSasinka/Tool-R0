# Pilot4 vs Pilot4.1 audit

Claim classes separate offline verified improvements from training/eval-only claims.

- **stages_related_rate**: p4=0.845 → p41=0.0 [VERIFIED_IMPROVEMENT]
- **mean_graph_edge_coverage**: p4=1.3643 → p41=0.0 [VERIFIED_IMPROVEMENT]
- **high_or_complete_graph_leak_rate**: p4=0.8833 → p41=0.0 [VERIFIED_IMPROVEMENT]
- **top1_skeleton_share**: p4=0.0017 → p41=0.001 [VERIFIED_IMPROVEMENT]
- **singleton_cell_rate**: p4=0.3205 → p41=0.1029 [VERIFIED_IMPROVEMENT]
- **n_topologies**: p4=139 → p41=37 [PROFILE_TRADEOFF]
- **mean_cell_support**: p4=3.85 → p41=14.71 [VERIFIED_IMPROVEMENT]
- **call_count_disclosed_rate**: p4=0.845 → p41=0.0 [VERIFIED_IMPROVEMENT]
- **call_count_tv_to_dev**: p4=1.0 → p41=1.0 [PROFILE_TRADEOFF]
