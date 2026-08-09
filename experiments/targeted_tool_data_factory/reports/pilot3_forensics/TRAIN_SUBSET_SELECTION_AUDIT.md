# TRAIN_SUBSET_SELECTION_AUDIT

- subset_identity_status: `INCONSISTENT`
- positional_match: False
- overlap with local full train: 62 / 300
- shuffle_interpretation: likely_interleaved_or_shuffled
- missing cells in compared-A: 12
- identity_warning: D1 train_subset_300.jsonl is NOT the first 300 rows of the local train_grpo_pilot3.jsonl freeze; overlap is far below 300. Local export may have been regenerated after the RunPod subset was frozen, or the subset was sliced from a different artifact. Topology/surface audits that use local train600 therefore only partially represent the true D1 training distribution.
- categorical: `{'generation_cell': {'tv': 0.39627053603549595, 'jsd': 0.22221832288757049, 'n_levels_first': 26, 'n_levels_rest': 38}, 'track': {'tv': 0.08058520206259745, 'jsd': 0.00469071371589937, 'n_levels_first': 2, 'n_levels_rest': 2}, 'call_bucket': {'tv': 0.15583403285765682, 'jsd': 0.03727616032879552, 'n_levels_first': 5, 'n_levels_rest': 5}, 'motif': {'tv': 0.16932485909581485, 'jsd': 0.025315310206347842, 'n_levels_first': 3, 'n_levels_rest': 3}, 'target_skill': {'tv': 0.2127353399688212, 'jsd': 0.035121328815511055, 'n_levels_first': 5, 'n_levels_rest': 5}, 'target_failure_mode': {'tv': 0.2127353399688212, 'jsd': 0.035121328815511055, 'n_levels_first': 5, 'n_levels_rest': 5}, 'semantic_program_family': {'tv': 0.990706319702593, 'jsd': 0.9801582796834518, 'n_levels_first': 62, 'n_levels_rest': 521}, 'graph_template_id': {'tv': 0.7106367669984384, 'jsd': 0.5840613588789012, 'n_levels_first': 32, 'n_levels_rest': 225}, 'answer_type': {'tv': 0.0665547427749131, 'jsd': 0.006761772597732956, 'n_levels_first': 6, 'n_levels_rest': 6}, 'paraphrase_status': {'tv': 0.0, 'jsd': 0.0, 'n_levels_first': 1, 'n_levels_rest': 1}}`
