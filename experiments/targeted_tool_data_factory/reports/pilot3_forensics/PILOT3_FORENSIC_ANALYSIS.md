# PILOT3_FORENSIC_ANALYSIS

## 1. Executive summary

### VERIFIED FACTS

- Matched C0-vLLM wins 277/500 (55.4%).
- Matched D1-vLLM wins 288/500 (57.6%).
- Paired flips: loss→win=27, win→loss=16.
- McNemar exact p=0.12628947438543037.
- Paired bootstrap 95% CI pp=[-0.40000000000000036, 4.799999999999994]
- Train-log dead_group_rate=0.51
- diagnostic-500 is balanced by call-count buckets (100 each for 2/3/4/5/6+).
- D1 subset identity vs local full train: status=INCONSISTENT overlap=62/300.

### SUPPORTED INTERPRETATIONS

- Matched-engine point estimate is small (+~2pp) and not statistically conclusive at conventional thresholds.
- Surface/schema mismatch is a candidate bottleneck because exact tool namespace overlap with diagnostic is low.
- Reward degeneracy is a candidate bottleneck given aggregate dead_group_rate≈0.5, but all-success vs all-fail split is not identifiable without rollouts.
- First-300 subset selection may be a data-selection bottleneck if cell distributions diverge from rest-300.

### OPEN HYPOTHESES

- D1 gains concentrate on coverage-friendly tasks while regressions are noise or path confound.
- Increasing topology novelty without improving distractor realism will not transfer.
- Paired A/G renderers will improve surface-invariant dependency skill.

### NOT IDENTIFIABLE

- Causal training effect magnitude after removing all inference-path confounds.
- Per-cell effective GRPO group rates for Pilot3 D1.
- Whether more data alone would help without composition changes.
- Semantic equivalence of proxy-mapped tools.
- Natural NESTFUL official win rate (diagnostic is not a natural sample).

## 2. Scope and non-goals

- Offline analysis only: no training, no model inference, no new NESTFUL eval, no new synthetic generation.
- Main contrast is C0-vLLM vs D1-vLLM on diagnostic-500.
- C0-HF is reported separately when present and is not mixed into the training-effect contrast.
- Goal: concrete Targeted Tool Data Factory changes, not a generic dashboard.

## 3. Input artifacts and integrity

- Pairing status: `VERIFIED`
- Pairing OK: `True`
- Eval manifest parity: `PARTIALLY_VERIFIED`
- Rollouts: `NOT_VERIFIABLE`
- See `INPUT_MANIFEST.json` and `INPUT_INTEGRITY.md`.

## 4. Reproduction of matched C0 vs D1 result

- Δ = **+2.20 pp** (matched-engine point estimate).
- Stratified bootstrap CI pp: `[-0.39999999999998925, 4.799999999999994]`
- Overall diagnostic win is a macro average across call-count buckets, not a natural NESTFUL estimate.
- Residual LoRA inference-path confound cannot be removed from these two trajectory sets alone.

## 5. What changed in trajectories

- Divergence category counts: `{'IDENTICAL_TEXT': 176, 'IDENTICAL_CALLS_DIFFERENT_TEXT': 173, 'REFERENCE_DIFFERENCE': 21, 'DIFFERENT_FIRST_TOOL': 35, 'TOOL_COUNT_DIFFERENCE': 68, 'DIFFERENT_LATER_TOOL': 13, 'SAME_TOOLS_DIFFERENT_ARGUMENTS': 14}`
- Mean first divergent turn (where defined): `0.3333333333333333`
- Details: `TRAJECTORY_PAIR_FEATURES.csv`, `TRAJECTORY_DIVERGENCE_SUMMARY.md`.

## 6. Gained vs lost task analysis

- n_gained=27, n_lost=16 (small n; avoid overclaiming significance).
- Top patterns: `[{'pattern': 'SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL', 'n_gained': 0, 'n_lost': 6, 'net_gain': -6, 'support': 6, 'effect_direction': 'loss', 'confidence': 'medium', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID', 'n_gained': 6, 'n_lost': 0, 'net_gain': 6, 'support': 6, 'effect_direction': 'gain', 'confidence': 'medium', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID', 'n_gained': 5, 'n_lost': 0, 'net_gain': 5, 'support': 5, 'effect_direction': 'gain', 'confidence': 'medium', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'FAIL_WRONG_FIRST_TOOL -> SUCCESS_OTHER_OFFICIAL', 'n_gained': 3, 'n_lost': 0, 'net_gain': 3, 'support': 3, 'effect_direction': 'gain', 'confidence': 'low', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'FAIL_NO_TOOL_CALL -> SUCCESS_ALTERNATIVE_VALID', 'n_gained': 3, 'n_lost': 0, 'net_gain': 3, 'support': 3, 'effect_direction': 'gain', 'confidence': 'low', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_TOOL_SEQUENCE', 'n_gained': 0, 'n_lost': 2, 'net_gain': -2, 'support': 2, 'effect_direction': 'loss', 'confidence': 'low', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTOR_ERROR', 'n_gained': 0, 'n_lost': 2, 'net_gain': -2, 'support': 2, 'effect_direction': 'loss', 'confidence': 'low', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}, {'pattern': 'FAIL_PARSE_INVALID -> SUCCESS_ALTERNATIVE_VALID', 'n_gained': 2, 'n_lost': 0, 'net_gain': 2, 'support': 2, 'effect_direction': 'gain', 'confidence': 'low', 'possible_mechanism': 'failure-class transition associated with flips (not causal)'}]`
- See `GAINED_LOST_AUDIT.md` and representative example markdowns.

## 7. Failure taxonomy

- Absolute counts in `FAILURE_ANALYSIS.md` / `FAILURE_TAXONOMY_PER_TASK.csv`.
- Primary category uses documented priority; secondary flags may co-occur.

## 8. Reward-signal observability and dead groups

- Aggregates: `{'dead_group_rate': 0.51, 'mean_reward': 0.79481, 'mean_reward_dense': 0.79481, 'mean_unique_rewards': 1.67, 'tasks_seen': 300, 'fallback_used': 0.0, 'kl_beta': 0.15, 'run_id': 'pilot3_D1_seed20260727_n300', 'reward_policy': 'A4_GATED_VERIFIABLE'}`
- Per-rollout available: `False`
- If per-rollout missing, dead-group composition and cell-level reward health are NOT IDENTIFIABLE.
- See `REWARD_AUDIT.md` and `MISSING_OBSERVABILITY.md`.

## 9. Train subset representativeness

- Shuffle interpretation: `likely_interleaved_or_shuffled`
- Missing generation cells in first300: `12`
- generation_cell JSD: `0.22221832288757049`
- See `TRAIN_SUBSET_SELECTION_AUDIT.md`.

## 10. Topology diversity and coverage

- Train300 summary: `{'n_programs': 300, 'n_unique_topology_hashes': 38, 'top1_share': 0.3, 'top5_share': 0.5766666666666667, 'top10_share': 0.7466666666666667, 'shannon_entropy': 4.046049565130658, 'effective_n_topologies': 16.51894408167903, 'singleton_rate': 0.21052631578947367, 'top_topologies': [{'topology_hash': 'topo_fdeb50f24873', 'count': 90, 'share': 0.3}, {'topology_hash': 'topo_911586a55488', 'count': 35, 'share': 0.11666666666666667}, {'topology_hash': 'topo_1c8dbe1830b2', 'count': 19, 'share': 0.06333333333333334}, {'topology_hash': 'topo_02b29308ce2b', 'count': 16, 'share': 0.05333333333333334}, {'topology_hash': 'topo_64b2717a77dd', 'count': 13, 'share': 0.043333333333333335}, {'topology_hash': 'topo_bd3dd51ec1a9', 'count': 11, 'share': 0.03666666666666667}, {'topology_hash': 'topo_d228dd6db98a', 'count': 11, 'share': 0.03666666666666667}, {'topology_hash': 'topo_5296967693d8', 'count': 11, 'share': 0.03666666666666667}, {'topology_hash': 'topo_30980484d6d7', 'count': 10, 'share': 0.03333333333333333}, {'topology_hash': 'topo_7279e5f4b24a', 'count': 8, 'share': 0.02666666666666667}, {'topology_hash': 'topo_926312b15db7', 'count': 6, 'share': 0.02}, {'topology_hash': 'topo_926c72bac680', 'count': 5, 'share': 0.016666666666666666}, {'topology_hash': 'topo_9e01698159b0', 'count': 5, 'share': 0.016666666666666666}, {'topology_hash': 'topo_8c144cdbfcfb', 'count': 5, 'share': 0.016666666666666666}, {'topology_hash': 'topo_5b37b75596d2', 'count': 5, 'share': 0.016666666666666666}, {'topology_hash': 'topo_63f405a71afa', 'count': 5, 'share': 0.016666666666666666}, {'topology_hash': 'topo_33e36888cf10', 'count': 4, 'share': 0.013333333333333334}, {'topology_hash': 'topo_bf579ac7c4b1', 'count': 3, 'share': 0.01}, {'topology_hash': 'topo_2b9a65f092bb', 'count': 3, 'share': 0.01}, {'topology_hash': 'topo_ff8ead63c33b', 'count': 3, 'share': 0.01}]}`
- Coverage vs diagnostic: `{'train_unique': 38, 'diagnostic_unique': 93, 'diagnostic_exact_topology_coverage_rate': 0.76, 'diagnostic_unseen_topology_rate': 0.24, 'n_diagnostic_covered': 380, 'n_diagnostic': 500}`
- Topology hash is call-order-indexed shape hash (see methodology limitations).

## 11. Surface, schema and reference mismatch

- Namespace overlap: `{'n_train': 132, 'n_diagnostic': 816, 'exact_overlap': 11, 'exact_overlap_rate_vs_diag': 0.013480392156862746, 'normalized_overlap': 11, 'normalized_overlap_rate_vs_diag': 0.013480392156862746, 'exact_names': ['add', 'divide', 'floor', 'inverse', 'is_within_range', 'multiply', 'negate', 'power', 'reminder', 'sqrt', 'subtract'], 'diag_only_exact': ['access_attrs', 'activation_function', 'add_1', 'add_duration', 'add_feature', 'add_id_to_dicts', 'add_list_elements', 'add_or_divide', 'add_tab', 'add_tuples_to_dict', 'add_up', 'align_left', 'all_even', 'analyze_sentiment', 'append_path_components', 'are_overlapping', 'area_of_triangle_from_sides', 'arithmetic_list', 'ascii_to_integer_list', 'authenticate_token', 'average_colors', 'average_of_top_k', 'average_value', 'beta_value', 'binary_classification_accuracy', 'bit_not_inverse', 'bit_representation', 'bits_to_num', 'bitwise_left_shift', 'bitwise_reorder', 'bool_indicator', 'both_positive_and_negative', 'build_facebook_url', 'build_file_path', 'build_span_string', 'buy_cookies', 'byte_string_to_bit_string', 'bytearray_to_string', 'calc_struct_area', 'calc_vector_distances', 'calculate_ab_error', 'calculate_average', 'calculate_average_except_zeros_and_negatives', 'calculate_average_speed', 'calculate_ber', 'calculate_bpm', 'calculate_derivatives', 'calculate_diameter', 'calculate_elastic_modulus', 'calculate_error']}`
- Distractor hardness: `{'train_mean': 0.031226499999999983, 'diag_mean': 0.0015219999999999997}`
- Overlaps are lexical/schema proxies, not semantic equivalence.

## 12. Registry coverage

- By-outcome preview: `[{'bucket': 'exact_full', 'n': 412, 'c0_wins': 236, 'd1_wins': 244, 'c0_win_rate': 0.5728155339805825, 'd1_win_rate': 0.5922330097087378, 'net_gain': 8, 'gained': 22, 'lost': 14, 'mean_ood': 0.05291262135922319}, {'bucket': 'exact_low', 'n': 3, 'c0_wins': 2, 'd1_wins': 2, 'c0_win_rate': 0.6666666666666666, 'd1_win_rate': 0.6666666666666666, 'net_gain': 0, 'gained': 0, 'lost': 0, 'mean_ood': 0.6222}, {'bucket': 'exact_none', 'n': 76, 'c0_wins': 36, 'd1_wins': 37, 'c0_win_rate': 0.47368421052631576, 'd1_win_rate': 0.4868421052631579, 'net_gain': 1, 'gained': 3, 'lost': 2, 'mean_ood': 0.6830039473684206}, {'bucket': 'exact_partial', 'n': 9, 'c0_wins': 3, 'd1_wins': 5, 'c0_win_rate': 0.3333333333333333, 'd1_win_rate': 0.5555555555555556, 'net_gain': 2, 'gained': 2, 'lost': 0, 'mean_ood': 0.32217777777777773}, {'bucket': 'ood_high', 'n': 70, 'c0_wins': 33, 'd1_wins': 35, 'c0_win_rate': 0.4714285714285714, 'd1_win_rate': 0.5, 'net_gain': 2, 'gained': 3, 'lost': 1, 'mean_ood': 0.711355714285714}, {'bucket': 'ood_low', 'n': 415, 'c0_wins': 236, 'd1_wins': 244, 'c0_win_rate': 0.5686746987951807, 'd1_win_rate': 0.5879518072289157, 'net_gain': 8, 'gained': 22, 'lost': 14, 'mean_ood': 0.05378313253012037}]`
- Proxy mappings labeled EXACT/HIGH_PROXY/MEDIUM_PROXY/LOW_PROXY/UNMAPPED.

## 13. Joint-distribution and OOD analysis

- Summary: `{'n_train': 300, 'n_diagnostic': 500, 'n_joint_cells_train': 58, 'n_joint_cells_diagnostic': 104, 'unseen_combination_rate': 1.0, 'rare_combination_rate': 1.0, 'by_ood_decile': [{'decile': 0, 'n': 43, 'c0_win_rate': 0.6046511627906976, 'd1_win_rate': 0.6046511627906976, 'net_gain': 0}, {'decile': 1, 'n': 57, 'c0_win_rate': 0.5789473684210527, 'd1_win_rate': 0.5614035087719298, 'net_gain': -1}, {'decile': 2, 'n': 43, 'c0_win_rate': 0.6976744186046512, 'd1_win_rate': 0.7209302325581395, 'net_gain': 1}, {'decile': 3, 'n': 53, 'c0_win_rate': 0.4716981132075472, 'd1_win_rate': 0.49056603773584906, 'net_gain': 1}, {'decile': 4, 'n': 53, 'c0_win_rate': 0.5471698113207547, 'd1_win_rate': 0.5849056603773585, 'net_gain': 2}, {'decile': 5, 'n': 22, 'c0_win_rate': 0.36363636363636365, 'd1_win_rate': 0.5, 'net_gain': 3}, {'decile': 6, 'n': 53, 'c0_win_rate': 0.660377358490566, 'd1_win_rate': 0.660377358490566, 'net_gain': 0}, {'decile': 7, 'n': 76, 'c0_win_rate': 0.5131578947368421, 'd1_win_rate': 0.5394736842105263, 'net_gain': 2}, {'decile': 8, 'n': 48, 'c0_win_rate': 0.6875, 'd1_win_rate': 0.6666666666666666, 'net_gain': -1}, {'decile': 9, 'n': 52, 'c0_win_rate': 0.36538461538461536, 'd1_win_rate': 0.4423076923076923, 'net_gain': 4}]}`
- Distance is transparent Gower-like mixed distance; models are associative only.

## 14. Data quality and shortcut risks

- Quality summary: `{'n': 600, 'n_unique_skeletons': 600, 'top1_skeleton_share': 0.0016666666666666668, 'n_unique_tool_combos': 585, 'top1_tool_combo_share': 0.006666666666666667, 'n_unique_program_hashes': 585, 'answer_leak_rate': 0.0, 'constants_in_question_rate': 0.9766666666666667, 'mean_gold_tool_offered_position': 5.0930376984127035, 'note': 'Lexical/template proxies only; not proof of reward hacking.'}`

## 15. Ranked bottlenecks

- **#1 `EVAL_PROTOCOL`** — SUPPORTED / HIGH: Original +8.8pp used mismatched HF vs vLLM backends; matched contrast is smaller.
- **#2 `TRAIN_SUBSET_SELECTION`** — SUPPORTED / HIGH: D1 train_subset_300 is not identical to the local train_grpo_pilot3 prefix/freeze; selection provenance is broken or the export was regenerated.
- **#3 `SURFACE_SCHEMA_MISMATCH`** — PARTIALLY_SUPPORTED / HIGH: Diagnostic gold tools largely outside factory exact namespace; transfer relies on schema/lexical proxies.
- **#4 `REWARD_SIGNAL`** — PARTIALLY_SUPPORTED / MEDIUM: Aggregate dead_group_rate=0.51 and low unique rewards imply many non-informative GRPO groups.
- **#5 `LORA_INFERENCE_PATH`** — NOT_IDENTIFIABLE / LOW: D1 uses adapter; C0 is base. Path differences can shift decoding even at T=0.
- **#6 `TOPOLOGY_DIVERSITY`** — PARTIALLY_SUPPORTED / MEDIUM: High top-1 topology share and/or high diagnostic unseen-topology rate indicate shape mismatch risk.
- **#7 `REGISTRY_SEMANTIC_COVERAGE`** — PARTIALLY_SUPPORTED / MEDIUM: Unmapped diagnostic gold tools on critical path associate with persistent failures (proxy).
- **#8 `JOINT_DISTRIBUTION_MISMATCH`** — PARTIALLY_SUPPORTED / MEDIUM: Margin match can hide unseen joint cells (topology×calls×answer×track).
- **#9 `DISTRACTOR_REALISM`** — PARTIALLY_SUPPORTED / MEDIUM: If train distractors are lexically far / type-impossible, model learns weak discrimination.
- **#10 `DATA_SCALE`** — PARTIALLY_SUPPORTED / MEDIUM: n=300 with 51% dead groups yields few effective updates; underpowered for +2pp.
- **#11 `REFERENCE_SYNTAX_MISMATCH`** — PARTIALLY_SUPPORTED / MEDIUM: Train and diagnostic may differ in $var vs $var_ and output key conventions.
- **#12 `OTHER`** — PARTIALLY_SUPPORTED / LOW: Template concentration / shortcut cues can inflate train reward without transfer.

## 16. Concrete changes to data generation

- `REC_001` [PROFILE_SAFE/P0]: increase unique topology hashes; cap top-1 family share below 10% — increase unique topologies ≥2×; cap top-1 topology share ≤10%
- `REC_002` [PROFILE_SAFE/P0]: fan_in and reuse (max_outdegree>=2) minimum floor — increase fan-in/reuse cells 1.5–2× vs current linear share
- `REC_003` [PROFILE_SAFE/P0]: any — reserve 20–30% cells for mid-difficulty; reduce trivially saturated cells
- `REC_004` [PROFILE_SAFE/P1]: any — ≥50% of A-track with NESTFUL-like keys
- `REC_005` [DIAGNOSTIC_INFORMED_EXPLORATORY/P1]: match high-frequency diagnostic unseen topology classes (abstract shapes only) — allocate 15–25% exploratory mass to top unmet joint cells
- `REC_006` [PROFILE_SAFE/P1]: keep short programs but non-trivial distractors — cap easy 2-call all-success-prone cells; keep discrimination-focused 2/3-call

## 17. Proposed generation cells

- Machine-readable: `RECOMMENDED_GENERATION_CELLS.json` / `.csv`.
- PROFILE_SAFE vs DIAGNOSTIC_INFORMED_EXPLORATORY are separated; the latter carries an explicit contamination disclaimer.

## 18. What can be concluded

- Matched-engine D1−C0 delta is small and not statistically conclusive.
- Multiple measurable factory mismatches exist (surface, joint cells, possible subset bias, reward aggregates).
- Next data generation should prioritize composition/constraints over naive scale-up.

## 19. What cannot be concluded

- Causal training effect magnitude after removing all inference-path confounds.
- Per-cell effective GRPO group rates for Pilot3 D1.
- Whether more data alone would help without composition changes.
- Semantic equivalence of proxy-mapped tools.
- Natural NESTFUL official win rate (diagnostic is not a natural sample).

## 20. Missing artifacts and future logging

- Pilot3 per-rollout reward groups / canary rollouts for D1.
- Multi-seed D1 runs.
- Null-LoRA matched inference control.
- See `MISSING_OBSERVABILITY.md`.

