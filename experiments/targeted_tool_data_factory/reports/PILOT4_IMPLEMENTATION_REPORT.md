# Pilot4 implementation report

- commit: `d174486ff105fc5b3daed71bdfb59ff572177fe5` (dirty: True, 32 files)
- generated: 2026-07-30T23:36:42.784544+00:00
- python 3.11.3 on Windows-10-10.0.26200-SP0
- schema: `ttdf.pilot4_implementation_report.v1`

## 1. Executive summary

**VERIFIED (from artifacts)**

- Pilot3 provenance is resolved: status `EXACT_FIRST_300_BYTES`, 300/300 subset rows matched inside the first 300 lines of the 600-row parent export.
- Subset is byte-identical to the parent's first 300 lines. Provenance fully verified.
- The earlier 62/300 sample-ID finding was an artifact of comparing a regenerated working-tree export against the trained one; the retraction is recorded in the provenance audit (`retracts_previous_claim`: True).
- Pilot3 training questions leak the plan: measured plan-leak rate d1_train_subset_300 0.6533, pilot3_train_600_as_trained 0.6417, pilot3_train_600_worktree 0.6717, pilot3_heldout_200 0.605, pilot3_reserve_200 0.615, nestful_dev_200 0.045.

**IMPLEMENTED**

- TargetProfile v2 with 12 conditional distributions, graph and surface feature blocks, plus derived per-bucket topology constraints.
- Capability taxonomy over 89 primitives in 25/25 families.
- 15 structural pattern families and 10 composable graph transformations, all DAG- and execution-checked.
- SemanticProgram / QueryRenderer / ToolSurfaceRenderer are separate layers: 3 query modes x 2 surface tracks, with paired renderings of one program kept in one split.
- Schema-semantic distractors with V8 validity checking, V7 plan-leak validation, per-task difficulty signatures, multi-objective selection with hard constraints, four samplers, and training/eval logging schemas.

**GENERATED**

- `pilot4_profile_safe`: 5045 candidates, 4523 validated (pass rate 0.8965), 1000 selected, split train 600, heldout 200, reserve 200.
- The split is family-safe: leakage {'semantic_program_id': 0, 'program_family_id': 0, 'paired_with': 0}, leak_free True.
- Re-running generation on the same commit and seed reproduces every data artifact byte for byte; only `freeze_manifest.json` differs, because it records the wall-clock time and the output path.

**NOT TESTED BY TRAINING**

- The adaptive samplers, the per-rollout/group/step logs and the sampler state checkpointing have unit tests and an offline simulation, but no GRPO step has been run with them.
- Whether pilot4 changes dead-group rate in a real run is unknown.

**NOT TESTED BY NESTFUL EVAL**

- No claim is made that pilot4 improves NESTFUL accuracy. The evaluation logging changes are schema-tested only; no evaluation was executed.

**OPEN**

- V4 (bounded minimal-path shortcut search) is disabled in the default run because it dominates runtime; V1-V3 and V5-V8 gate every selected task and the run can be repeated with --run-v4.
- Fan-out and output reuse are generated even though dev-200 shows none of them. That is deliberate coverage of the structures pilot3 lacked entirely, and it is a known distribution-mismatch risk rather than a profile match.
- At two and three calls the topology space is exhausted by 1 and 3 shapes, so per-bucket diversity there cannot be improved further; the top-1 share is pinned by the join rate the profile asks for.

## 2. Files changed

Modified (7):

- `experiments/nestful_mtgrpo_minimal/grpo_train.py`
- `experiments/targeted_tool_data_factory/runpod_bundle_pilot3/eval_nestful500_sharded.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/cli.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/registry/__init__.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/validation/__init__.py`
- `experiments/targeted_tool_data_factory/tests/test_pilot2.py`
- `experiments/targeted_tool_data_factory/tests/test_validation.py`

Added (25):

- `experiments/__init__.py`
- `experiments/nestful_mtgrpo_minimal/train_observability.py`
- `experiments/targeted_tool_data_factory/__init__.py`
- `experiments/targeted_tool_data_factory/analysis/`
- `experiments/targeted_tool_data_factory/reports/PILOT4_IMPLEMENTATION_REPORT.json`
- `experiments/targeted_tool_data_factory/reports/PILOT4_IMPLEMENTATION_REPORT.md`
- `experiments/targeted_tool_data_factory/reports/capability/`
- `experiments/targeted_tool_data_factory/reports/pilot3_forensics/`
- `experiments/targeted_tool_data_factory/reports/pilot3_provenance/`
- `experiments/targeted_tool_data_factory/reports/pilot3_vs_pilot4/`
- `experiments/targeted_tool_data_factory/reports/profile_v2/`
- `experiments/targeted_tool_data_factory/reports/query_realism/`
- `experiments/targeted_tool_data_factory/reports/sampler_simulation/`
- `experiments/targeted_tool_data_factory/runpod_bundle_pilot3/eval_observability.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/capability.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/observability/`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/pilot4/`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/pilot4_cli.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/profile_v2.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/provenance.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/query_realism.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/registry/extensions.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/repro.py`
- `experiments/targeted_tool_data_factory/src/targeted_tool_data/sampling/`
- `experiments/targeted_tool_data_factory/tests/test_pilot4.py`

## 3. Pilot3 provenance resolution

| field | value |
|---|---|
| status | EXACT_FIRST_300_BYTES |
| parent rows | 600 |
| subset rows | 300 |
| canonical matches | 300 |
| matches inside parent prefix | 300 |
| matches form the identity prefix | True |
| parent candidates searched | 15 |
| resolved | True |
| retracts previous claim | True |

Multi-key overlap (each key counted separately):

| key | overlap |
|---|---|
| sample_id | {"subset_defined": 300, "overlap_with_parent_any": 300, "overlap_with_parent_first_n": 300, "parent_unique": 600, "subset_unique": 300} |
| task_id | {"subset_defined": 0, "overlap_with_parent_any": 0, "overlap_with_parent_first_n": 0, "parent_unique": 0, "subset_unique": 0} |
| semantic_program_id | {"subset_defined": 300, "overlap_with_parent_any": 295, "overlap_with_parent_first_n": 295, "parent_unique": 583, "subset_unique": 295} |
| semantic_program_family | {"subset_defined": 300, "overlap_with_parent_any": 295, "overlap_with_parent_first_n": 295, "parent_unique": 583, "subset_unique": 295} |
| graph_template_id | {"subset_defined": 300, "overlap_with_parent_any": 143, "overlap_with_parent_first_n": 143, "parent_unique": 228, "subset_unique": 143} |
| generation_cell_id | {"subset_defined": 300, "overlap_with_parent_any": 36, "overlap_with_parent_first_n": 36, "parent_unique": 38, "subset_unique": 36} |

Byte-level comparison of the parent's first 300 lines against the subset:

| field | value |
|---|---|
| n_lines_taken | 300 |
| parent_prefix_sha256 | 8b0a16a39a1b5815ae768ac4cd57a9941114426c91465e28783dfad941939c69 |
| subset_sha256 | 8b0a16a39a1b5815ae768ac4cd57a9941114426c91465e28783dfad941939c69 |
| exact_bytes_match | True |
| match_after_trailing_newline_normalization | False |
| note |  |

## 4. Query-realism findings

Operation lexicon version `ttdf.operation_lexicon.v1`, hand-auditable and independent of diagnostic-500.

| dataset | n | plan_leak_rate | exact_op_cov | lexical_op_cov | sequence_leakage | procedural_cues |
|---|---|---|---|---|---|---|
| d1_train_subset_300 | 300 | 0.6533 | 0.7618 | 0.8521 | 0.8129 | 6.5233 |
| pilot3_train_600_as_trained | 600 | 0.6417 | 0.7566 | 0.8378 | 0.7922 | 6.6833 |
| pilot3_train_600_worktree | 600 | 0.6717 | 0.7596 | 0.845 | 0.812 | 6.3967 |
| pilot3_heldout_200 | 200 | 0.605 | 0.6976 | 0.8284 | 0.8121 | 6.49 |
| pilot3_reserve_200 | 200 | 0.615 | 0.7211 | 0.8138 | 0.7543 | 6.27 |
| nestful_dev_200 | 200 | 0.045 | 0.0653 | 0.1645 | 0.152 | 0.645 |

Query-mode distribution per dataset:

- `d1_train_subset_300`: {"GOAL_BASED_IMPLICIT": 0.0067, "PROCEDURAL_EXPLICIT": 0.6533, "PROCEDURAL_PARTIAL": 0.33, "SEMI_IMPLICIT": 0.0067, "UNCLASSIFIED": 0.0033}
- `pilot3_train_600_as_trained`: {"GOAL_BASED_IMPLICIT": 0.01, "PROCEDURAL_EXPLICIT": 0.6417, "PROCEDURAL_PARTIAL": 0.3267, "SEMI_IMPLICIT": 0.0133, "UNCLASSIFIED": 0.0083}
- `pilot3_train_600_worktree`: {"GOAL_BASED_IMPLICIT": 0.01, "PROCEDURAL_EXPLICIT": 0.67, "PROCEDURAL_PARTIAL": 0.3017, "SEMI_IMPLICIT": 0.0133, "UNCLASSIFIED": 0.005}
- `pilot3_heldout_200`: {"GOAL_BASED_IMPLICIT": 0.005, "PROCEDURAL_EXPLICIT": 0.61, "PROCEDURAL_PARTIAL": 0.355, "SEMI_IMPLICIT": 0.025, "UNCLASSIFIED": 0.005}
- `pilot3_reserve_200`: {"GOAL_BASED_IMPLICIT": 0.02, "PROCEDURAL_EXPLICIT": 0.635, "PROCEDURAL_PARTIAL": 0.32, "SEMI_IMPLICIT": 0.015, "UNCLASSIFIED": 0.01}
- `nestful_dev_200`: {"GOAL_BASED_IMPLICIT": 0.7, "PROCEDURAL_EXPLICIT": 0.035, "PROCEDURAL_PARTIAL": 0.125, "SEMI_IMPLICIT": 0.115, "UNCLASSIFIED": 0.025}

## 5. TargetProfile v2

Source `nestful_dev_200` (200 rows), mode `PROFILE_SAFE`. Conditional distributions: `P(answer_type|call_count)`, `P(depth|call_count)`, `P(fan_out_count|call_count)`, `P(join_count|call_count)`, `P(motif|call_count)`, `P(offered_tool_count|call_count)`, `P(operation_explicitness|query_mode)`, `P(query_mode|call_count)`, `P(reference_density|call_count)`, `P(reuse_count|call_count)`, `P(schema_complexity|call_count)`, `P(sequence_leakage|query_mode)`.

Measured per-bucket topology diversity and the constraints derived from it:

| bucket | n | distinct topologies | top1 share | join rate | derived constraints |
|---|---|---|---|---|---|
| 2 | 66 | 1 | 1.0 | 0.0 | {"allowed_patterns": ["LINEAR_CHAIN"], "minimum_pattern_families": 1} |
| 3 | 44 | 4 | 0.6818 | 0.2273 | {"maximum_top1_topology_share": 0.6, "minimum_join_rate": 0.136, "minimum_pattern_families": 3} |
| 4 | 27 | 2 | 0.6296 | 0.6296 | {"maximum_top1_topology_share": 0.6, "minimum_join_rate": 0.378, "minimum_pattern_families": 3} |
| 5 | 19 | 4 | 0.3158 | 0.7895 | {"maximum_top1_topology_share": 0.379, "minimum_join_rate": 0.474, "minimum_multi_join_rate": 0.158, "minimum_pattern_families": 3, "minimum_reuse_rate": 0.15} |
| 6+ | 44 | 26 | 0.1364 | 1.0 | {"maximum_top1_topology_share": 0.18, "minimum_fan_out_rate": 0.2, "minimum_join_rate": 0.6, "minimum_late_reference_rate": 0.6, "minimum_multi_join_rate": 0.273, "minimum_pattern_families": 10, "minimum_reuse_rate": 0.15} |

## 6. Capability registry changes

- 89 primitives, 25/25 declared capability families populated.
- empty families: []
- primitives outside the taxonomy: []
- registry validation errors: 0

## 7. New structural patterns

Pattern families: `LINEAR_CHAIN`, `FAN_IN_SINGLE`, `FAN_IN_MULTIPLE`, `FAN_OUT`, `DIAMOND`, `PARALLEL_THEN_MERGE`, `REUSE_EARLY_OUTPUT`, `LATE_REFERENCE`, `TWO_STAGE_AGGREGATION`, `MULTI_JOIN`, `ALTERNATING_BRANCH_CHAIN`, `MIXED_INDEPENDENT_DEPENDENT`, `REPEATED_PRIMITIVE`, `TYPE_TRANSITION_CHAIN`, `NESTED_AGGREGATION`.

Transformations: `INSERT_NODE_ON_EDGE`, `SPLIT_BRANCH`, `MERGE_BRANCHES`, `REUSE_OUTPUT`, `ADD_PARALLEL_BRANCH`, `ADD_LATE_JOIN`, `ADD_SECOND_JOIN`, `REPEAT_PRIMITIVE_WITH_NEW_ARGS`, `CHANGE_TYPE_PATH`, `EXTEND_CRITICAL_PATH`.

Topologies present in the selected set against the number that can exist at all (blank where the space is too large to enumerate):

| bucket | topologies in selected | topologies that exist |
|---|---|---|
| 2 | 1 | 1 |
| 3 | 3 | 3 |
| 4 | 12 | 21 |
| 5 | 34 | 315 |
| 6+ | 167 |  |

## 8. Query renderer changes

Modes: `GOAL_BASED_IMPLICIT`, `PROCEDURAL_EXPLICIT`, `SEMI_IMPLICIT`. Share of the selected set by the mode that was rendered: {"GOAL_BASED_IMPLICIT": 0.602, "PROCEDURAL_EXPLICIT": 0.12, "SEMI_IMPLICIT": 0.278}; by the mode the audit classifier reads back from the question: {"GOAL_BASED_IMPLICIT": 0.646, "PROCEDURAL_EXPLICIT": 0.091, "PROCEDURAL_PARTIAL": 0.121, "SEMI_IMPLICIT": 0.086, "UNCLASSIFIED": 0.056}.

V7 keeps 0.8335 of validated candidates inside the leakage bucket their query mode allows; explicit tasks are kept but quota-limited rather than discarded.

## 9. Surface renderer changes

Tracks: `A_NATIVE`, `G_GENERAL`. Selected shares: {"A_NATIVE": 0.498, "G_GENERAL": 0.502}.
Paired renderings of the same semantic program: 575 records.

## 10. Distractor changes

Levels: `EASY_TYPE_INCOMPATIBLE`, `MEDIUM_SAME_OUTPUT_TYPE`, `HARD_SAME_ARITY_AND_TYPES`, `HARD_SAME_CAPABILITY_FAMILY`, `HARD_SEMANTIC_NEIGHBOR`, `HARD_REPEATED_SURFACE_AMBIGUITY`.
Schema-compatible distractor share in the selected set: 0.955; mean hard distractors per task: 4.047.

## 11. Validation V7-V8

| field | value |
|---|---|
| candidates | 5045 |
| validated | 4523 |
| pass rate | 0.8965 |
| V7 in target bucket | 0.8335 |
| V8 pass rate | 1.0 |
| V4 | skipped: reported as a known limitation |

Per-layer failures: {"V3": 522}

## 12. Selection v2

| constraint | requested | achieved | abs deficit | rel deficit | met | reason not met |
|---|---|---|---|---|---|---|
| call_bucket_share[2] | 0.33 | 0.316 | 0.014 | 0.0424 | True |  |
| call_bucket_share[3] | 0.22 | 0.214 | 0.006 | 0.0273 | True |  |
| call_bucket_share[4] | 0.135 | 0.136 | 0.0 | 0.0 | True |  |
| call_bucket_share[5] | 0.095 | 0.101 | 0.0 | 0.0 | True |  |
| call_bucket_share[6+] | 0.22 | 0.233 | 0.0 | 0.0 | True |  |
| min_goal_based_share | 0.35 | 0.646 | 0.0 | 0.0 | True |  |
| min_g_general_share | 0.4 | 0.502 | 0.0 | 0.0 | True |  |
| min_schema_compatible_distractor_share | 0.6 | 0.955 | 0.0 | 0.0 | True |  |
| min_topology_diversity_5call | 6 | 34 | 0.0 | 0.0 | True |  |
| min_topology_diversity_6plus | 10 | 167 | 0.0 | 0.0 | True |  |

All hard constraints met: True. Candidate rejections by constraint: {"call_bucket_cap[5]": 15464, "call_bucket_cap[6+]": 67617}.

## 13. Pilot4 dataset composition

- cells: 204
- counts: {"candidates": 5045, "heldout": 200, "reserve": 200, "selected": 1000, "train": 600, "validated": 4523}
- splits: {"heldout": 200, "reserve": 200, "train": 600}
- call-bucket share (selected): {"2": 0.316, "3": 0.214, "4": 0.136, "5": 0.101, "6+": 0.233}
- call-bucket share (train): {"2": 0.3133, "3": 0.2267, "4": 0.1433, "5": 0.1, "6+": 0.2167}
- difficulty bands: {"easy": 0.138, "hard": 0.415, "medium": 0.447}
- ordered sample-ID hash: `4c380e1ea38d`
- deficits: {}

Artifact hashes:

| file | sha256 |
|---|---|
| candidates.jsonl | 62ba72b377f398ccae74bcfff7e3448c3805dfb56cb3eaa7b8ef7c70d52d7912 |
| canonical.jsonl | a5ede4b138653fa7ffd71e924a482ab4cd29e75d15e805590b38e0c951353cfa |
| freeze_manifest.json | 0e1214d47dae79c27b275efb4654c0634ea2c917e635d8d1716c5c44c8e3377b |
| generation_cells.json | 78d7fded405e66b1ecf91dfc4c32a55b7e05c7d03aec54e23eb405df7a3e4d42 |
| heldout.jsonl | f9803934f72816560cfe642daa411d7b6cae8b19c7d35b470f235360cb550fcc |
| nestful_compat.jsonl | 2f355d981bb223ff9cc8712ad10a885f3c068e8df0c696b86f9c208411f8facf |
| pilot4_tasks.csv | f39bc46c61c451b1505cca43961f25dfcdb380d1f484d439a88d4af2e0d828ec |
| reserve.jsonl | 9a4789547d4671dd7f91ee56cbc5dead8fd18c6f9e8f30f0ebf2754c7e01ca34 |
| selected.jsonl | 7db469618e844e023853fdbcb9bd44d8e7a0d44db94ce12f970c135afa4423d4 |
| selection_report.json | b3e522b43b82491318751ab5544ff0cbed55889932066f1579151f0a6ef5e3b8 |
| split_manifest.json | d380fa73ddf8694cc25f0c5c6b46283a50b5e25f1b6754bcfbb9bd7476fc07bb |
| target_profile_v2.json | db0fbe29a5d44f0b73e02345196fa337437368637b1c6f4af928d51959cf015b |
| topology_constraints.json | 630477e48ed3fbea559b50572bc5e86197eafdef63c88a79073c8025b75e9e0c |
| train.jsonl | bc8ed06fd131ee83cb8c2257df196bd29f3237e6457b52e20052eb89e43b7193 |
| validated.jsonl | 874aeb52c5728bc787ce5f941f7ecd7987f9b6440f456ee91f3a237921987dfa |
| validation_report.json | ea790e66561e47a43347d1b6dce238c88568c10cd6e0c4e5d673d166f8617df5 |

## 14. Pilot3 vs Pilot4 offline comparison

75 metrics, verdicts: {"closer_to_target": 7, "coverage_added": 10, "descriptive": 1, "further_from_target": 14, "improved": 30, "structurally_capped": 6, "unchanged": 7}.

Distribution distances to the dev-200 profile: {"call_count_tv_pilot3": 0.0817, "call_count_tv_pilot4": 0.0217, "query_mode_tv_pilot3": 0.8084, "query_mode_tv_pilot4": 0.09}

Every metric with its direction of improvement and caveat is in `reports/pilot3_vs_pilot4/PILOT3_VS_PILOT4_METRICS.csv`. Selected rows:

| metric | pilot3 | pilot4 train600 | dev-200 target | direction | verdict |
|---|---|---|---|---|---|
| multi_join_rate | 0.0 | 0.2133 | 0.125 | higher_is_better | improved |
| fan_out_rate | 0.0 | 0.27 | 0.0 | coverage_required_above_profile | coverage_added |
| reuse_rate | 0.0 | 0.27 | 0.0 | coverage_required_above_profile | coverage_added |
| plan_leak_rate | 0.6417 | 0.0967 | 0.035 | lower_is_better | improved |
| goal_based_share | 0.01 | 0.6533 | 0.7 | closer_to_target_is_better | closer_to_target |
| mean_operation_explicitness | 0.8378 | 0.235 | 0.1645 | lower_is_better | improved |
| mean_sequence_leakage | 0.7922 | 0.2141 | 0.152 | lower_is_better | improved |
| mean_procedural_cue_count | 6.6833 | 1.3467 | 0.645 | lower_is_better | improved |
| n_distinct_output_keys | 1 | 8 |  | higher_is_better | improved |
| n_capability_families | 0 | 25 |  | higher_is_better | improved |
| schema_compatible_distractor_share | 0.0 | 0.95 |  | higher_is_better | improved |
| bucket[3].top1_topology_share | 0.6262 | 0.7206 | 0.6818 | lower_is_better | structurally_capped |
| bucket[5].n_distinct_topologies | 4 | 18 | 4 | higher_is_better | improved |
| bucket[6+].n_distinct_topologies | 16 | 62 | 26 | higher_is_better | improved |
| bucket[6+].top1_topology_share | 0.1939 | 0.0465 | 0.1364 | lower_is_better | improved |

## 15. Adaptive sampler implementation

Offline simulation over 600 pilot4 prompts for 200 steps. Response model: `synthetic_difficulty_model`.

Caveat: synthetic difficulty model: exercises the sampler only, it is not evidence about the trained policy

| sampler | dead-group rate before filter | effective-group rate after filter | rollout utilization | refill rounds | final entropy | prompts touched |
|---|---|---|---|---|---|---|
| uniform | 0.0361 | 0.9639 | 0.9639 | 1.005 | 1.0 | 600 |
| dynamic_effective_group | 0.0372 | 0.9628 | 0.9628 | 1.01 | 0.998805 | 600 |
| history_adaptive | 0.0322 | 0.9678 | 0.9678 | 1.01 | 0.994196 | 600 |
| cell_curriculum | 0.0302 | 0.9698 | 0.9698 | 1.01 | 0.995081 | 600 |

## 16. Training logging implementation

The trainer writes `TRAIN_RUN_MANIFEST.json`, `train_rollouts.jsonl`, `train_groups.jsonl`, `train_steps.jsonl` and, at every checkpoint, `sampler_state.json`, `sampler_cell_stats.csv` and `sampler_prompt_stats.parquet` (CSV fallback when pyarrow is absent). Per-rollout rows carry the reward components, the group mean and standard deviation, both advantages, parse/execution status and the response hash; the response text can be gzipped. Checkpoint resume restores the sampler state. No training was run.

## 17. Evaluation logging implementation

Each eval run writes `EVAL_RUN_MANIFEST.json`, `eval_inputs.jsonl`, `eval_trajectories.jsonl` and `eval_task_scores.csv`, recording backend and engine identity, adapter path and hash, decoding parameters, chat-template and tool-schema serialization hashes, the input dataset hash and ordered sample IDs, the shard manifest and the parser/scorer versions. A paired-run gate compares task set, task order, prompt hashes, tool-schema hashes and scorer version between two runs and refuses to compare scores when they differ. No evaluation was run.

## 18. Tests

126 test functions, run with `python -m pytest tests -q`; parametrised ones expand to more cases at collection time.

| file | test functions |
|---|---|
| test_cli_dummy.py | 2 |
| test_core.py | 11 |
| test_export_probe.py | 6 |
| test_generation.py | 7 |
| test_pilot2.py | 31 |
| test_pilot3.py | 3 |
| test_pilot4.py | 48 |
| test_selection_split.py | 6 |
| test_validation.py | 12 |

## 19. Known limitations

- No training, rollout, vLLM, Hugging Face generation or NESTFUL evaluation was run, so nothing here is evidence about model accuracy.
- V4 (bounded minimal-path shortcut search) is disabled in the default run because it dominates runtime; V1-V3 and V5-V8 gate every selected task and the run can be repeated with --run-v4.
- Fan-out and output reuse are generated even though dev-200 shows none of them. That is deliberate coverage of the structures pilot3 lacked entirely, and it is a known distribution-mismatch risk rather than a profile match.
- At two and three calls the topology space is exhausted by 1 and 3 shapes, so per-bucket diversity there cannot be improved further; the top-1 share is pinned by the join rate the profile asks for.
- Query realism is measured by a versioned rule-based lexicon. A low leakage score is not proof that a question reads naturally to a human.
- The sampler simulation uses a synthetic difficulty model, not recorded rollouts, because no per-rollout reward log exists from the pilot3 run. It exercises the sampler mechanics only.
- Training and evaluation logging are implemented and unit-tested against their schemas, but no run has produced the artifacts yet.
- Capability demand for the diagnostic-informed gap report is exploratory and is never used as a generation quota.

## 20. Commands for reproduction

From `experiments/targeted_tool_data_factory` with `PYTHONPATH=src`:

```bash
python -m targeted_tool_data.cli audit-provenance
python -m targeted_tool_data.cli audit-query-realism --profile-safe
python -m targeted_tool_data.cli build-profile-v2
python -m targeted_tool_data.cli capability-audit
python -m targeted_tool_data.cli generate-pilot4
python -m targeted_tool_data.cli compare-datasets --baseline pilot3 --candidate pilot4_profile_safe
python -m targeted_tool_data.cli simulate-sampler
python -m targeted_tool_data.cli implementation-report
python -m pytest tests -q
```

## 21. Recommended next experiment, not executed

- Freeze the pilot4 train-600 split and run one MT-GRPO training with the history-adaptive sampler enabled and the new per-rollout logging on, keeping every other hyperparameter identical to the pilot3 run.
- Read dead_group_rate and effective_group_rate from train_steps.jsonl rather than inferring them, and confirm the sampler reduces the dead-group share.
- Evaluate the resulting adapter and the base model in one matched-engine paired run, checking the eval manifest equality gates (same task set, order, prompt hashes, tool-schema hashes, scorer version) before comparing scores.
- Only then compare NESTFUL accuracy, with a paired significance test, and treat the pilot3 +2.2 pp result as the baseline to beat.

