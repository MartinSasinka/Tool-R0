# INPUT_INTEGRITY

Status legend: VERIFIED | PARTIALLY_VERIFIED | NOT_VERIFIABLE | INCONSISTENT

## pairing
- status: `VERIFIED`
- n_c0: `500`
- n_d1: `500`
- n_shared: `500`
- only_c0: `[]`
- only_d1: `[]`
- n_only_c0: `0`
- n_only_d1: `0`
- duplicate_c0: `[]`
- duplicate_d1: `[]`
- num_gold_calls_mismatches: `[]`
- n_gold_mismatches: `0`
- wins_c0_recount: `277`
- wins_d1_recount: `288`
- official_win_invalid_count: `0`
- pairing_ok: `True`

## c0_wins_vs_metrics
- status: `VERIFIED`
- declared_n_wins: `277`
- recounted_wins: `277`
- official_win_metric: `0.554`

## d1_wins_vs_metrics
- status: `VERIFIED`
- declared_n_wins: `288`
- recounted_wins: `288`
- official_win_metric: `0.576`

## c0_shards
- status: `VERIFIED`
- n_shard_files: `4`
- n_shard_rows: `500`
- n_merged: `500`
- duplicate_shard_ids: `[]`
- only_shards: `[]`
- only_merged: `[]`
- wins_shards: `277`
- wins_merged: `277`

## d1_shards
- status: `VERIFIED`
- n_shard_files: `4`
- n_shard_rows: `500`
- n_merged: `500`
- duplicate_shard_ids: `[]`
- only_shards: `[]`
- only_merged: `[]`
- wins_shards: `288`
- wins_merged: `288`

## eval_manifest_parity
- status: `PARTIALLY_VERIFIED`
- c0_diagnostic_basename: `nestful_diagnostic_500.jsonl`
- d1_diagnostic_basename: `nestful_diagnostic_500.jsonl`
- c0_n_diagnostic: `500`
- d1_n_diagnostic: `None`
- c0_n_gpus: `4`
- d1_n_gpus: `4`
- c0_checkpoint: `None`
- d1_checkpoint: `/workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3/outputs/runs/pilot3_D1_seed20260727_n300/checkpoints/FINAL`

## diagnostic
- status: `VERIFIED`
- sha256: `db1525560b20b47ea5567bc52a64f6134655908e96a7b972d2b9bb555045086c`
- n_lines: `500`
- note: `diagnostic-500 is a balanced call-count slice, not a natural NESTFUL sample`

## train_subset_identity
- status: `INCONSISTENT`
- n_subset: `300`
- n_full: `600`
- overlap: `62`
- positional_prefix_match: `False`
- note: `D1 subset is not the local train_grpo_pilot3 prefix; local full-train audits only partially represent D1.`

## rollouts
- status: `NOT_VERIFIABLE`
- reason: `Pilot2 rollouts are not D1 training rollouts; reward cell audit limited`

## train_log_hparams
- status: `PARTIALLY_VERIFIED`
- extracted: *(see JSON)*

## Artifact usage
- `bundle_sha_manifest` [UNUSED] `experiments/targeted_tool_data_factory/runpod_bundle_pilot3/MANIFEST.sha256.json` — not yet selected
- `c0_hf_trajectories` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/final_eval_trajectories.jsonl` — C0 HF backend confound arm
- `c0_manifest` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/eval_C0_nestful500_vllm_matched_v2/eval_manifest.json` — not yet selected
- `c0_metrics` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/eval_C0_nestful500_vllm_matched_v2/metrics_merged.json` — 
- `c0_shards_dir` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/eval_C0_nestful500_vllm_matched_v2/shards` — 
- `c0_trajectories` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/eval_C0_nestful500_vllm_matched_v2/final_eval_trajectories.jsonl` — C0 matched vLLM preferred
- `d1_manifest` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/eval/D1_nestful500/eval_manifest.json` — not yet selected
- `d1_metrics` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/eval/D1_nestful500/metrics_merged.json` — 
- `d1_predictions` [UNUSED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/eval/D1_nestful500/final_eval_predictions.partial.jsonl` — not yet selected
- `d1_shards_dir` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/eval/D1_nestful500/shards` — 
- `d1_trajectories` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/eval/D1_nestful500/final_eval_trajectories.jsonl` — D1 vLLM
- `diagnostic_data` [USED] `experiments/targeted_tool_data_factory/runpod_bundle_pilot2/data/nestful_diagnostic_500.jsonl` — 
- `export_manifest` [USED] `experiments/targeted_tool_data_factory/outputs/selected/export_pilot3/manifest_pilot3.json` — 
- `full_train_data` [USED] `experiments/targeted_tool_data_factory/outputs/selected/export_pilot3/train_grpo_pilot3.jsonl` — 
- `generation_cells` [USED] `experiments/targeted_tool_data_factory/outputs/candidates/cells_pilot3.json` — 
- `heldout_data` [USED] `experiments/targeted_tool_data_factory/outputs/selected/export_pilot3/heldout_grpo_pilot3.jsonl` — 
- `pilot2_train` [USED] `experiments/targeted_tool_data_factory/outputs/selected/export_pilot2/train_grpo_pilot2.jsonl` — 
- `preflight_gold_replay` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/preflight_gold_replay.json` — 
- `profile_match` [USED] `experiments/targeted_tool_data_factory/outputs/selected/profile_match_pilot3.json` — 
- `reserve_data` [USED] `experiments/targeted_tool_data_factory/outputs/selected/export_pilot3/reserve_grpo_pilot3.jsonl` — 
- `rollout_log` [UNUSED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/rollouts.jsonl` — Pilot2 rollouts are not D1 training rollouts; reward cell audit limited
- `selection_trace` [USED] `experiments/targeted_tool_data_factory/outputs/selected/selection_trace_pilot3.jsonl` — 
- `target_profile` [USED] `experiments/targeted_tool_data_factory/outputs/profiles/nestful_profile.json` — 
- `train_data` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/train_subset_300.jsonl` — D1 trained subset n=300
- `train_log` [USED] `experiments/targeted_tool_data_factory/outputs/runpod_pilot3_from_zip2/runpod_pilot3/train_nestful500/run_train_nestful500_20260727_005500.log` — 
