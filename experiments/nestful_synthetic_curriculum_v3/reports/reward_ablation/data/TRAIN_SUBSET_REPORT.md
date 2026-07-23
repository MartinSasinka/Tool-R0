# TRAIN_SUBSET_REPORT — 160-task deterministic stratified train subset

Generated: 2026-07-23T19:53:56.143640+00:00

## Source dataset

- Path: `experiments/nestful_synthetic_curriculum_v3/data/training_ready_v5/filtered/stage3_train_ready.jsonl`
- Rows: 326
- SHA-256 (raw bytes, this Windows checkout): `0d3a2c6cce18ea14ead14e182a59b4b97ad3e76c65cff034b9196ccdea689e00`
- SHA-256 (LF-normalized): `7df704bff35c8f8fd0ffb2b50e3c7c4c1e8d7f9a0f3e0c02a43327ef820dd596`
- SHA-256 recorded in `pure_stage3_2ep_20260719_221918/run_manifest.json`: `7df704bff35c8f8fd0ffb2b50e3c7c4c1e8d7f9a0f3e0c02a43327ef820dd596`
- **Match**: LF-normalized hash == run-manifest hash -> content is byte-for-byte identical to the
  dataset used by the original production run; the raw-bytes mismatch is purely a CRLF checkout
  artifact on Windows (see `source_hash_note` in the manifest).

## Subset

- Seed: 20260724
- Selected: 160 / target 160
- SHA-256: `b64d3ec2477319f67ac20128b974055c539e203c0cd185630b63decf07673b74`
- Identical task IDs used for every reward arm: True
- No arm-specific filtering: True
- Excludes NESTFUL dev/test IDs: True
- Easy tier (`quality_tier=easy_anchor`) fraction: 5.6% (max allowed 10%)

## Full synthetic executor replay (no gold_replay)

- Registry version: `5.0.2`
- Registry hash: `f945b18ccdc260b1960e5fbb20e4d76312628af42fef0a65d4977af83dd6dc0d`
- Replay status: `ok`
- Replay rows validated: 160

## Stratification — primary axis (motif_type x quality_tier)

Allocation per cell (subset / source):

| cell | subset n | source n |
|---|---:|---:|
| argument_binding|easy_anchor | 2 | 4 |
| argument_binding|frontier | 18 | 37 |
| argument_binding|partial_frontier | 12 | 24 |
| distractor_heavy|easy_anchor | 2 | 5 |
| distractor_heavy|frontier | 15 | 30 |
| distractor_heavy|partial_frontier | 15 | 30 |
| fan_in|easy_anchor | 2 | 5 |
| fan_in|frontier | 15 | 30 |
| fan_in|partial_frontier | 18 | 36 |
| long_chain|easy_anchor | 2 | 4 |
| long_chain|frontier | 17 | 36 |
| long_chain|partial_frontier | 12 | 24 |
| reference_reuse|easy_anchor | 1 | 2 |
| reference_reuse|frontier | 15 | 31 |
| reference_reuse|partial_frontier | 14 | 28 |

## motif_type distribution

| value | source (326) | source % | subset (160) | subset % |
|---|---:|---:|---:|---:|
| argument_binding | 65 | 19.9% | 32 | 20.0% |
| distractor_heavy | 65 | 19.9% | 32 | 20.0% |
| fan_in | 71 | 21.8% | 35 | 21.9% |
| long_chain | 64 | 19.6% | 31 | 19.4% |
| reference_reuse | 61 | 18.7% | 30 | 18.8% |

## quality_tier distribution

| value | source (326) | source % | subset (160) | subset % |
|---|---:|---:|---:|---:|
| easy_anchor | 20 | 6.1% | 9 | 5.6% |
| frontier | 164 | 50.3% | 80 | 50.0% |
| partial_frontier | 142 | 43.6% | 71 | 44.4% |

## answer_type distribution (reported, not a sampling axis)

| value | source (326) | source % | subset (160) | subset % |
|---|---:|---:|---:|---:|
| boolean | 92 | 28.2% | 37 | 23.1% |
| list | 5 | 1.5% | 2 | 1.2% |
| scalar | 201 | 61.7% | 111 | 69.4% |
| string | 28 | 8.6% | 10 | 6.2% |

## tool_family distribution (reported, not a sampling axis — heuristic keyword bucketing of the first gold call's tool name)

| value | source (326) | source % | subset (160) | subset % |
|---|---:|---:|---:|---:|
| list | 1 | 0.3% | 1 | 0.6% |
| math | 50 | 15.3% | 29 | 18.1% |
| object | 17 | 5.2% | 7 | 4.4% |
| other | 249 | 76.4% | 118 | 73.8% |
| string | 9 | 2.8% | 5 | 3.1% |

## has_reference_dependency distribution (reported, not a sampling axis)

| value | source (326) | source % | subset (160) | subset % |
|---|---:|---:|---:|---:|
| True | 326 | 100.0% | 160 | 100.0% |

## Notes / limits

- Stratification uses a 2D key (`motif_type` x `quality_tier`) via the largest-remainder method so
  proportions of the 326-task source are preserved as closely as an integer allocation of 160 items
  allows. `answer_type`, `tool_family`, and `has_reference_dependency` are reported for transparency
  but are not independent sampling axes (with only 326 source rows, a full cross of all seven
  requested fields would produce mostly-empty cells and an unstable/non-deterministic selection).
- `tool_family` is a lightweight keyword heuristic over the first gold call's tool name (this
  dataset's tools are code-generated synthetic functions without a first-class "family" field);
  it is descriptive only.
- Selection does not use any C0 (or any other) rollout/eval result — inputs are dataset metadata
  only, per the ablation spec (no arm-specific / outcome-based filtering).
