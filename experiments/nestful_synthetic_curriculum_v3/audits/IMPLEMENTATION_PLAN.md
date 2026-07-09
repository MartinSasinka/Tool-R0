# IMPLEMENTATION PLAN — proposed future work (nothing implemented yet)

Date: 2026-07-09. Ordered by expected payoff per unit of work. Each item lists interface,
acceptance criteria, and dependencies.

## P0 — Unified evaluator + metric schema (fixes the biggest credibility gap)

**`src/eval/unified_evaluator.py`** — one entry point for every evaluation this project runs:

```
evaluate(checkpoint | None, dataset: {nestful_dev, nestful_test, nestful_full, synthetic_stageN, custom_path},
         paradigm: react, temperature: float = 0.0, seed: int, out_dir) -> EvalResult
```

- Always runs BOTH scorers: internal diagnostics and official NESTFUL `calculate_win_score`.
- Always writes per-sample records (`per_sample.jsonl`: sample_id, predicted calls, executed
  values, official_win, internal flags) so paired comparisons never need re-runs.
- Refuses to write results without a `decoding` block (temperature, top_p, max_new_tokens,
  seed) in the output.

**Unified metric JSON schema** (`metrics.schema.json`), single file per eval cell:

```json
{
  "schema_version": 1,
  "run_id": "...", "checkpoint": "...", "dataset": {"name": "nestful_full", "sha256": "...", "n": 1861},
  "decoding": {"temperature": 0.0, "top_p": 0.95, "seed": 42},
  "primary": {"official_nestful_win_rate": 0.5438},
  "official": {"full_sequence_accuracy": 0.024, "partial_sequence_accuracy": 0.189, "f1_func": 0.908, "f1_param": 0.454},
  "diagnostics": {"internal_final_answer_win": 0.613, "final_answer_pass": 0.598, "solution_equivalent_pass": 0.546,
                   "strict_gold_trace_pass": 0.214, "executable_rate": 0.97, "too_few_calls_rate": 0.41,
                   "avg_predicted_calls": 2.28, "parse_error_rate": 0.0, "no_tool_call_rate": 0.09},
  "paired_vs_baseline": {"baseline_cell": "...", "gained": 0, "regressed": 0, "net": 0}
}
```

Acceptance: re-running the Stage-3 batch through the unified evaluator reproduces
`metrics_official.json` win rates to 3 decimals; internal/official are never confusable.

## P0 — Deterministic eval runner with mandatory same-batch baseline

`scripts/run_eval_batch.py --cells baseline,ckptA,ckptB --dataset nestful_full --temp 0.0`

- Hard-fails if `baseline` is not among the cells (override flag for explicit diagnostics).
- Fixed seed, temp 0.0 default, one batch directory (no double-nesting), writes
  `BATCH_REPORT.md` with paired gained/regressed and binomial CIs.
- Fixes the report-generator path bug (looks in the directory it writes to).

## P1 — Run manifest + artifact registry

- `run_manifest.json` written at launch by every trainer/evaluator: git commit, dirty flag,
  dataset paths + sha256, config hash, env (GPU count, vLLM version), wandb run id.
- `artifacts/registry.jsonl` (append-only): one line per dataset/checkpoint/eval with sha256,
  producer run id, and role (`canonical | ablation | archive`). The dataset A/B confusion and
  the "which corpus did July-2 use" question would have been one lookup.

## P1 — GRPO signal diagnostics (live, not post-hoc)

Promote the audit's train-log analysis into the trainer:

- per-step wandb scalars: dead_group_rate (corrected), mixed_rate, unique episode rewards per
  group, reward-band histogram, position_artifact_rate, too_few_calls, avg_predicted_calls;
- early-abort rule: if dead_group_rate over the last 100 groups > 0.8, stop and report
  (saves pod-hours; three of seven runs would have aborted in minutes);
- per-motif dead-rate table at epoch end (string_output/boolean_output/reference_reuse are
  known offenders).

## P1 — W&B logging standardization

One project (`nestful-curriculum-v3_1`), run name = run_id, group = experiment family
(s2_replay/s2_noreplay/s2_tf/s3), tags = dataset sha + reward policy. Log the unified metric
JSON as artifacts. Eval batches log as their own runs linked to the checkpoints' runs.

## P2 — Multi-GPU training runner

Formalize the current env-var choreography (`ROLLOUT_DP_GPUS`, `DP_LEARNER_GPU`, vLLM memory
splits) into a single `train.py --gpus 4` entry that validates the topology up front —
today misconfiguration surfaces as mid-run CUDA OOM or silent single-GPU rollouts.

## P2 — Stage probe runner

`scripts/probe_stage.py --stage N --checkpoint X --n 100`: cheap forward-only probe that
reports solve-rate/band histogram on a stage BEFORE training on it. Would have caught Stage-1
saturation (dead rate 1.0, 0 optimizer steps) without spending a training run; use it as the
stage-advance/skip decision input.

## P3 — Reward re-design experiment (informed by REWARD_AUDIT / FAILURE_MODE_AUDIT)

Not infrastructure but the highest-leverage research change: densify the reward inside the
dominant bands (per-call correctness credit, argument-binding partial credit, explicit
penalty step between "answered directly" and "made call 1 then stopped") to cut dead-group
rate; validate with the stage probe + signal diagnostics before any full run.
