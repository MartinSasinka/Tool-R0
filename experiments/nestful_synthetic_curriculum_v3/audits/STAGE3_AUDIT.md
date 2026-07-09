# STAGE 3 AUDIT — run 20260708_212347_v3_1 and its temp0 final eval

Date: 2026-07-09 · Read-only audit.

## 1. What Stage 3 was

Run `20260708_212347_v3_1`: Stage 3 only (3-call composition, dataset A
`stage3_3call_composition.jsonl`, 800 tasks), 2 epochs, initialized from the base model
(no Stage 1/2 checkpoint), **no mixed replay**, reward `execution_aware_v3_1_stepwise`,
8 generations/group, rollout T=1.0, lr 5e-7, kl_beta 0.15, QLoRA r16/α32.

Training health: 59/71 optimizer steps; dead-group rate 0.709 → 0.646; 17 unique reward
values (fractional reward confirmed live); too_few_calls 0.42; avg predicted calls 2.19 of 3.
**The stage gate FAILED** on `position_artifact_rate_lt_max` (0.35–0.41 vs max 0.2): a third
of "alive" groups derive their reward variance from turn position only, not from
between-completion differences — those groups train on an artifact.

## 2. Internal vs official win, all available evals

### NESTFUL dev (200 tasks, sampled decoding, same-run batch — comparable within this table)

| cell | internal win | official win | strict_trace | final_answer_pass |
|---|---|---|---|---|
| baseline (same run) | 0.635 | **0.565** | — | — |
| s3_e1 | 0.605 | **0.545** | 0.190 | 0.590 |
| s3_e2 | 0.595 | **0.535** | — | — |

Verdict on dev: Stage 3 is **-2 pp official vs its own same-batch baseline** after epoch 1
and -3 pp after epoch 2. Within noise for n=200 (±3.5 pp), but the direction is not an
improvement.

### Full NESTFUL (1,861 tasks, temp0, batch `final_eval_stage3_e1e2_20260709_093453_temp0`)

| cell | internal win | official win | full acc | partial acc | zero_calls | avg calls |
|---|---|---|---|---|---|---|
| baseline | **ABSENT from batch** | **ABSENT** | — | — | — | — |
| s3_e1 | 0.6131 | 0.5438 | 0.024 | 0.189 | 0.090 | 2.28 |
| s3_e2 | 0.6077 | 0.5368 | 0.027 | 0.191 | 0.091 | 2.29 |

The nearest baseline numbers are from batch `final_eval_all_runs_20260708_164607_temp0`
(same temp0 decoding, one day earlier): baseline internal 0.5986, **official unavailable**
(that batch predates the official scorer in the final-eval path). Comparing s3_e1 internal
0.6131 vs 0.5986 across batches would suggest +1.5 pp — **do not use this**: different batch,
no official baseline score, and the gap is ~2σ at best under optimistic assumptions.

**Conclusion: Stage 3 is NOT interpretable as an improvement.** On the only same-batch
comparison that exists (dev), it is slightly below baseline. On full NESTFUL there is no
baseline cell at all.

## 3. Why internal and official win differ (6–7 pp, systematic)

Both are "executed calls lead to gold answer" metrics, but:

1. **Answer-matching tolerance.** The internal evaluator (`metrics.py`) uses lenient
   normalization (numeric tolerance, string normalization, container coercion) when comparing
   the final executed value to `gold_answer`; the official NESTFUL `calculate_win_score`
   requires its own stricter equality on the executed variable store. Borderline cases
   (floats formatted differently, lists vs scalars, case) count as internal-win but
   official-loss.
2. **What gets executed.** The internal path scores the trajectory the agent actually ran
   (including recovery turns and the executor's leniency); the official scorer re-executes
   the extracted call sequence from scratch — an agent that "won" via a messy trace can fail
   official re-execution.
3. **Population**: on dev both use the same 200 tasks, so the difference (0.635→0.565 etc.)
   is purely definitional, consistently ≈ +6–7 pp inflation of internal over official across
   every cell measured (baseline, s2, s3, dev and full).

Practical rule adopted in METRIC_STANDARD_PROPOSAL.md: internal win is a diagnostic only;
paper numbers must come from `metrics_official.json`.

## 4. Exact commands to produce a valid same-batch temp0 comparison

Run on the training pod (4×GPU, vLLM), one batch containing baseline + s3_e1 + s3_e2, all
with the official scorer and temperature 0.0. Do **not** run this now — proposal only.

```bash
cd /workspace/Tool-R0
TS=$(date +%Y%m%d_%H%M%S)
ROOT=experiments/nestful_synthetic_curriculum_v3/outputs/runs/final_eval_s3_with_baseline_${TS}_temp0
mkdir -p "$ROOT"
RUN=experiments/nestful_synthetic_curriculum_v3/outputs/runs/20260708_212347_v3_1

for cell in baseline s3_e1 s3_e2; do
  case $cell in
    baseline) CKPT_ARGS="" ;;
    s3_e1)    CKPT_ARGS="--checkpoint $RUN/stage_3/checkpoints/adapter_epoch_1" ;;
    s3_e2)    CKPT_ARGS="--checkpoint $RUN/stage_3/checkpoints/adapter_epoch_2" ;;
  esac
  python experiments/nestful_mtgrpo_minimal/run.py \
    --mode final_eval \
    --config experiments/nestful_mtgrpo_partial/config.yaml \
    $CKPT_ARGS \
    --override hardware.use_vllm=true \
    --override data.eval_paradigm=react \
    --override generation.temperature=0.0 \
    --override paths.full_nestful_jsonl=experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl \
    --override experiment.output_dir="$ROOT/$cell" \
    2>&1 | tee "$ROOT/$cell.log"
done
```

Notes:

- Keep the argument order exactly as in the successful Batch-3 invocation — the
  `s3_e1.log` shows an earlier attempt failed with `run.py: error: unrecognized arguments`
  when flags were passed in a different form; verify with `--help` before launching.
- Verify all three cells emit `metrics_official.json`; if the baseline cell doesn't, the
  official scorer hook is checkpoint-conditional somewhere and must be fixed first.
- Report `official win ± 1.96·sqrt(p(1-p)/1861)` (≈ ±2.3 pp) and paired
  gained/regressed counts from per-sample official wins, not just the aggregate.
- Optionally add `--override data.eval_paradigm=react` cells for nestful_test-only (1,661)
  to avoid the dev-in-full contamination noted in DATASET_AUDIT §1.
