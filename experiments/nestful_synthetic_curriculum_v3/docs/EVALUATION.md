# EVALUATION

## Primary metric

**`official_nestful_win_rate`** — the official NESTFUL scorer's Win Rate: the extracted
predicted call sequence is re-executed against the IBM executable functions and wins iff it
produces the gold answer. Computed by `nestful_mtgrpo_minimal/nestful_official_score.py`
(wrapping the NESTFUL repo scorer), written to `metrics_official.json` per eval cell, and
surfaced as `primary.official_nestful_win_rate` in `metrics_unified.json`.

Headline conditions (all enforced by the batch runner):
- temperature 0.0 (or explicitly recorded otherwise),
- baseline cell in the same batch,
- full dataset (no `--max-tasks`),
- official scorer output present for every cell.

## Why internal win is diagnostic-only

The internal evaluator (`metrics.py`) also reports a "win" (executed trajectory reaches the
gold answer), but it systematically **inflates the official number by ~6–7 pp** across every
cell measured in the audit (STAGE3_AUDIT §3). Two definitional differences:

1. **Tolerance** — internal comparison uses lenient normalization (numeric tolerance, string
   normalization, container coercion); the official scorer is stricter (decimal-aware
   equality on the executed variable store).
2. **Execution path** — internal scores the trajectory the agent actually ran (including
   recovery turns and executor leniency); the official scorer re-executes the extracted call
   sequence from scratch.

Therefore the internal number is exported only as `diagnostics.internal_final_answer_win`
and must never appear in a headline or be called "win rate" in reports.

## Metric inventory (per eval cell)

| name | file | meaning | use |
|---|---|---|---|
| `official_nestful_win_rate` | `metrics_official.json` → unified `primary` | official re-executed win | **paper headline** |
| official `f1_func`, `f1_param`, `partial/full_sequence_accuracy` | `metrics_official.json` | official trace-level scores | paper secondary |
| `internal_final_answer_win` | `metrics.json` (`internal_metrics_diagnostic.win_rate`) | lenient executed-trajectory win | diagnostic |
| `final_answer_pass` | `metrics.json` (`our_metrics`) | final answer matches gold (any trace) | diagnostic |
| `solution_equivalent_pass` | `metrics.json` | executed trace yields gold answer via valid alternative | diagnostic |
| `strict_gold_trace_pass` | `metrics.json` | exact gold-trace match | diagnostic |
| `too_few_calls_rate`, `avg_predicted_calls`, `no_tool_call_rate`, `parse_error_rate` | recomputed from `final_eval_trajectories.jsonl` by the batch runner | call behavior | diagnostic |
| `paired_vs_baseline {gained, regressed, net}` | `metrics_unified.json` | per-sample official-win pairing vs baseline | required alongside any delta |

Caution: `strict_gold_trace_pass` in **training** `train_log.jsonl` is actually mean episode
reward (misnamed; METRIC_AUDIT §3.10) — do not confuse it with the eval metric above.

## How to run a same-batch evaluation

```bash
# temp0 batch: baseline + two checkpoints, on nestful_test (1,661 tasks)
CELLS="baseline,s3_e1=<adapter_dir_1>,s3_e2=<adapter_dir_2>" \
DATASET=nestful_test \
BATCH_NAME=s3_reeval \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# preview commands without any GPU work
DRY_RUN=1 CELLS="baseline,..." bash .../eval_batch_temp0.sh

# tiny smoke batch (NOT reportable)
MAX_TASKS=5 CELLS="baseline,..." bash .../eval_batch_temp0.sh
```

The runner refuses to start without a `baseline` cell (exit 2; `--allow-no-baseline` is the
explicit escape hatch and stamps the report as non-comparable), refuses legacy dataset-B
paths (exit 3), and fails the batch if any cell lacks `metrics_official.json` (exit 4).

Dataset choice: `nestful_test` (1,661) is the clean headline set; `nestful_full` (1,861)
includes the 200 dev tasks used for checkpoint selection (mild contamination; DATASET_AUDIT
§1); `nestful_dev` (200) is for selection only.

Platform note: `run.py` skips the official Win computation on Windows and when the IBM
`executable_functions` dir is missing — run eval batches on the Linux pod and check
`scripts/setup/check_env.sh` first.

## How to interpret paired gained/regressed counts

For each non-baseline cell the runner pairs per-sample official wins with the baseline cell
(same task ids) and reports `gained` (lost→won), `regressed` (won→lost), `net`. Rules:

- An aggregate delta with a tiny paired flux (e.g. +3/−1) is noise; a real improvement moves
  many samples with clearly positive net.
- Always read the net against the 95 % binomial CI on the aggregate (~±2.3 pp at n=1861,
  ~±2.4 pp at n=1661).
- Cross-batch comparisons are invalid regardless of how close the settings look — decoding,
  scorer availability, and code drift have all shifted numbers ~1 pp in past batches
  (RUN_AUDIT §2).

## Regenerating reports for existing batches

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/eval/run_eval_batch.py \
  --report-only <batch_dir> [--baseline-cell baseline] [--allow-no-baseline]
```

Writes `metrics_unified.json` per cell and `BATCH_REPORT.md`; historical batches without a
baseline cell get a non-comparable warning stamped in the report.
