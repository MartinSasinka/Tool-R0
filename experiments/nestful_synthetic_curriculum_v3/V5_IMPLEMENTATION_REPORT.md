# Curriculum v5 — Implementation Report

Refactor of the synthetic tool-use training pipeline: real executable
synthetic functions during training, a much broader tool space, and simple
deterministic evaluation / checkpoint selection.

Date: 2026-07-15 · Registry: v5.0.0, hash `31a99c56b050281f…` · No large
training run or full-scale generation was launched.

---

## 1. Root causes found

1. **`gold_replay` cannot falsify wrong argument VALUES.** The legacy
   executor compared the predicted call's *name + argument shape* against the
   gold trace and then returned the *gold observation*. A rollout that called
   the right tool with wrong values still received the gold result, so the
   reward signal collapsed for exactly the failure mode Stage 2 was meant to
   train away (F1 Parameter). This is the primary reason the agentic Stage 2
   run moved almost nothing.
2. **34-tool ceiling in the v4 generator.** The agentic generator drew from a
   fixed 34-tool pool with near-uniform templates; a handful of tools
   dominated the corpus and the model could exploit surface regularities
   instead of learning argument grounding.
3. **Synthetic tools had no implementations.** Tool schemas existed only as
   JSON; there was nothing to execute, which is why gold-replay existed in the
   first place.
4. **Training path sprawl.** `run_grpo.sh` mixed stage overrides, silent
   `executor.mode` forcing, mixed-replay banners and multi-script resume
   logic; runs were hard to reproduce and easy to misconfigure.
5. **Stochastic validation.** Epoch selection used temperature-0.7 rollouts —
   noisy enough to pick a bad checkpoint (observed: "best" chosen from two
   statistically indistinguishable points).

## 2. Files added / changed

### Added

| File | Purpose |
| --- | --- |
| `nestful_synthetic_curriculum_v3/lib/synthetic_tools.py` | Versioned executable registry: 163 tools, schema + deterministic implementation + semantic types + samplers, `registry_hash()`, `validate_registry()` self-check (determinism probes, behavioural-duplicate detection). |
| `nestful_synthetic_curriculum_v3/lib/synthetic_gen_v5.py` | Registry-driven generator: semantically compatible dependency graphs, all observations computed by execution, inverse-frequency tool balancing with hard share cap, 4 motifs, contamination-safe. |
| `nestful_mtgrpo_minimal/synthetic_tool_registry.py` | Loader that makes the v3 registry importable from the trainer (singleton, explicit load errors). |
| `nestful_mtgrpo_minimal/tests/test_synthetic_executor.py` | 17 regression tests: wrong values ≠ gold result, invalid refs fail, wrong types fail, chains get real observations, determinism, coercion, unknown/missing args. |
| `nestful_synthetic_curriculum_v3/scripts/data/build_v5_dataset.py` | Dataset builder: generation + NESTFUL contamination check + full replay of every row through the REAL trainer executor + manifest (registry version/hash, sha256 per file). |
| `nestful_synthetic_curriculum_v3/scripts/data/score_v5_dataset.py` | Quality/diversity scorer with configurable thresholds (max tool share, min unique tools, min replay pass). |
| `nestful_synthetic_curriculum_v3/scripts/training/run_v5_pipeline.py` | ONE training path: run manifest, registry-hash gate, per-epoch checkpoint + deterministic dev eval, lexicographic checkpoint selection, optional early stopping, explicit resume. |
| `nestful_synthetic_curriculum_v3/scripts/eval/final_eval_v5.py` | Final temp-0 eval (`run`) + paired comparison (`compare`): by-call-count metrics, gained/regressed lists, diagnostics, paired bootstrap 95% CI. |
| `nestful_synthetic_curriculum_v3/scripts/v5/*.sh` | 9 documented entry points (below), strict mode, env-var config, resolved-config printing. |

### Changed

| File | Change |
| --- | --- |
| `nestful_mtgrpo_minimal/executor.py` | New `executor.mode="synthetic"`: strict schema validation (unknown keys, missing required, types, min/max/min_len), field-aware `$varN.field$` resolution, REAL execution, error propagation. `gold_replay` retained, documented as LEGACY. |
| `nestful_mtgrpo_minimal/reward.py` | `compute_gold_observations(..., mode=)` re-executes gold calls with the run's executor mode so graded rewards use real gold observations. |
| `nestful_mtgrpo_minimal/data.py` | Preserves `observations`, `registry_version`, `registry_hash` metadata from v5 rows. |
| `nestful_mtgrpo_minimal/rollout.py` | `exec_failure_categories()` — categorised executor-failure counters (`execfail_*`). |
| `nestful_mtgrpo_minimal/grpo_train.py` | Executor-failure categories aggregated into per-task train-log records alongside dead-group rate, unique rewards, reward variance, train win, reward classes and predicted call counts. |
| `nestful_mtgrpo_minimal/vllm_dp_pool.py` | Same failure categories flow back from DP rollout workers. |
| `nestful_mtgrpo_minimal/run.py` / partial `run.py` | Call sites pass the configured executor mode to gold-observation computation. |
| `nestful_synthetic_curriculum_v3/scripts/training/run_grpo.sh` | Rejects curriculum_v5 datasets (points to the new pipeline); the agentic-v4 `gold_replay` forcing now prints an explicit LEGACY warning. |

## 3. Legacy paths deprecated (not deleted)

- `executor.mode=gold_replay` — still available, explicitly labelled
  legacy/baseline; it is never the default for the v5 pipeline and `run_grpo.sh`
  now warns loudly when it is auto-selected for old v4 agentic data.
- `run_grpo.sh` multi-stage launcher — kept working for pre-v5 datasets but
  refuses v5 data and redirects to `scripts/v5/train_stage2.sh`.
- The v4 agentic generator and its datasets remain untouched under
  `scripts/data/` and `outputs/`; frozen reward v3.1 artifacts unchanged.

## 4. Registry size and diversity

- **163 genuinely distinct executable tools** (was 34), 11 domains, 39
  families. Arity: 65 unary, 79 binary, 19 ternary. Output types: 111 number,
  16 integer, 16 string, 12 boolean, 7 object (with nested fields), 1 array.
- Semantic types (money, length_m, kg, percent, text, flag, …) gate chaining;
  `validate_registry()` rejects behavioural duplicates (identical outputs on
  shared probes AND identical semantic signature) — renamed clones cannot
  enter the registry.
- Pilot corpus (160 tasks, 4 stages): **135 unique tools used, max single-tool
  share 3.1 %**, replay pass rate 1.0, fielded-reference/reuse/multi-reference
  motifs present. Thresholds are CLI-configurable (`--max-tool-share`,
  `--min-unique-tools`, `--min-replay-pass`).

## 5. Executor behavior (synthetic mode)

For every predicted call: tool must exist in the registry and in the task's
offered schema → unknown keys rejected → required keys enforced → types
validated (numeric strings coerced; genuine type errors fail) → min/max and
list constraints enforced → `$varN(.field)$` references resolved only against
previous outputs → the REAL function executes with the PREDICTED values →
the actual observation (or a categorised error) is returned. Gold traces are
used only by the reward, never to simulate success. Wrong values therefore
produce wrong observations and wrong final answers — proven by regression
tests.

## 6. Simplified training flow

`run_v5_pipeline.py` is the single path: one dataset (sha256 verified before
every epoch), one executor mode (default `synthetic`), one reward policy, a
run manifest with registry path/version/hash + git state, per-epoch
checkpoint + train summary + deterministic dev eval, resume that prints the
exact source checkpoint and refuses configuration drift. It aborts when the
dataset's `registry_hash` differs from the registry the trainer would execute.

## 7. Checkpoint-selection policy

Deterministic dev eval after every epoch: temperature 0.0, top_p 1.0, one
rollout, ReAct, fixed NESTFUL dev set, full executor, official scorer.
Selection is lexicographic: **official ReAct win rate → F1 Parameter →
full-sequence accuracy**. Early stopping is optional (`--patience`, default
off) and never fires before `--min-epochs`. The winning adapter is copied to
`<run_dir>/best_adapter` and the full ranking recorded in
`checkpoint_selection.json`.

## 8. Shell entry points (`scripts/v5/`)

1. `validate_registry.sh` — registry self-check + executor regression tests
2. `gen_pilot.sh` — small dataset pilot (40 rows/stage, full replay gate)
3. `score_dataset.sh` — quality/diversity scoring with thresholds
4. `train_smoke.sh` — minutes-long end-to-end smoke (tiny caps)
5. `train_stage2.sh` — full Stage 2 training (deliberate launch only)
6. `resume.sh` — resume with manifest cross-check
7. `dev_eval.sh` — deterministic dev eval of one checkpoint
8. `final_eval.sh` — final temp-0 NESTFUL eval of one arm
9. `compare_ckpts.sh` — paired baseline/best/final report with bootstrap CI

All use `set -euo pipefail`, env-var configuration, print the fully resolved
configuration, fail on missing files/conflicting options, and write manifests
into the run directory.

## 9. Verification performed (this machine, no GPU work)

- Registry self-validation: **ok, 0 errors, 0 fatal duplicates** (behaviour
  twins across distinct semantics are reported, allowed by design).
- `nestful_mtgrpo_minimal` suite: **148 passed, 5 skipped** (incl. the 17 new
  executor tests; scorer tests needed `jsonlines` + `scikit-learn` installed).
- v3 suite: **47 passed**.
- Pilot generation: 4×40 rows, contamination check against 1860 NESTFUL
  question hashes / 1847 trace hashes, **100 % replay through the real
  trainer executor** (a generator bug — chained values violating downstream
  min-constraints — was caught by this gate and fixed).
- Quality scorer: all thresholds green (see §4).
- Pipeline + final-eval dry runs: resolved config printed, registry-hash gate
  verified; compare path validated on a synthetic fixture (bootstrap CI,
  gained/regressed, by-call-count buckets).
- GPU smoke training was NOT run here (Windows dev box); first pod step below.

## 10. Exact commands for the next controlled experiment (on the pod)

```bash
cd experiments/nestful_synthetic_curriculum_v3

# 0) gates
bash scripts/v5/validate_registry.sh

# 1) full Stage-2 dataset (~800 rows) + quality gate
.venv/bin/python scripts/data/build_v5_dataset.py \
  --stages v5_stage2_3call --examples-per-stage 800 --seed 42 \
  --output-dir data/curriculum_v5_registry
bash scripts/v5/score_dataset.sh "data/curriculum_v5_registry/filtered/v5_stage2_3call.jsonl"

# 2) end-to-end smoke (minutes) — MUST pass before the real run
DATASET=data/curriculum_v5_registry/filtered/v5_stage2_3call.jsonl \
USE_VLLM=1 EVAL_TP=4 bash scripts/v5/train_smoke.sh

# 3) full Stage-2 training (3 epochs, 8 rollouts, real synthetic executor)
DATASET=data/curriculum_v5_registry/filtered/v5_stage2_3call.jsonl \
RUN_DIR=outputs/runs/v5_s2_$(date +%Y%m%d_%H%M%S) \
EPOCHS=3 NUM_GENERATIONS=8 USE_VLLM=1 ROLLOUT_DP_GPUS=1,2,3 EVAL_TP=4 \
bash scripts/v5/train_stage2.sh

# 4) final temp-0 evaluation + paired comparison
RUN=outputs/runs/v5_s2_<ts>
LABEL=baseline OUT_DIR=$RUN/final_eval/baseline USE_VLLM=1 EVAL_TP=4 bash scripts/v5/final_eval.sh
LABEL=best CHECKPOINT=$RUN/best_adapter OUT_DIR=$RUN/final_eval/best USE_VLLM=1 EVAL_TP=4 bash scripts/v5/final_eval.sh
BASELINE_DIR=$RUN/final_eval/baseline BEST_DIR=$RUN/final_eval/best \
OUT_DIR=$RUN/final_eval/compare bash scripts/v5/compare_ckpts.sh
```

## Acceptance criteria — status

| Criterion | Status |
| --- | --- |
| Valid predicted calls produce real observations | ✅ regression-tested |
| Wrong values never receive gold observations/answers | ✅ regression-tested |
| Generator, dataset and trainer share one registry hash | ✅ hash gate in builder + pipeline (dry run verified) |
| Registry substantially >34 executable tools | ✅ 163 tools / 11 domains |
| Generated tasks pass full replay through the real executor | ✅ 100 % on pilot, hard gate in builder |
| Training defaults to the real synthetic executor | ✅ pipeline default `synthetic` |
| Validation + final eval always temperature 0 | ✅ forced by explicit overrides, never inherited |
| Best checkpoint from deterministic official dev metrics | ✅ win → F1 param → full-seq |
| All tests and smoke checks pass | ✅ 148+47 passed, offline smokes green |
| No large training / expensive generation auto-started | ✅ nothing launched |
