# Two-Phase v5 GRPO Runbook (RunPod, 4× GPU)

Continuous **in-process** training: Phase 1 (429× Stage 2) and Phase 2 (466× Stage 3 + replay)
share one Python interpreter, one AdamW optimizer, and monotonic `global_step`.
Rollout workers (GPUs 1–3) stay up between phases; training teardown shuts down
workers **and** unloads the HF learner from GPU 0 so EVAL_TP=4 can use all GPUs
before deferred C1/C2 dev eval (TP=4 on all four GPUs).

**Canonical entry points (do not fork):**

| Role | Path |
|------|------|
| Orchestrator | `scripts/training/run_two_phase_v5_grpo.py` |
| Shell launcher | `scripts/v5/run_two_phase_grpo.sh` |
| In-process session | `scripts/training/two_phase_train_session.py` |
| Preflight | `scripts/training/preflight_training_datasets.py` |
| Trainer | `nestful_mtgrpo_minimal/grpo_train.py` (via `TwoPhaseTrainSession`) |
| Config | `nestful_mtgrpo_partial/config.yaml` |
| Reward | `lib/reward_v3_2_dense.py` |
| Eval | `scripts/eval/final_eval_v5.py` |
| Deps installer | `nestful_mtgrpo_minimal/install_deps.sh` |

**Startup assertions (hard abort):** `executor.mode=synthetic`, `reward=execution_aware_v3_2_dense`, `registry=v5.0.2`, no `gold_replay`.

---

## Datasets

| Phase | File | Rows |
|-------|------|------|
| 1 | `data/training_ready_v5/filtered/phase1_stage2_train.jsonl` | 429 |
| 2 | `data/training_ready_v5/filtered/phase2_stage3_plus_stage2_replay.jsonl` | 466 |

Manifest: `data/training_ready_v5/manifests/training_ready_v5_manifest.json`

---

## 1. Clone & enter repo

```bash
cd /workspace
git clone https://github.com/YOUR_ORG/Tool-R0.git
cd Tool-R0
git rev-parse HEAD   # recorded in run_manifest.json + W&B
```

---

## 2. Dependencies

```bash
bash experiments/nestful_mtgrpo_minimal/install_deps.sh
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh
```

---

## 3. Hugging Face

```bash
export HF_TOKEN="hf_..."
huggingface-cli login --token "$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface
```

---

## 4. Weights & Biases

```bash
export WANDB_API_KEY="..."          # never commit
export WANDB_PROJECT="nestful-v5-curriculum"
export WANDB_ENTITY="your-team"     # optional
export WANDB_MODE="online"
```

**One run group** (`WANDB_RUN_GROUP`) with separate runs:

- `{group}-eval-C0`
- `{group}-phase1-train`
- `{group}-phase2-train`
- `{group}-phase2-stage-metrics` (Stage 2 vs Stage 3 split)
- `{group}-eval-C1` (after Phase 2 — chronology only, not scientific)
- `{group}-eval-C2`
- `{group}-compare`

Reproducibility env (recorded in manifest + W&B config):

```bash
export SEED=42
export DATA_SEED=42
export ROLLOUT_SEED=42
```

---

## 5. GPU topology

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_VLLM=1
export ROLLOUT_DP_GPUS=1,2,3    # training rollouts
export EVAL_TP=4                # eval tensor parallel
export VLLM_GPU_UTIL=0.85
```

- **GPU 0:** HF QLoRA learner (stays loaded across both phases)
- **GPUs 1–3:** vLLM DP rollout workers (stay up between Phase 1 and Phase 2; shut down after C2 save, before eval)

**Why C1 eval is deferred:** `EVAL_TP=4` needs all four GPUs. Running C1 eval between phases would evict the learner + AdamW from GPU 0 and break continuous optimizer state. C1 checkpoint is saved atomically after Phase 1; dev eval runs after training teardown.

---

## 6. Preflight (required)

```bash
cd experiments/nestful_synthetic_curriculum_v3

bash -n scripts/v5/run_two_phase_grpo.sh

python scripts/training/preflight_training_datasets.py \
  data/training_ready_v5/filtered/phase1_stage2_train.jsonl \
  data/training_ready_v5/filtered/phase2_stage3_plus_stage2_replay.jsonl \
  --report /tmp/preflight_report.json

python tests/test_two_phase_pipeline.py -v
```

Preflight must confirm: **429** Stage 2, **466** Phase 2, **895** rows total, 100% executable replay, registry **v5.0.2**, no duplicate sample IDs.

Dry-run manifest (no training):

```bash
export RUN_DIR=outputs/runs/_dry_run
mkdir -p "$RUN_DIR"
python scripts/training/run_two_phase_v5_grpo.py --run-dir "$RUN_DIR" --dry-run
```

---

## 7. Smoke test (4 GPU, end-to-end)

Must cover: C0 eval → Phase 1 steps → atomic C1 → Phase 2 transition (same optimizer) → C2 → teardown → C1/C2 eval → compare.

```bash
cd /workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3

export RUN_DIR="outputs/runs/two_phase_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

export MAX_TRAIN_TASKS=8
export DEV_MAX_TASKS=20
export USE_VLLM=1
export ROLLOUT_DP_GPUS=1,2,3
export EVAL_TP=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/v5/run_two_phase_grpo.sh 2>&1 | tee "$RUN_DIR/console.log"
```

**Post-smoke checklist** (grep logs / inspect state):

| Check | Where |
|-------|-------|
| `global_step` monotonic Phase 1 → Phase 2 | `two_phase_state.json`, train logs |
| Same `optimizer_id` across phases | `two_phase_state.json` → `phase1_train.optimizer_id` == `phase2_train.optimizer_id` |
| Phase 2 loads C1 adapter | `phase2_train.continuous_from_phase1: true` |
| Rollout workers synced to C1 | console: `learner checkpoint/version: C1`, `all workers acknowledged` |
| C1/C2 checkpoints readable | `$RUN_DIR/checkpoints/C1/checkpoint_manifest.json` |
| All GPUs free before eval | console: `training session closed; all GPUs free for eval` |
| Same 20 dev tasks for C0/C1/C2 | eval dirs under `$RUN_DIR/eval/` |
| Epoch coverage clean | `phase*/train/epoch_1/train_summary.json` → `epoch_coverage.ok` |
| Manifest hashes | `run_manifest.json` |

---

## 8. Full training (tmux)

```bash
tmux new -s two_phase_grpo
cd /workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3

export USE_VLLM=1 ROLLOUT_DP_GPUS=1,2,3 EVAL_TP=4 VLLM_GPU_UTIL=0.85
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SEED=42 DATA_SEED=42 ROLLOUT_SEED=42
export WANDB_PROJECT=nestful-v5-curriculum WANDB_API_KEY=...

export RUN_DIR="outputs/runs/two_phase_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

bash scripts/v5/run_two_phase_grpo.sh 2>&1 | tee "$RUN_DIR/console.log"
```

Detach: `Ctrl-b d` · Reattach: `tmux attach -t two_phase_grpo`

**Automatic workflow order:**

1. preflight  
2. eval C0 (base model)  
3. open continuous training session  
4. Phase 1 — exactly 429 rows, 1 epoch  
5. atomically save C1 (`checkpoints/C1.tmp` → verify → rename)  
6. Phase 2 — exactly 466 rows, 1 epoch (same optimizer + global_step, rollout sync to C1)  
7. atomically save C2  
8. teardown training session (free all GPUs)  
9. eval C1 (deferred)  
10. eval C2  
11. paired compare C0/C1/C2  

Checkpoints: `$RUN_DIR/checkpoints/C1/`, `$RUN_DIR/checkpoints/C2/` (each with `checkpoint_manifest.json`)

Epoch coverage audit: `$RUN_DIR/phase1/train/epoch_1/train_summary.json` → `epoch_coverage`  
Phase 2 stage split: `$RUN_DIR/phase2/stage_split_metrics.json`

---

## 9. Monitoring

```bash
RUN_DIR=outputs/runs/two_phase_YYYYMMDD_HHMMSS

tail -f "$RUN_DIR/console.log"
tail -f "$RUN_DIR/logs/phase1_train.log"    # if subprocess logs exist
cat "$RUN_DIR/two_phase_state.json" | python -m json.tool
cat "$RUN_DIR/run_manifest.json" | python -m json.tool
cat "$RUN_DIR/eval/compare_C0_C1_C2/final_compare_report.json" | python -m json.tool
```

---

## 10. Resume (phase-level only)

```bash
export RUN_DIR="outputs/runs/two_phase_YYYYMMDD_HHMMSS"   # SAME dir
export RESUME=1

bash scripts/v5/run_two_phase_grpo.sh 2>&1 | tee -a "$RUN_DIR/console.log"
```

| Mode | What happens |
|------|----------------|
| **Same-process run** | Phase 1 → Phase 2 keeps AdamW + `global_step` (preferred) |
| **`--resume` after crash** | Skips completed steps; keeps C1; discards incomplete C2; restarts Phase 2 from C1 with **fresh optimizer** — NOT exact step resume |
| **Crash during C2 save** | Incomplete `checkpoints/C2` / `C2.tmp` removed on resume; Phase 2 re-run from C1 |
| **Exact step resume** | **Not supported** (Adam state not persisted to disk) |

The launcher trap sends SIGTERM only to rollout worker PIDs listed in `$RUN_DIR/logs/rollout_worker_pids.json` (never `killall python`).

---

## 11. Standalone eval (C0 / C1 / C2)

Dev set = fixed 200 IDs from `nestful_devtest_manifest.json`, disjoint from test.

```bash
cd experiments/nestful_synthetic_curriculum_v3
export USE_VLLM=1 EVAL_TP=4

# C0
CHECKPOINT= OUT_DIR=$RUN_DIR/eval/C0_baseline \
  bash scripts/v5/dev_eval.sh

# C1
CHECKPOINT=$RUN_DIR/checkpoints/C1 OUT_DIR=$RUN_DIR/eval/C1_phase1 \
  bash scripts/v5/dev_eval.sh

# C2
CHECKPOINT=$RUN_DIR/checkpoints/C2 OUT_DIR=$RUN_DIR/eval/C2_phase2 \
  bash scripts/v5/dev_eval.sh

# Compare (win rate, F1 func/param, full/partial seq, executability,
# under-calling, unsupported traces, breakdown by call count)
python scripts/eval/final_eval_v5.py compare \
  --baseline $RUN_DIR/eval/C0_baseline \
  --best     $RUN_DIR/eval/C1_phase1 \
  --final    $RUN_DIR/eval/C2_phase2 \
  --out      $RUN_DIR/eval/compare_C0_C1_C2
```

**Headline test eval** — run only after checkpoint selection (not during training):

```bash
TEST_PATH="../nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl"
N_TEST=$(python -c "print(sum(1 for l in open('$TEST_PATH') if l.strip()))")

python scripts/eval/final_eval_v5.py run \
  --label C2_test \
  --checkpoint $RUN_DIR/checkpoints/C2 \
  --out-dir $RUN_DIR/eval/C2_nestful_test \
  --eval-set "$TEST_PATH"
echo "Evaluated $N_TEST test tasks"
```

C0/C1/C2 dev evals use identical: task IDs, ReAct prompt template, parser, full IBM executor, temp=0, top_p=1, 1 rollout.

---

## 12. Hyperparameters (defaults)

| Knob | Value |
|------|-------|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Epochs | 1 per phase (no auto 3rd epoch) |
| `num_generations` | 8 |
| `learning_rate` | 3e-7 |
| `kl_beta` | 0.15 |
| train `temperature` / `top_p` | 1.0 / 0.95 |
| `max_grad_norm` | 1.0 |
| `executor.mode` | `synthetic` |
| `reward.train_policy` | `execution_aware_v3_2_dense` |

---

## 13. Known limitations

1. **No LR scheduler / grad scaler** in current `grpo_train.py` — only AdamW (constant LR).
2. **Adam state not saved to disk** — crash resume reloads last adapter, not optimizer momentum.
3. **Official win rate** needs IBM functions at `nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/executable_functions/`.
4. **4 GPUs** required for production throughput; smoke works with caps on CPU/single-GPU (slow).
5. **Stage 2 → Stage 2 control run** (same optimizer reset policy) should use the same session code path if you add a control — not a separate legacy launcher.

---

## Changed files (this revision)

- `nestful_mtgrpo_minimal/grpo_train.py` — continuous training kwargs (`optimizer`, `global_step_start`, …)
- `nestful_mtgrpo_minimal/vllm_dp_pool.py` — `worker_pids` property
- `scripts/training/run_two_phase_v5_grpo.py` — in-process orchestrator
- `scripts/training/two_phase_train_session.py` — single-process session
- `scripts/training/two_phase_utils.py` — repro, assertions, epoch coverage, dev/test hygiene
- `scripts/training/preflight_training_datasets.py` — duplicate ID check
- `scripts/v5/run_two_phase_grpo.sh` — RUN_DIR guard, seeds, trap cleanup
- `tests/test_two_phase_pipeline.py` — unit tests
- `scripts/training/RUNBOOK_TWO_PHASE_GRPO.md` — this document
