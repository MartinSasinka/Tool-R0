# Pure Stage 3 — Two-Epoch GRPO Overnight (RunPod, 4× GPU)

Isolates **H1**: can clean Stage 3 data alone (326 rows, no Stage 2 / replay)
improve nested tool use starting from the base model?

Does **not** change reward, γ, λ_episode, turn scores, LR, KL, or prompt.

**Canonical entry points:**

| Role | Path |
|------|------|
| Overnight launcher | `scripts/v5/run_pure_stage3_two_epoch_overnight.sh` |
| Orchestrator | `scripts/training/run_pure_stage3_two_epoch.py` |
| Materialize 326 | `scripts/data/materialize_pure_stage3.py` |
| Syntax audit | `scripts/data/audit_stage3_nestful_syntax.py` |
| Preflight | `scripts/training/preflight_training_datasets.py` |
| Session (reuse) | `scripts/training/two_phase_train_session.py` |
| Credit probe | `scripts/analysis/pure_stage3_credit_probe.py` |
| Eval | `scripts/eval/final_eval_v5.py` |
| Deps | `nestful_mtgrpo_minimal/install_deps.sh` |

**Dataset:** `data/training_ready_v5/filtered/stage3_train_ready.jsonl` (326 rows),
extracted from `phase2_stage3_plus_stage2_replay.jsonl` (original untouched).

**Checkpoints:** `S3_E1`, `S3_E2` under `$RUN_DIR/checkpoints/`.

**Startup hard abort:** `executor=synthetic`, `reward=execution_aware_v3_2_dense`,
registry `v5.0.2`, exactly 326 Stage 3 rows, no Stage 2, 100% replay, no `gold_replay`.

---

## 1. Clone & deps

```bash
cd /workspace
git clone https://github.com/YOUR_ORG/Tool-R0.git
cd Tool-R0
git rev-parse HEAD

bash experiments/nestful_mtgrpo_minimal/install_deps.sh
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh
```

---

## 2. Tokens & GPU

```bash
export HF_TOKEN="hf_..."
huggingface-cli login --token "$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface

export WANDB_API_KEY="..."
export WANDB_PROJECT="nestful-v5-pure-stage3"
export WANDB_ENTITY="your-team"   # optional
export WANDB_MODE="online"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_VLLM=1
export ROLLOUT_DP_GPUS=1,2,3
export EVAL_TP=4
export VLLM_GPU_UTIL=0.85

export SEED=42 DATA_SEED=42 ROLLOUT_SEED=42
export EPOCHS=2

nvidia-smi
```

---

## 3. Materialize + syntax audit + preflight

```bash
cd /workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3

bash -n scripts/v5/run_pure_stage3_two_epoch_overnight.sh

python scripts/data/materialize_pure_stage3.py

python scripts/data/audit_stage3_nestful_syntax.py \
  --input data/training_ready_v5/filtered/stage3_train_ready.jsonl \
  --report-dir reports/stage3_syntax_audit

# Expected verdict for this curriculum: NO_MISMATCH
# (Tool-R0 ReAct teaches $varN.field$; Stage 3 already matches; executor
#  also accepts NESTFUL $var_N.field$. No derived file written.)

python scripts/training/preflight_training_datasets.py \
  data/training_ready_v5/filtered/stage3_train_ready.jsonl \
  --report /tmp/pure_stage3_preflight.json

python tests/test_pure_stage3_pipeline.py -v
```

---

## 4. Smoke test (4 GPU)

```bash
cd /workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3

export RUN_DIR="outputs/runs/pure_stage3_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

export MAX_TRAIN_TASKS=8
export DEV_MAX_TASKS=20
export SKIP_TEST_EVAL=1          # optional for smoke
export ALLOW_FEWER_GPUS=0

bash scripts/v5/run_pure_stage3_two_epoch_overnight.sh 2>&1 | tee "$RUN_DIR/console.log"
```

**Smoke checklist:** `SUCCESS` marker; `checkpoints/S3_E1` + `S3_E2` with different
`adapter_hash`; `epoch_1/epoch_coverage.json` + `epoch_2/...` ok; optimizer continuous
in same process (`optimizer_unchanged=true` when no mid-run resume).

Unset smoke caps before full run:

```bash
unset MAX_TRAIN_TASKS DEV_MAX_TASKS SKIP_TEST_EVAL
```

---

## 5. Full overnight (tmux)

```bash
tmux new -s pure_stage3

cd /workspace/Tool-R0/experiments/nestful_synthetic_curriculum_v3

export WANDB_PROJECT="nestful-v5-pure-stage3"
export WANDB_API_KEY="..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_VLLM=1 ROLLOUT_DP_GPUS=1,2,3 EVAL_TP=4 VLLM_GPU_UTIL=0.85
export SEED=42 DATA_SEED=42 ROLLOUT_SEED=42 EPOCHS=2

export RUN_DIR="outputs/runs/pure_stage3_2ep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

bash scripts/v5/run_pure_stage3_two_epoch_overnight.sh 2>&1 | tee "$RUN_DIR/console.log"
```

Detach: `Ctrl-b d` · Reattach: `tmux attach -t pure_stage3`

**Automatic order:**

1. env / GPU / IBM / disk assertions  
2. materialize 326 Stage 3  
3. syntax audit → `NO_MISMATCH` or derive  
4. preflight  
5. dry-run manifest  
6. C0 dev eval  
7. Epoch 1 → atomic `S3_E1` → weight sync  
8. Epoch 2 (same AdamW + global_step) → atomic `S3_E2`  
9. teardown  
10. credit probe (100 groups)  
11. E1/E2/C0 dev eval + compare  
12. C0 + E2 nestful_test + compare  
13. reports + artefact SHA-256  
14. `SUCCESS` marker  

---

## 6. Monitoring

```bash
RUN_DIR=outputs/runs/pure_stage3_2ep_YYYYMMDD_HHMMSS

tail -f "$RUN_DIR/console.log"
cat "$RUN_DIR/overnight_state.json" | python -m json.tool
cat "$RUN_DIR/run_manifest.json" | python -m json.tool
cat "$RUN_DIR/pure_stage3_training_comparison.json" | python -m json.tool
ls "$RUN_DIR/checkpoints/S3_E1" "$RUN_DIR/checkpoints/S3_E2"
```

Reports:

- `reports/stage3_nestful_syntax_audit.md`
- `reports/pure_stage3_credit_probe_summary.md`
- `reports/PURE_STAGE3_TWO_EPOCH_RESULT.md`

---

## 7. Resume

```bash
export RUN_DIR="outputs/runs/pure_stage3_2ep_YYYYMMDD_HHMMSS"
export RESUME=1
bash scripts/v5/run_pure_stage3_two_epoch_overnight.sh 2>&1 | tee -a "$RUN_DIR/console.log"
```

| Situation | Behaviour |
|-----------|-----------|
| Crash before E1 done | Restart epoch 1 from base |
| E1 done, crash before E2 | Load `S3_E1` adapter, **fresh AdamW** (not on disk) — logged in `resume_events` |
| Same-process E1→E2 | Shared optimizer + monotonic `global_step` (preferred) |
| Train OK, eval OOM on GPU 0 | Teardown now unloads the HF learner before eval; `RESUME=1` skips done train steps and re-runs pending evals. Before resume: `nvidia-smi` and kill leftover `VLLM::EngineCore` / `run.py` if any |

Exact step resume is **not** supported.

**Eval teardown note:** `session.close()` shuts down rollout workers **and** unloads the QLoRA learner + AdamW from GPU 0. Without that, `EVAL_TP=4` at `VLLM_GPU_UTIL=0.85` fails with ~9 GB free on cuda:0.

Even with learner unloaded (~29 GB free on cuda:0), util=0.85 can still fail the vLLM startup check (needs free ≥ 0.85×total while rank-0 holds a CUDA context). `vllm_generate` now auto-caps util from measured free VRAM; before each eval the orchestrator also runs `prep_gpus_for_eval` (pkill leftover EngineCore + wait).

---

## 8. Standalone eval

```bash
cd experiments/nestful_synthetic_curriculum_v3
export USE_VLLM=1 EVAL_TP=4

LABEL=baseline unset CHECKPOINT
OUT_DIR=$RUN_DIR/eval/C0_dev bash scripts/v5/dev_eval.sh

LABEL=best CHECKPOINT=$RUN_DIR/checkpoints/S3_E1 \
  OUT_DIR=$RUN_DIR/eval/S3_E1_dev bash scripts/v5/final_eval.sh

LABEL=final CHECKPOINT=$RUN_DIR/checkpoints/S3_E2 \
  EVAL_SET=../nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl \
  OUT_DIR=$RUN_DIR/eval/S3_E2_test bash scripts/v5/final_eval.sh
```

---

## 9. Hyperparameters (fixed for this experiment)

| Knob | Value |
|------|-------|
| Base | `Qwen/Qwen3-4B-Instruct-2507` |
| Epochs | 2 on same 326 Stage 3 rows |
| Start | base (not C1) |
| Stage 2 / replay | none |
| `num_generations` | 8 |
| `learning_rate` | 3e-7 |
| `kl_beta` | 0.15 |
| temp / top_p (train) | 1.0 / 0.95 |
| `gamma` / `lambda_episode` | 1.0 / 1.0 (unchanged) |
| reward | `execution_aware_v3_2_dense` |
| executor | `synthetic` |

---

## 10. Known limitations

1. AdamW state is **not** saved to disk — mid-run resume after E1 uses a fresh optimizer.  
2. Credit probe recomputes `G_t` / advantages from `train_log`; per-call `name_ok` etc. are not in the log (null).  
3. Official test eval is C0 vs **E2** only (E1 test not used for selection).  
4. Syntax audit `NO_MISMATCH` means Tool-R0 stack compatibility; NESTFUL gold prefers `$var_N` but scorer/executor accept both.  
5. This run does **not** ablate credit assignment — reward/`_turn_returns` stay fixed.

---

## Comparison intent

After the night, compare:

- original curriculum **C0 / C1 / C2** (`two_phase_20260718_192902`)
- vs this **C0 / S3_E1 / S3_E2**

Question answered: *Do two exposures of pure Stage 3 from base help without Stage 2?*
