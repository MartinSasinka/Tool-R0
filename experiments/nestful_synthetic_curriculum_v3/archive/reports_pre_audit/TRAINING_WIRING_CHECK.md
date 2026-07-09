# Training Wiring Check

Date: 2026-07-02 (post pilot-prep)  
**Training started: NO**

## Checklist

| item | status | notes |
|------|--------|-------|
| `run_curriculum_v3.sh` defaults `STAGES="1 2"` | **PASS** | |
| Stage 3/4 blocked without gates | **PASS** | shell exits if STAGES contains 3 or 4 |
| Preflight gates auto-run | **PASS** | validate → audit → gold replay → tool realism → preflight |
| `ALLOW_PROTOTYPE_TRAINING=1` required | **PASS** | enforced for PASS_PROTOTYPE_ONLY |
| `lib/reward_motif.py` importable | **PASS** | verified locally |
| `run.py` importable | **PASS** | loads partial driver |
| Reward actually used in training | **PASS** | `v3/run.py` hooks `_select_train_reward` → patches `execution_aware_v2_1_motif` **after** partial would override |
| `reward_v2_1_motif.yaml` | **PASS** | weights documented; applied via `lib/reward_motif.py` |
| Base training config | **PASS** | `CONFIG=$PARTIAL/config.yaml` (not bare training_v3.yaml) |
| `training_v3.yaml` merge | **PARTIAL** | key overrides via `EXTRA_TRAIN_OVERRIDES_STR`; full YAML merge still optional |
| Stage file mapping | **PASS** | symlinks to `epoch_{1..4}_{N}call.jsonl` |
| Output root | **PASS** | `experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp>/` |

## Reward wiring (fixed)

Previous issue: `v3/run.py` patched reward before partial `_select_train_reward`, which overwrote with `partial_gold_trace`.

**Fix:** `_hook_select_train_reward()` intercepts `execution_aware_v2_1_motif` policy and applies motif patch without falling through to partial default.

## Stage data (pilot)

| stage | tasks | file |
|-------|------:|------|
| stage1 | 417 | `curriculum_v3/stage1_linear_simple.jsonl` |
| stage2 | 223 | `curriculum_v3/stage2_reference_reuse.jsonl` |

## Pilot runnable?

**YES** — with `ALLOW_PROTOTYPE_TRAINING=1` on pod after DRY RUN.

Not final-experiment-ready (tool realism partial, not IBM tools).

## DRY RUN command

```bash
cd /workspace/Tool-R0
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```
