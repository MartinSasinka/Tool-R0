# Pod resume — curriculum v3 pilot

## Symptom (train-stage2-e1 crash)

```
FileNotFoundError: adapter checkpoint directory does not exist:
.../stage_1/checkpoints/adapter_epoch_1
```

**Cause:** With `MAX_EPOCHS_PER_STAGE=2`, epoch 2 training overwrites the generic
`adapter_epoch_1` slot and renames it to `adapter_epoch_2`. If epoch 1 was the
best checkpoint, the path `adapter_epoch_1` no longer exists when stage 2 starts.

Fixed in `nestful_mtgrpo_minimal/run_curriculum.sh` (snapshot before overwrite).
Pull latest before resuming.

## 1. Inspect what survived on the pod

```bash
RUN=experiments/nestful_synthetic_curriculum_v3/outputs/runs/20260702_112150
ls -la "$RUN/stage_1/checkpoints/"
ls -la "$RUN/best_react_win_adapter/" 2>/dev/null || true
cat "$RUN/stage_1/epoch_summary.jsonl" 2>/dev/null
```

Typical layout after stage 1 with 2 epochs (pre-fix):

| Path | Meaning |
|------|---------|
| `stage_1/checkpoints/adapter_epoch_2` | Last epoch weights (usually exists) |
| `stage_1/checkpoints/adapter_epoch_1` | **Missing** if epoch 2 ran |
| `best_react_win_adapter/` | Best validation ReAct Win (if val_eval ran) |

Pick the checkpoint to resume from (prefer `best_react_win_adapter` if present,
else `adapter_epoch_2`, else highest `adapter_epoch_*` with `adapter_config.json`).

## 2. Resume stage 2 only

```bash
cd /workspace/Tool-R0/Tool-R0

RUN=experiments/nestful_synthetic_curriculum_v3/outputs/runs/20260702_112150
CKPT="$RUN/stage_1/checkpoints/adapter_epoch_2"   # adjust after ls above

# Optional: use best-by-val if it exists
if [ -f "$RUN/best_react_win_adapter/adapter_config.json" ]; then
  CKPT="$RUN/best_react_win_adapter"
fi

ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 \
  ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  OUTPUT_ROOT="$RUN" \
  CHECKPOINT_IN="$CKPT" \
  STAGES="2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

`OUTPUT_ROOT` reuses the same run dir (symlinks in `data_base/` are recreated).
`CHECKPOINT_IN` seeds stage 2 epoch 1.

## 3. After pulling the checkpoint fix

Re-run is only needed if you want a clean stage 1 with both epochs kept on disk.
For the current pilot, resuming from `adapter_epoch_2` (or `best_react_win_adapter`)
is fine — epoch 2 is usually ≥ epoch 1 on strict_pass.

## 4. Manual sanity check before train

```bash
test -f "$CKPT/adapter_config.json" && echo "OK: $CKPT" || echo "BAD: $CKPT"
```
