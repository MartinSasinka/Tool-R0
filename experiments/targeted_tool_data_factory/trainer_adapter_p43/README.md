# P43 trainer adapter

Exposes **Pilot4.3** `ops.build_ops()` (199 ops / 597 surface names) to
`nestful_mtgrpo_minimal` via `executor.mode=synthetic`.

```bash
export SYNTHETIC_TOOLS_DIR=$PWD/trainer_adapter_p43
python trainer_adapter_p43/preflight_gold_replay.py \
  --data outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl
```

Do **not** use `trainer_adapter/` (pilot2/3, 190 tools) for PROFILE_1000.
