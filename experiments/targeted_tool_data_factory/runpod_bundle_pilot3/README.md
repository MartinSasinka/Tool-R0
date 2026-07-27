# runpod_bundle_pilot3

Frozen bundle for the **pilot3** dataset (1000 tasks:
600 train / 200 held-out / 200 reserve). Pilot2 under `runpod_bundle_pilot2/`
is untouched.

## Train + NESTFUL-500 (what you usually want)

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
# full 600 train, then eval only the new ckpt on NESTFUL-500 (no C0)
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_train_nestful500_4gpu.sh
# or a 200-task slice of the frozen train file:
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_train_nestful500_4gpu.sh --train-n 200
```

## Optional signal probe (inference only, no GRPO)

Probe = rollouts without training, to measure which tasks have reward signal
before a canary. Skip it if you already want to train.

```bash
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
```

## Rebuild freeze

```bash
python experiments/targeted_tool_data_factory/runpod_bundle_pilot3/build_bundle.py
python experiments/targeted_tool_data_factory/runpod_bundle_pilot3/verify_hashes.py
```
