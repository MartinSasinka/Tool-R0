# RunPod Pilot3 signal-probe runbook

## One command (after syncing the frozen bundle)

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
```

Flags: `--dry-run`, `--resume`, `--stage p2|select|p3|report`.

## Local freeze (dev machine)

```bash
cd experiments/targeted_tool_data_factory
python scripts/run_pilot3.py --dry-run
python scripts/run_pilot3.py                # needs OPENROUTER_API_KEY
python runpod_bundle_pilot3/build_bundle.py
```

GRPO is **not** started by the probe. Read `SIGNAL_PROBE_REPORT.md` first.
