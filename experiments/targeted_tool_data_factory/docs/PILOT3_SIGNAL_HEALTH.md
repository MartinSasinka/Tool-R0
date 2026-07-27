# Pilot3 signal-health report

Generated 2026-07-27 03:15 UTC.

## Local student probe

The factory `probe` step ran with `template_only` / `--no-llm` unless a local
OpenAI-compatible endpoint was available. Full signal measurement is the
**RunPod signal probe** on the frozen 600-task train split:

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
```

That probe:
- rolls out **all 600** train tasks × 4 (P2);
- re-probes a boundary subset × 8 (P3);
- selects a NESTFUL-matched Phase-1 subset with terminal/process mixed signal
  (default target 400; hard structural buckets are not dropped for difficulty);
- never trains.

After the probe, replace this stub with
`outputs/runpod_pilot3/signal_probe/SIGNAL_PROBE_REPORT.md`.

## Pipeline timings (this machine)

| step | wall_s |
|---|---|
| export | 16.22 |
| generate | 118.81 |
| generate_expand | 58.05 |
| paraphrase | 8013.04 |
| probe | 2.37 |
| report | 5.13 |
| select | 20.24 |
| split | 7.16 |
| validate | 2098.44 |
