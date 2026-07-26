# runpod_bundle_pilot2

Frozen, self-contained bundle for the D0-vs-D1 experiment on a 4-GPU RunPod
instance. `data/` is data-frozen on purpose: nothing is regenerated on the pod.

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
# full D0 vs D1
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh
# C0 vs D1 only (skip D0 training; keep C0 eval)
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh --c0-vs-d1
```

Full documentation: `../docs/RUNPOD_PILOT2_RUNBOOK.md`.

| file | purpose |
|---|---|
| `run_all_4gpu.sh` | entry point; `--c0-vs-d1` skips D0 training and reports C0 vs D1 |
| `test_c0_vs_d1_mode.sh` | dry-run gate: no D0 trainer, C0 eval kept |
| `test_c0_vs_d1_unit.py` | unit checks for eval/report without a D0 checkpoint |
| `install.sh` | idempotent dependency install |
| `build_bundle.py` | freezes the data and rewrites `MANIFEST.sha256.json` (run on the dev machine, not the pod) |
| `verify_hashes.py` | fails if any frozen artefact changed |
| `check_config_parity.py` | fails if D0 and D1 differ outside the dataset and its registry |
| `check_canary_gates.py` | fails if the canary did not really execute factory tools |
| `run_eval_all.py` | six eval jobs across GPU0-3 |
| `make_paired_report.py` | paired D0 vs D1 statistics |
| `run_full_nestful_test.sh` | confirmation run on all 1661 NESTFUL tasks, disabled by default |

Re-freeze after changing anything under `data/`, `configs/` or any script:

```bash
python experiments/targeted_tool_data_factory/runpod_bundle_pilot2/build_bundle.py
```
