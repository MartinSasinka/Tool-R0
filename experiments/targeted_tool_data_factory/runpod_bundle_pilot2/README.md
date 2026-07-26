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

## Run the signal probe FIRST

Before spending GPU hours on GRPO, check that the 160 frozen train tasks
actually produce a usable rollout/reward signal for the base checkpoint. The
probe is inference only — no optimizer, no gradients, no LoRA update:

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_signal_probe_4gpu.sh
```

P2 rolls out all 160 tasks × 4 completions across the 4 GPUs, P3 re-probes up to
64 boundary tasks × 8 completions, and the report ends in a verdict:

| verdict | gate | what to do |
|---|---|---|
| `PASS` | dead ≤ 50 % and reward ordering valid | train on `recommended_phase1_train.jsonl` |
| `CONDITIONAL` | dead 50–70 % with real process variance | A4 only, expect slow movement |
| `STOP` | dead > 70 % or reward ordering broken | fix the reward or the task mix first |

Everything lands in `../outputs/runpod_pilot2/signal_probe/`. Interrupted runs
resume from the content-hash cache with `--resume`; `--dry-run` prints the whole
plan without generating anything.

## After the probe: Phase-1 canary (not full D1-160)

Do **not** jump to full D1 on 160 tasks. First fix the reward audit offline,
verify the 80-task Phase-1 subset, then run a small GRPO canary:

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_phase1_canary_4gpu.sh \
  --probe-dir experiments/targeted_tool_data_factory/outputs/runpod_pilot2/signal_probe
```

Pipeline: offline reward audit (no new inference) → Phase-1 verify (80 / replay /
leakage / NESTFUL JSD) → C1 train on `recommended_phase1_train.jsonl` with the
selected reward variant (Qwen3-4B-Instruct-2507, 8 rollouts, ~20 optimizer steps,
GPU0 learner + GPU1-3 workers) → eval C0 vs C1 on structural held-out 80 and
frozen NESTFUL-500 → `C0_VS_C1_PHASE1_REPORT.md/json`. Full NESTFUL-1661 is
never started.

Optional later re-probe of deferred Phase-2 tasks (not started by the canary):

```bash
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_deferred_reprobe_4gpu.sh
```

| file | purpose |
|---|---|
| `run_signal_probe_4gpu.sh` | signal probe entry point (inference only, 4 GPUs) |
| `signal_probe_worker.py` | one-GPU rollout worker; writes per-rollout records |
| `signal_probe_analyze.py` | groups, P3 selection, report, Phase-1 / Phase-2 split |
| `signal_probe_lib.py` | group metrics, selection, reward-ordering audit, verdict |
| `offline_reward_audit.py` | rescore stored rollouts; select safest reward variant |
| `verify_phase1_subset.py` | 80-task Phase-1 gates (replay, leakage, NESTFUL JSD) |
| `run_phase1_canary_4gpu.sh` | Phase-1 GRPO canary (C0 vs C1; not full D1) |
| `run_phase1_train.py` | C1 train entry with selected reward patch |
| `run_deferred_reprobe_4gpu.sh` | optional re-probe of deferred_phase2_tasks.jsonl |
| `test_signal_probe.py` | probe tests (no GPU) |
| `test_phase1_next.py` | Phase-1 audit / verify / canary tests (no GPU) |
| `run_all_4gpu.sh` | entry point; `--c0-vs-d1` skips D0 training and reports C0 vs D1 |
| `test_c0_vs_d1_mode.sh` | dry-run gate: no D0 trainer, C0 eval kept |
| `test_c0_vs_d1_unit.py` | unit checks for eval/report without a D0 checkpoint |
| `install.sh` | idempotent dependency install |
| `build_bundle.py` | freezes the data and rewrites `MANIFEST.sha256.json` (run on the dev machine, not the pod) |
| `verify_hashes.py` | fails if any frozen artefact changed |
| `check_config_parity.py` | fails if D0 and D1 differ outside the dataset and its registry |
| `check_canary_gates.py` | fails if the canary did not really execute factory tools |
| `run_eval_all.py` | six eval jobs across GPU0-3 |
| `make_paired_report.py` | paired D0 vs D1 / C0 vs C1 statistics |
| `run_full_nestful_test.sh` | confirmation run on all 1661 NESTFUL tasks, disabled by default |

Re-freeze after changing anything under `data/`, `configs/` or any script:

```bash
python experiments/targeted_tool_data_factory/runpod_bundle_pilot2/build_bundle.py
```
