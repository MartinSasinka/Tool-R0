# RunPod pilot2 runbook — D0 vs D1

Generated 2026-07-26 12:03 UTC.

## The experiment

One manipulated variable: the training dataset.

| | D0 | D1 |
|---|---|---|
| training data | 160 old Stage-3 tasks | 160 pilot2 factory tasks |
| base checkpoint | C0 | C0 |
| reward | A4_GATED_VERIFIABLE | A4_GATED_VERIFIABLE |
| seed | 20260726 | 20260726 |
| LR / KL / LoRA / optimizer / credit / decoding | identical | identical |
| rollouts | 8 | 8 |
| optimizer-step budget | identical | identical |
| GPUs | 0 learner, 1-3 rollout workers | same |
| tool registry | legacy `synthetic_tools.py` | factory adapter |

### The one confound you cannot remove

The tool registry travels with the dataset. A Stage-3 task can only be executed by the legacy registry and a pilot2 task only by the factory adapter, so `SYNTHETIC_TOOLS_DIR` is set per arm rather than globally (setting it globally would make every D0 tool call fail). This means the comparison is dataset **plus** executor implementation, not dataset alone. It cannot be avoided without rewriting one dataset against the other's registry, which would destroy the thing being tested. What it costs is bounded: both executors are deterministic and both are verified by a 100 % gold-replay preflight before training, so neither arm is being scored against a broken oracle. Say this out loud in the write-up rather than claiming a clean single-variable manipulation.

D0 and D1 run **sequentially**. Running them concurrently on the same pod would let them contend for GPU memory and quietly change the effective batch timing, which is exactly the kind of hidden difference this design exists to avoid.

## One command

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...            # required
export WANDB_API_KEY=...       # optional
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh
```

## What it does, in order

1. refuses to start unless 4 GPUs are visible;
2. installs dependencies (idempotent);
3. verifies every frozen artefact against `MANIFEST.sha256.json`;
4. checks D0/D1 config parity — any difference outside the dataset and its registry aborts;
5. runs the gold-replay preflight through the real trainer executor (160 train + 80 held-out, must be 100 %);
6. runs the dispatch/executor canary: 24 stratified pilot2 tasks x 8 rollouts, A1 first then A4;
7. gates the canary: configured policy == resolved policy on every train row, A1 and A4 produce *different* rewards on hash-matched completions, no NaN/Inf, no terminal ordering inversion, executor and trajectory logging present, and every executed tool is a factory surface with no legacy fallback;
8. only on PASS: trains D0, then D1;
9. evaluates C0, D0 and D1 across GPU0-3 in parallel on the structural held-out 80 (G and A tracks reported separately) and the frozen NESTFUL diagnostic-500;
10. writes the paired report: Win Rate, Function F1, Parameter F1, executability, gained/lost, paired bootstrap 95 % CI, exact McNemar, failure taxonomy.

## Why the canary gate is not optional

Round 1 of the reward ablation trained five arms that all silently resolved to the same reward. The canary exists so that a dispatch regression costs twenty minutes instead of an entire experiment. If A1 and A4 produce identical rewards on identical completions, the script stops and no training starts.

## Useful flags

```bash
bash run_all_4gpu.sh --dry-run        # print every command, train nothing
bash run_all_4gpu.sh --resume         # continue interrupted runs
bash run_all_4gpu.sh --skip-canary    # reuse an earlier PASS
bash run_all_4gpu.sh --stage eval     # re-run one stage only
```

## After the run

Read `outputs/runpod_pilot2/D0_VS_D1_REPORT.md`. Interpret it carefully: a D1 gain on the structural held-out only shows the model learned the new data. Transfer is the NESTFUL diagnostic-500 number. Reporting the first as if it were the second is the mistake that made Round 1 uninterpretable.

The full NESTFUL test (1661) is a separate, deliberately gated command:

```bash
CONFIRM_FULL_NESTFUL=yes bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_full_nestful_test.sh
```

## Bundle contents

| file | purpose |
|---|---|
| `run_all_4gpu.sh` | the single entry point |
| `install.sh` | dependency install |
| `verify_hashes.py` | manifest verification |
| `check_config_parity.py` | D0/D1 differ only in the dataset |
| `check_canary_gates.py` | pilot2 executor gates on the canary |
| `run_eval_all.py` | 6 eval jobs across GPU0-3 |
| `make_paired_report.py` | paired D0 vs D1 statistics |
| `run_full_nestful_test.sh` | disabled-by-default confirmation run |
| `build_bundle.py` | freezes the data and writes the manifest |
| `data/` | frozen pilot2 train/held-out/reserve, canonical, canary, D0 data, NESTFUL diagnostic-500 |
| `configs/` | D0 and D1 run configs |
| `MANIFEST.sha256.json` | sha256 of every frozen artefact |
