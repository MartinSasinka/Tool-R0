# Trainer integration

Generated 2026-07-26 12:03 UTC.

## The problem

The GRPO trainer resolves tools through a `synthetic_tools` module: a global `TOOLS` mapping from tool name to a callable plus a JSON-Schema. The factory instead has *primitives* with *surfaces* — the same primitive can be exposed under several names with different parameter names. Those two contracts only agree if every exported tool name maps to exactly one parameter signature.

pilot1 violated that: the same surface name appeared with generic (`arg_0`, `arg_1`) and semantic parameter names, so the adapter saw schema drift and the preflight refused the data. That is why pilot1 cannot be trained on with this adapter, and why pilot2 enforces surface-name uniqueness at generation time rather than patching it at export time.

## The adapter

`trainer_adapter/lib/synthetic_tools.py` is a drop-in replacement discovered through `SYNTHETIC_TOOLS_DIR`. It:

- builds `TOOLS` by walking every factory surface, so the trainer executes the   real deterministic primitive rather than a re-implementation;
- maps surface parameter names onto the canonical primitive parameters;
- validates arguments strictly (unknown key, missing key, wrong type all raise   instead of silently coercing);
- resolves `$varN.output_M$` references against previous observations;
- returns observations in the trainer's own format;
- exposes `registry_hash()` / `factory_hashes()` so the runtime log records   exactly which registry+executor+adapter produced a trajectory.

There is no fallback to the legacy Stage-3 `synthetic_tools.py`. If a tool name is unknown the adapter raises; a silent fallback would produce trajectories that look fine and score against the wrong executor.

## Gold-replay preflight

| dataset | rows | gold replay | reference args resolved | status |
|---|---|---|---|---|
| `train_grpo_pilot2.jsonl` | 160/160 | 160/160 | 440 in 160 rows | PASS |
| `heldout_grpo_pilot2.jsonl` | 80/80 | 80/80 | 216 in 80 rows | PASS |

Verdict: **PASS** (0 problems).

| hash | value |
|---|---|
| `registry_hash` | `f5210beb12d4a33af42de6c8bf1c2b8c3acb3cdd8cef7f0b030e09ee7d2ac562` |
| `executor_hash` | `42d2696bf584cdd1f5c2c95ce0b281dc9de0a1598741256dc7bbe7a531a176e4` |
| `adapter_registry_hash` | `55cd6806a0da4603c69ab8856adee646a56b810910bae036025d5488df8a1141` |
| `adapter_version` | `ttdf-adapter-1.0.0` |
| `generator_version` | `ttdf-0.1.0` |
| `n_tools` | `132` |

The preflight exits non-zero on a single failure. It runs again on RunPod as step 4 of `run_all_4gpu.sh`, before any GPU time is spent, because a replay failure there means the pod's executor differs from the one that produced the oracle.
