# NESTFUL MT-GRPO — Partial (graded) reward

A sibling experiment to [`../nestful_mtgrpo_minimal`](../nestful_mtgrpo_minimal).
It is **identical in every way except the training reward**: instead of the
strict binary gold-trace reward (R = 1 only if the *entire* gold trace + gold
answer are reproduced), training uses a **partial / graded** reward that gives
credit for each tool call the model gets right and for reaching the gold answer.

Everything else — data, model loading, ReAct rollout, executor, the MT-GRPO
trainer, the official NESTFUL scorer, W&B logging, the curriculum loop — is
**reused** from the sibling. This folder only adds the new reward plus a thin
driver, so the validated strict pipeline is left untouched.

## Why (research motivation, RQ2)

The strict reward is very sparse: one early mistake zeroes the whole episode,
even if 3 of 4 calls were correct. The partial reward is a **denser** signal.

Crucially, **evaluation is unchanged** (strict `strict_gold_trace_pass` +
official NESTFUL F1 / Partial / Full / **Win Rate**). So the two experiments are
directly comparable and let us answer:

> Does optimizing for *partial* gold-trace reproduction improve execution-based
> task success (Win Rate), or does it just inflate the training reward without
> transferring? — i.e. is trace fidelity the same as functional correctness?

## The reward (`partial_reward.py`)

Per gold position `i`, a graded turn score in `[0, 1]`:

```
turn_score_i = w_name * 1{name_ok}
             + w_keys * 1{name_ok AND keys_ok}
             + w_exec * 1{name_ok AND keys_ok AND exec_ok}
```

`keys`/`exec` credit is **gated on the tool name** being correct (matching
argument keys against the wrong tool is meaningless → 0). With the defaults
(`0.4 / 0.3 / 0.3`) a fully-correct turn scores `1.0`.

Episode reward combines the mean turn score with the final-answer signal:

```
R = w_trace * mean(turn_score) + w_final * 1{final_answer_pass}
    - length_penalty * max(0, num_calls - gold_n) / gold_n
R = clip(R, 0, 1)
```

Defaults (`w_trace=0.7`, `w_final=0.3`) keep a perfect episode at `R = 1.0`, on
the **same [0,1] scale as the strict reward**. Clipped (truncated) rollouts get
`R = 0` and are masked from the update, exactly like the strict reward.

For turn-level MT-GRPO, `episode_turn_reward_seq` maps these graded scores onto
the model's generated turns (same contract as the strict version), so the
trainer's advantage math is unchanged — it simply receives graded floats in
`r_seq` instead of `0/1`.

### Tuning the weights

Edit the `partial_reward:` block in `config.yaml` (or pass `--override
partial_reward.w_final=0.5`). Examples:

- `w_trace=0`, `w_final=1` → reward only the final answer.
- `length_penalty=0.5` → discourage emitting more calls than gold.

## Layout

```
nestful_mtgrpo_partial/
├── partial_reward.py     # the graded reward (the only real new logic)
├── run.py                # thin driver: reuses sibling, swaps train reward
├── config.yaml           # = sibling config, reward.train_policy=partial_gold_trace
├── run_curriculum.sh     # wrapper around the sibling curriculum loop
├── tests/                # correctness tests for the partial reward
└── outputs/              # checkpoints + eval written here (created on first run)
```

How `run.py` stays isolated:

- It imports the heavy modules (`grpo_train`, `reward`, `rollout`, `executor`,
  …) from the sibling via `sys.path`, and loads the sibling `run.py` by explicit
  path (both folders have a `run.py`).
- For `--mode train` it monkeypatches `grpo_train.episode_turn_reward_seq` with
  the partial version **before** calling the sibling trainer. The trainer
  resolves that name at call time, so the swap is clean and local.
- Eval modes (`smoke` / `rollout_eval` / `final_eval`) are delegated to the
  sibling **unchanged** (strict + official metrics).
- INPUT data paths resolve against the sibling (dataset lives there); OUTPUT
  paths resolve against this folder.

## Run

Setup (once, on the GPU box) — reuse the sibling installer:

```bash
bash ../nestful_mtgrpo_minimal/install_deps.sh
```

Quick smoke (no training, sanity-checks generation + reward wiring):

```bash
python run.py --mode smoke --config config.yaml
```

Single-stage training with the partial reward + W&B:

```bash
export WANDB_API_KEY=...            # enables W&B
python run.py --mode train --config config.yaml \
  --override logging.use_wandb=true \
  --override data.train_stage=3 \
  --override data.eval_stage=4
```

Full curriculum (tmux on RunPod recommended) with vLLM + W&B:

```bash
tmux new -s partial
export WANDB_API_KEY=...
CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 PROFILE=curriculum STAGES="1 2 3 4" \
  bash run_curriculum.sh
# detach: Ctrl-b d   (keeps running if you disconnect)   reattach: tmux attach -t partial
```

Evaluate a partial-trained checkpoint (strict + official metrics, comparable to
the strict experiment):

```bash
python run.py --mode final_eval --config config.yaml \
  --checkpoint outputs/curriculum/stage_3/checkpoints/adapter_epoch_3 \
  --override data.eval_paradigm=react
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers: perfect trace == 1.0, partial credit where strict gives 0, name-only
credit, monotonicity in number of correct steps, missing/extra turns, clipped
zeroing, the per-generated-turn `r_seq` mapping, and config-driven weights.

## Notes / caveats

- With partial-reward training, the **train-loop** `mean_reward` logged to W&B is
  the *partial* reward (graded). The per-epoch **curriculum gate** still reads
  the strict `strict_gold_trace_pass` from the (unchanged) `rollout_eval`, so the
  `curriculum.advance_threshold` keeps its original meaning.
- Win Rate requires `signal.SIGALRM` (Unix only); it is computed on Linux
  (RunPod) and skipped on Windows — same behavior as the sibling.
- This experiment requires a **fresh training run**; you cannot retro-fit partial
  credit onto checkpoints trained with the strict reward.
```
