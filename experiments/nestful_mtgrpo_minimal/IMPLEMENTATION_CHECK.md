# Implementation Check — nestful_mtgrpo_minimal

This document is the evidence pack for the implementation report. Every claim
links to a concrete file + line range so it can be verified by inspection or
test.

---

## A. Standalone check — no forbidden imports

**Check command (PowerShell / bash):**

```powershell
Select-String -Pattern "^\s*(import|from)\s+(curricullum|nestful_evaluation)" `
    -Path experiments/nestful_mtgrpo_minimal/*.py -Recurse
```

**Result:** 0 matches.

Each `.py` file carries an explicit module-level docstring confirming this:

| File | Docstring statement |
|---|---|
| `reward.py` line 9–10 | "This file is a minimal standalone reimplementation; it imports nothing from curricullum/ or nestful_evaluation/." |
| `grpo_train.py` line 16 | "This file imports nothing from curricullum/ or nestful_evaluation/." |
| `prompt.py` line 3–5 | "This file is a minimal standalone reimplementation inspired by the original project prompt (nestful_evaluation/run.py). The system prompt text is vendored locally so the folder has no external dependency." |
| `rollout.py` line 3–4 | "This file is a minimal standalone reimplementation inspired by the original project rollout (curricullum/train/evaluate_nestful_stage.py::rollout_task). The model only ever sees its OWN previous turns + real executor observations." |
| `executor.py` | Implements IBMFunctionRegistry locally; no import from nestful_evaluation. |

---

## B. MT-GRPO train implementation — concrete function map

### `r_t` — per-turn strict gold reward

| Concept | File + function | Key lines |
|---|---|---|
| Per-turn `r_t` values computed | `reward.py::strict_gold_trace_reward` | lines 110–145: loop over gold turns, `turn_pass` + `turn_rewards = [1.0 if p else 0.0 for p in turn_pass]` |
| `r_t` exposed as list | `reward.py::strict_gold_turn_rewards` | lines 169–186: calls `strict_gold_trace_reward`, returns `rr.diagnostics["turn_rewards"]` |
| `r_t` aligned to generated turns | `reward.py::episode_turn_reward_seq` | lines 189–217: maps r_seq to each generated turn (0 for parse-fail/clipped/terminal) |

### `R_episode` — strict gold-trace episode reward

| Concept | File + function | Key lines |
|---|---|---|
| Binary episode reward 1 iff full gold trace + final answer | `reward.py::strict_gold_trace_reward` | lines 152–162: `reward = 1.0 if (trace_ok and final_answer_pass) else 0.0` |
| Alias confirming identity | `reward.py` line 165–166 | `strict_gold_trace_episode_reward = strict_gold_trace_reward` |
| Used as R_episode in trainer | `grpo_train.py::train` line 220 | `ep.reward = rinfo["episode_reward"]` (from `episode_turn_reward_seq`) |

### `G_t` return formula

```
G_t = sum_{k=t}^{T} gamma^(k-t) * r_k  +  lambda_episode * gamma^(T-t+1) * R_episode
```

| Concept | File + function | Key lines |
|---|---|---|
| Formula implementation | `grpo_train.py::_turn_returns` | lines 340–357: explicit loop + episode term |
| Called per episode | `grpo_train.py::train` | line 231: `ep_returns.append(_turn_returns(r_seq, ep.reward, gamma, lambda_episode))` |

### Group-relative advantage normalization

| Concept | File + function | Key lines |
|---|---|---|
| Flat pool of all (episode, turn) returns | `grpo_train.py::train` | lines 234–242: `flat = [g for ep, gs in ... if not clipped]` |
| Mean + std over flat pool | `grpo_train.py::train` | lines 237–242: `gmean`, `gstd` |
| Normalize per turn | `grpo_train.py::train` | line 281: `adv = (gs[j] - gmean) / (gstd + 1e-8)` |

### Prompt-token masking (loss only on assistant tokens)

| Concept | File + function | Key lines |
|---|---|---|
| `prompt_ids` and `completion_ids` stored separately | `grpo_train.py::_rollout_episode_for_train` | lines 97–102: `p_ids, c_ids` captured, stored as `TurnTokens(p_ids, c_ids)` |
| `_sequence_logprob` only touches completion tokens | `grpo_train.py::_sequence_logprob` | lines 142–160: `start = p.numel()`, `target = input_ids[0, start:]` — prompt tokens never appear in `target` |
| Loss call uses `completion_ids` only | `grpo_train.py::train` | lines 287–291: `cur, n = _sequence_logprob(model, tt.prompt_ids, tt.completion_ids, ...)` |

### Fallback episode-level

| Concept | File + function | Key lines |
|---|---|---|
| `allow_fallback` from config | `grpo_train.py::train` | line 184: `allow_fallback = bool(mt.get("fallback_episode_level_if_needed", True))` |
| Fallback mode (single episode-level advantage) | `grpo_train.py::train` | lines 282–284: `else: adv = (ep.reward - mean_r)` (when `use_turn_level=False`) |
| `fallback_used = True` written to summary | `grpo_train.py::train` | lines 303–304: `if contributing == 0 and allow_fallback: summary["fallback_used"] = True` |

---

## C. Fallback behavior — precise description

### When does fallback occur?

Fallback to episode-level is triggered when **all turn slots contributed 0
gradient steps** in a task group — i.e. `contributing == 0` after iterating
all episodes and turns. This happens when:
- All episodes in a group are clipped (`mask_clipped_from_update=true`) and
  every turn produces no usable `completion_ids`, **or**
- `mt_grpo.mode` is explicitly set to `episode_level` in config.

The `use_turn_level=False` branch (`adv = ep.reward - mean_r`) is the fallback
logic; it assigns a single episode-level advantage to every turn instead of the
per-turn return `G_t`.

### Where is `fallback_used` written?

`grpo_train.py::train` lines 303–304:

```python
if contributing == 0 and allow_fallback:
    summary["fallback_used"] = True
```

`summary` is returned from `train()` and written to
`outputs/train_summary.json` in `run.py::_run_train` (see `run.py` line ~310).

### How to confirm turn_level_minimal ran?

From `train_summary.json`:

```json
{
  "mt_grpo_mode": "turn_level_minimal",
  "fallback_used": false
}
```

**Both conditions must hold.** If `fallback_used=true`, the run was
episode-level for at least one task group and **must NOT be presented as
turn-level MT-GRPO** in any report or paper.

See `examples/example_train_summary_turn_level.json` for the expected shape
and `examples/example_train_summary_fallback.json` for the fallback shape.

---

## D. Reportability flags — where they are written

### In `final_eval` (via `metrics.py::aggregate_final_eval`)

`metrics.py` lines (the `aggregate_final_eval` return block):

```python
reportable = executor_mode == "full"
report = {
    "executor_mode": executor_mode,
    "solution_equivalent_reportable": reportable,
    "win_rate_reportable": reportable,
    ...
}
if not reportable:
    report["warning"] = "Alternative-path metrics are limited because ..."
```

### In `rollout_eval` (via `run.py::_run_rollout_eval`)

`run.py` lines ~244–257:

```python
exec_mode = rows[0]["executor_mode"] if rows else "n/a"
reportable = exec_mode == "full"
metrics["executor_mode"] = exec_mode
metrics["solution_equivalent_reportable"] = reportable
metrics["win_rate_reportable"] = reportable
if not reportable:
    metrics["warning"] = "Alternative-path metrics are limited ..."
metrics["clipped_completion_rate"] = metrics.pop("clipped", 0.0)
```

Both paths write all four fields: `executor_mode`, `solution_equivalent_reportable`,
`win_rate_reportable`, `warning` (gold_replay only).

See `examples/example_metrics_gold_replay.json` for the gold_replay shape.

---

## E. Token diagnostics — where they are stored

### Per-turn (in `rollout.py`)

`rollout.py::Turn` dataclass (lines 18–28):

```python
prompt_tokens: int = 0
completion_tokens: int = 0
clipped_completion: bool = False
```

Populated in `run_episode` via `generate_once`, which returns `(text, prompt_tokens, completion_tokens, clipped)`.

### Per-trajectory

`rollout.py::Trajectory` dataclass stores `clipped_any` (set to `True` the
moment any turn clips).

### In training

`grpo_train.py::_rollout_episode_for_train` lines 100–102:

```python
turn = Turn(turn_idx, text, prompt_tokens=p_len,
            completion_tokens=c_len, clipped_completion=clipped)
```

### Clipped masking from update

`grpo_train.py::train` lines 174 + 275:

```python
mask_clipped = bool(tr.get("mask_clipped_from_update", True))
...
if mask_clipped and ep.trajectory.clipped_any:
    continue
```

Config key: `training.mask_clipped_from_update: true` (default).

### `clipped_completion_rate` in metrics.json

`run.py::_run_rollout_eval` line ~255:

```python
metrics["clipped_completion_rate"] = metrics.pop("clipped", 0.0)
```

---

## F. LoRA / QLoRA — config-driven decision

**Config key:** `finetuning.method: qlora | lora`

**Decision logic** in `run.py::load_model_and_tokenizer` lines 102–150:

```python
method = ft.get("method", "qlora")                        # default: qlora
use_4bit = bool(ft.get("load_in_4bit", ...)) and method == "qlora"
```

| method | Effect |
|---|---|
| `qlora` | `BitsAndBytesConfig(load_in_4bit=True)` passed to `AutoModelForCausalLM.from_pretrained`; `prepare_model_for_kbit_training` called |
| `lora` | Model loaded in bf16/fp16; `gradient_checkpointing_enable()` called; no 4-bit quantization |

**Only PEFT adapter is saved** — `run.py` line ~330:

```python
model.save_pretrained(adapter_dir)   # PEFT model → saves adapter_config.json + adapter weights only
tokenizer.save_pretrained(adapter_dir)
```

`get_peft_model` wraps the base model; `save_pretrained` on a `PeftModel`
saves only the LoRA weights, never the full base model. Full fine-tuning is
impossible because only `requires_grad=True` params (adapter layers) are in
the optimizer:

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=lr)
```

---

## Test coverage summary

| Claim | Test file + test name |
|---|---|
| Fixture loads 3 tasks with correct fields | `test_fixtures_and_flags.py::test_fixture_*` |
| Parser gate (no tag / multiple / invalid JSON / missing name / bad args) | `test_parser.py`, `test_fixtures_and_flags.py::test_parser_gate_*` |
| `strict_gold_turn_reward` r_t values | `test_turn_level.py::test_turn_rewards_*` |
| `strict_gold_trace_reward` = `R_episode` alias | `test_fixtures_and_flags.py::test_episode_reward_is_alias` |
| `strict_gold_trace_reward` correct / wrong name / too few calls | `test_reward.py`, `test_fixtures_and_flags.py::test_strict_reward_*` |
| `G_t` formula (2-turn, discount, single-turn) | `test_turn_level.py::test_turn_returns_*`, `test_fixtures_and_flags.py::test_turn_returns_formula_single_turn` |
| `fallback_used=false` in example_train_summary_turn_level.json | `test_fixtures_and_flags.py::test_example_turn_level_summary_has_correct_keys` |
| `fallback_used=true` in example_train_summary_fallback.json | `test_fixtures_and_flags.py::test_example_fallback_summary_has_correct_keys` |
| Reportability flags for gold_replay (example) | `test_fixtures_and_flags.py::test_example_gold_replay_reportability` |
| `solution_equivalent` not in reward.py return statements | `test_fixtures_and_flags.py::test_solution_equivalent_is_not_training_reward` |
| Token budget all 6 stages + smoke override | `test_turn_level.py::test_stage_token_budget_*`, `test_fixtures_and_flags.py::test_stage_budget_*` |
| `solution_equivalent_score` + NESTFUL official metrics | `test_solution_equivalent.py` |

**Total: 51 tests, 51 passing** (as of last run).
