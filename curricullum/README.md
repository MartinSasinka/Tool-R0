# NESTFUL Curriculum Data Generation

Minimal Tool-R0-style pipeline for generating synthetic NESTFUL nested tool-calling curriculum data. Intended as the data-generation stage for later GRPO/QLoRA training (e.g. `Qwen/Qwen3-4B-Instruct-2507`). Training, evaluation, and embedding analysis are **not** included here.

## Purpose

Produce curriculum JSONL files with increasing nested tool-call depth:

| Epoch | Tool calls | Dependency requirement (strict_chain) | Output file |
|-------|------------|----------------------------------------|-------------|
| 1 | exactly 1 | none | `data/filtered/epoch_1_1call.jsonl` |
| 2 | exactly 2 | call 2 → call 1 | `data/filtered/epoch_2_2call.jsonl` |
| 3 | exactly 3 | call 2→1, call 3→2, depth ≥ 3 | `data/filtered/epoch_3_3call.jsonl` |
| 4 | exactly 4 | call 2→1, call 3→2, call 4→3, depth ≥ 4 | `data/filtered/epoch_4_4call.jsonl` |

Combined: `data/filtered/curriculum_all.jsonl`

Each final row has **exactly** these fields (NESTFUL-compatible):

```json
{
  "sample_id": "synthetic-epoch2-000001",
  "input": "...",
  "tools": "[...]",
  "output": "[...]",
  "gold_answer": "..."
}
```

- `tools` and `output` are JSON **strings**
- `gold_answer` is a **string** (numbers are stringified)

## Relation to NESTFUL and Tool-R0

- **NESTFUL**: Real benchmark data at `eval/data/NESTFUL-main/data_v2/nestful_data.jsonl`. Tool schemas are sampled from this corpus; default `--seed_mode schema_only` avoids copying benchmark tasks into prompts.
- **Tool-R0**: Mirrors the step2 pipeline:
  - `step1_gen_candidates.py` ≈ `step2_gen.py`
  - `step2_verify_candidates.py` ≈ `step2_genverify.py` (deterministic rules, no solver LLM)
  - `step3_select_curriculum.py` ≈ `step2_select_curriculum.py` (scoring/dedup, no judge LLM)
  - `run_generate_curriculum.sh` ≈ `run_step2.sh`

## Setup

```bash
export OPENROUTER_API_KEY="your-key"
```

Uses the existing `openai` package with OpenRouter (`https://openrouter.ai/api/v1`, default model `deepseek/deepseek-v4-flash`). The HuggingFace ID `deepseek-ai/DeepSeek-V4-Flash` is the same model but must use the OpenRouter slug when calling this pipeline.

## Run full pipeline

Default profile is **`fast`** (~900 candidates / 400 final per epoch, ~1600 total curriculum).

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File curricullum/run_generate_curriculum.ps1
```

```bash
# Linux / DGX
bash curricullum/run_generate_curriculum.sh
```

Live log (Windows): `curricullum/data/reports/full_run_live.log`  
Status: `curricullum/data/reports/run_status.txt`

### Profiles

| Profile | `CURRICULUM_PROFILE` | Gen / epoch | Final / epoch | Total curriculum |
|---------|----------------------|-------------|---------------|------------------|
| **fast** (default) | `fast` | 900 (max 1400) | 400 | ~1600 |
| full | `full` | 2500 (max 5000) | 1000 | ~4000 |

### Environment overrides

| Variable | fast default | full default | Description |
|----------|--------------|--------------|-------------|
| `CURRICULUM_PROFILE` | `fast` | `full` | Preset bundle |
| `N_GENERATE` | 900 | 2500 | Parsed candidates per epoch |
| `MAX_GENERATE` | 1400 | 5000 | Max API attempts per epoch |
| `N_FINAL` | 400 | 1000 | Selected samples per epoch |
| `PARALLEL_WORKERS` | 12 | 6 | Concurrent OpenRouter calls per batch |
| `PARALLEL_EPOCHS` | 0 | 0 | Set `1` to run all 4 step1 jobs in parallel (~4× faster step1) |
| `SEED_MODE` | `schema_only` | | Anti-leakage default |
| `DEPENDENCY_MODE` | `strict_chain` | | Nested chain validation |
| `TARGET_PROMPT_TOKENS` | 2048 | | Max estimated prompt tokens (GRPO `prompt_length`) |
| `TARGET_MAX_COMPLETION_TOKENS` | 4096 | | Max estimated completion tokens (`max_completion_length`) |
| `MAX_INPUT_CHARS` | 512 | | Hard cap on user `input` text |
| `TOOL_MENU_MIN` / `TOOL_MENU_MAX` | 4 / 6 | | Tools in menu (used + distractors) |

### DGX / GRPO context budget

Generated data is trimmed to fit Tool-R0 GRPO on 3×40 GB GPUs with LoRA (see `Usage_calculator/estimate_context_vram.py`):

| Training param | Default | Curriculum alignment |
|----------------|---------|----------------------|
| `prompt_length` | 2048 | `--target_prompt_tokens 2048` |
| `max_completion_length` | 4096 | `--target_max_completion_tokens 4096` |
| `num_generations` | 4 | (VRAM driver — not data size) |
| `per_device_train_batch_size` | 1 | — |

Step1 rejects over-budget samples (short `input`, compact tool schemas, 4–6 tool menu). Step2 re-checks; step3 prefers samples under budget with room for longer completions.

Example VRAM estimate (LoRA r=32, grad ckpt, batch=1, G=4):

```bash
python Usage_calculator/estimate_context_vram.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --gpu_memory_gb 40 --num_gpus 3 --zero_stage 2 \
  --train_mode lora --lora_r 32 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --num_generations 4 --prompt_length 2048 --max_completion_length 4096 \
  --gradient_checkpointing true
```

With `max_completion_length=4096` you stay near ~29 GB/GPU; raising to ~8192 is possible if you lower `num_generations` or accept tighter prompt/tool menus. Inspect final JSONL context stats:

```bash
python curricullum/data/inspect_dataset.py \
  --path curricullum/data/filtered_toolr0_synthetic/curriculum_toolr0_all.jsonl
```

### Speed notes

- Step1 runs **parallel OpenRouter calls** (`--parallel_workers`, default **12**).
- Larger batch sizes (12/10/8/8) keep workers busy between round-trips.
- OpenRouter **`max_tokens`** per epoch is capped lower than GRPO training budget (1024→3072) — generated JSON is small; this cuts latency without changing DGX `max_completion_length=4096`.
- **`PARALLEL_EPOCHS=1`**: all four step1 jobs at once (~4× faster generation phase; may hit OpenRouter rate limits — reduce `PARALLEL_WORKERS` if you see 429 errors).
- Epochs 1–4 use tiered API `max_tokens`; GRPO context budget stays 2048/4096 via `context_budget.py`.
- **fast** profile: up to ~5600 API calls (4×1400 max), typically **~4–6 hours** sequential or **~2–3 hours** with `PARALLEL_EPOCHS=1`.

Full profile for production-scale data:

```bash
CURRICULUM_PROFILE=full bash curricullum/run_generate_curriculum.sh
```

## Sanity check (no API cost)

```bash
bash curricullum/sanity_check.sh
```

Optional smoke test (requires `OPENROUTER_API_KEY`):

```bash
python curricullum/data/step1_gen_candidates.py \
  --nestful_path eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \
  --out_json curricullum/data/raw_toolr0/_smoke.json \
  --epoch 1 --n_generate 5 --max_generate 10 --batch_size 2 --seed 42
```

## Single-epoch commands

```bash
python curricullum/data/step1_gen_candidates.py \
  --nestful_path eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \
  --out_json curricullum/data/raw_toolr0/epoch_1_candidates.json \
  --epoch 1 --n_generate 2500 --batch_size 10 \
  --model deepseek/deepseek-v4-flash --seed 42 --seed_mode schema_only

python curricullum/data/step2_verify_candidates.py \
  --in_json curricullum/data/raw_toolr0/epoch_1_candidates.json \
  --out_json curricullum/data/verified_toolr0/epoch_1_verified.json \
  --rejected_json curricullum/data/rejected_toolr0/epoch_1_rejected.json \
  --epoch 1 --dependency_mode strict_chain --use_executor

python curricullum/data/step3_select_curriculum.py \
  --in_json curricullum/data/verified_toolr0/epoch_1_verified.json \
  --out_jsonl curricullum/data/filtered_toolr0_synthetic/epoch_1_1call.jsonl \
  --n_final 400 --epoch 1 --seed 42
```

## Inspect output

```bash
python curricullum/data/inspect_dataset.py \
  --path curricullum/data/filtered_toolr0_synthetic/curriculum_toolr0_all.jsonl
```

## Traceability IDs

| ID | Where | Purpose |
|----|-------|---------|
| `raw_id` | `*_raw_responses.jsonl`, candidate `meta` | Links every API response |
| `candidate_id` | candidate / verified / rejected `meta` | Stable hash from raw_id + input |
| `sample_id` | final JSONL | `synthetic-epoch{N}-{idx:06d}` |

Raw responses are **always** written to disk before parse rejection.

## Intermediate artifacts

| Path | Content |
|------|---------|
| `data/raw_toolr0/epoch_N_candidates.json` | Parsed candidates + `meta` |
| `data/raw_toolr0/epoch_N_candidates_raw_responses.jsonl` | All raw model responses |
| `data/verified_toolr0/epoch_N_verified.json` | Passed verification |
| `data/rejected_toolr0/epoch_N_rejected.json` | Rejected + `reason` + `candidate_id` |
| `data/filtered_toolr0_synthetic/epoch_N_Ncall.jsonl` | Final NESTFUL-compatible training JSONL |
| `data/reports/step{1,2,3}_epochN_*` | Human `.txt` + machine `.json` summaries |
| `data/reports/tool_schema_conflicts.json` | Tool name schema variants from real NESTFUL |

## CLI flags (notable)

| Flag | Script | Description |
|------|--------|-------------|
| `--seed_mode schema_only` | step1 | Default; schemas only, no benchmark few-shot tasks |
| `--dependency_mode strict_chain` | step1, step2 | Enforces sequential call→call chain |
| `--allow_new_tools` | step1 | Allow tool names outside NESTFUL catalog |
| `--use_executor` | step1, step2 | **Not implemented** — exits with clear error |
| `--executable_functions_path` | step1, step2 | Future hook for execution validation |

## Limitations

**Current validation checks structure, schema, references, contamination, and nested dependency quality. It does not fully guarantee semantic executability or `gold_answer` correctness.**

- Execution validation (`--use_executor`) is not implemented yet.
- Epoch 4 may yield **fewer than 1000** valid samples — shortfalls are reported loudly; never padded or duplicated.
- Tool schema conflicts in real NESTFUL use the **most frequent** schema as canonical (see `tool_schema_conflicts.json`).
- `--seed_mode fewshot` includes real benchmark examples in prompts — use only for debugging; default `schema_only` is safer for research.
