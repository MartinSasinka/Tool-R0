# Agentic (Autodata-style) OpenRouter data generation

Second NESTFUL-like synthetic data path, complementary to the deterministic
`curriculum_v4_nestful_like` generator. Based on **Autodata / Agentic
Self-Instruct** (Kulikov et al., arXiv:2606.25996): an agent acting as a
*data scientist* iteratively creates data, analyzes what failed, and revises
its generation recipe — accepting only examples where a **weak solver fails
and a strong solver passes**.

This is a **research dataset**, not a guaranteed fix. It must survive scoring,
the stage probe, and a same-batch official NESTFUL eval before any claim.

## What "Autodata-inspired" means here

The paper's loop, adapted to executable tool-use:

| Paper component | Our implementation |
|---|---|
| Data scientist / orchestrator | `lib/agentic_data/orchestrator.py` — drives batches, tracks rejections, revises the recipe, enforces budgets |
| Challenger LLM | proposes question + gold-call plan over our synthetic tool registry (`lib/agentic_data/challenger.py`) |
| Weak solver | same cheap model, 1 attempt, low token budget, no scaffolding — models the training target (`lib/agentic_data/solvers.py`) |
| Strong solver | same model with more inference compute: 3 attempts (best kept), planning scaffold, 2x tokens |
| Verifier / judge | **deterministic executor first** (`lib/agentic_data/verifier.py`); LLM judge is secondary and only checks naturalness/ambiguity/NESTFUL style — it can reject but can never override execution results |
| Recipe revision | top rejection reasons per batch map to targeted challenger instructions (`lib/agentic_data/recipe.py`), mirroring the paper's "too easy → different angle" feedback |
| Weak-fail/strong-pass filter | acceptance requires `strong >= 0.80`, `weak <= 0.50`, `gap >= 0.25` (deterministic execution-based scores) |

Paper practices we kept: the strong solver runs **only when the weak solver
failed** (compute saving); multiple strong attempts reduce variance; batch
statistics (mean rounds per accepted item, one-sided failure modes) are
reported; a final quality re-validation pass runs over the whole corpus.

Key deviation from the paper: gold answers are **never** LLM-claimed. Every
gold trace is executed by the deterministic tool registry
(`lib/nestful_like_generator.py` — pure-python tools, written from scratch,
aggregate NESTFUL style only). The executor computes observations and the
final gold answer; non-executable candidates are rejected.

## OpenRouter configuration

`scripts/data/openrouter_client.py` (stdlib only): retry with exponential
backoff + `Retry-After`, JSON-mode with automatic fallback for models that
reject `response_format`, robust JSON extraction/repair, prompt-hash response
cache, request/spend budget guards, redacted raw dumps.

```bash
export OPENROUTER_API_KEY="..."                       # environment ONLY, never printed
export OPENROUTER_CHALLENGER_MODEL="deepseek/deepseek-chat"
export OPENROUTER_WEAK_MODEL="deepseek/deepseek-chat"
export OPENROUTER_STRONG_MODEL="deepseek/deepseek-chat"
export OPENROUTER_JUDGE_MODEL="deepseek/deepseek-chat"
export OPENROUTER_MAX_RETRIES=5
export OPENROUTER_CACHE=1          # cache responses by prompt hash
export OPENROUTER_SAVE_RAW=1       # raw responses under data/.../raw/<role>/
export OPENROUTER_DRY_RUN=0
```

DeepSeek is only a **default**; model slugs drift on OpenRouter, so every role
model is configurable. Weak and strong can be the same model — the strong mode
gets more attempts/tokens/scaffolding (per the paper).

Cost: OpenRouter `usage.cost` when present, else fallback prices
(`OPENROUTER_PRICE_PROMPT_PER_M` / `OPENROUTER_PRICE_COMPLETION_PER_M`).

## Budgets and stop conditions

```bash
OPENROUTER_MAX_REQUESTS=1000        # hard request budget
OPENROUTER_MAX_ACCEPTED_PER_STAGE=800
OPENROUTER_MAX_SPEND_USD=20         # hard spend budget
```

Generation stops early (with partial outputs + reports) when: the request or
spend budget is exhausted; the acceptance rate is below threshold after warmup
(``< 2%`` after 5 batches on fresh runs; ``< 0.5%`` after 50 batches and at least
100 iterations on ``--resume``) — **disable via** ``MIN_ACCEPT_RATE=0``; the
contamination gate fires 10 times; the per-stage iteration budget runs out.
Override via ``WARMUP_BATCHES``, ``MIN_ACCEPT_RATE``, ``RESUME_MIN_ITERATIONS``.
Cost report: `reports/OPENROUTER_COST_REPORT.md`.

## Acceptance gates (per example)

Hard gates, in cost order: valid challenger JSON → schema (call count for
stage, registry tools, motif) → no CoT leakage in `rationale` → deterministic
execution (gold replay, non-null answer, no unresolved `$var`, no metadata
leakage in the question, multi-call tasks must actually chain) → in-corpus
dedup (question hash, trace hash) → **NESTFUL overlap = 0** (question hash,
gold-trace hash, sample_id vs dev/test/full) → weak solver fails
(`<= 0.50`) → strong solver passes (gap `>= 0.25`; see strong-pass policy
below) → **diversity caps** (see below) → LLM judge style check. Rejects land
in `rejected/rejected_examples.jsonl` + `rejected/rejection_reasons.csv` with
one of the reason codes from `lib/agentic_data/schema.py::REJECTION_REASONS`.

Deterministic solver scores: `1.0` executable win / solution-equivalent,
`0.5–0.8` correct prefix (partial credit by depth), `0.0–0.4` failures. Every
non-win gets an explicit **failure type**: `no_tool_call`, `parse_error`,
`wrong_tool`, `wrong_args`, `invalid_reference` (unresolved `$var` reuse),
`execution_error`, `under_call`, `correct_prefix_then_stop` (clean stop after
a correct prefix), `partial_prefix` (diverged mid-trace),
`correct_answer_wrong_trace` (right answer, broken trace), `wrong_answer`.
For a 2-call task, `weak_score = 0.5` means exactly "the weak solver got call
1 right and then failed" — useful, but it must not dominate the dataset.

### Strong-pass policy

`STRONG_PASS_POLICY=exact_win` (default): the strong solver must produce a
**true executable win or solution-equivalent answer (score = 1.0)**. Partial
strong solutions are only kept in `rejected/` logs, never in the filtered
training set. `threshold` restores the legacy `>= 0.80` rule (today
mathematically equivalent, since partial-prefix scores are capped below 0.8).

### Diversity caps on accepted examples

To avoid a homogeneous dataset (e.g. 95% `weak_score=0.5` /
`correct_prefix_then_stop`), the orchestrator enforces per-stage caps on the
**accepted** set, checked before the (paid) judge call:

```bash
DIVERSITY_MAX_SAME_WEAK_SCORE=0.40    # max fraction in one weak-score bucket
DIVERSITY_MAX_SAME_FAILURE_TYPE=0.40  # max fraction with one failure type
DIVERSITY_ENFORCE_AFTER=25            # warmup: caps off below this many accepted
```

A candidate that would push a bucket over its cap is rejected with
`diversity_cap_weak_score` / `diversity_cap_failure_type`; that feedback also
steers the challenger recipe toward other difficulty levels and failure
modes. `weak_score=0.5` examples are still accepted — just not past 40% of
the **enforced** set.

**On `--resume` (Variant B + C combined):**

- Diversity caps apply only to **newly accepted rows this run**, not to the
  legacy seed. A homogeneous partial corpus (e.g. 93% `weak_score=0.50`) no
  longer blocks the common bucket forever; caps still enforce diversity among
  the new 572 rows at 0.40 per bucket/type.
- Acceptance-rate early-stop is more patient: default `WARMUP_BATCHES=50`,
  `MIN_ACCEPT_RATE=0.005`, and no stop before `RESUME_MIN_ITERATIONS=100`
  (override via env). Fresh runs keep `WARMUP_BATCHES=5`, `MIN_ACCEPT_RATE=0.02`.

```bash
# Resume stage2 with diversity on NEW rows (recommended)
export CONFIRM_FULL_AGENTIC_GENERATION=1
export WEAK_SOLVER_MODE=handicapped
export OPENROUTER_MAX_ITERATIONS_PER_STAGE=1200
export OPENROUTER_MAX_REQUESTS=20000
export OPENROUTER_MAX_SPEND_USD=15
# caps at 0.40 — enforced on NEW accepts only when resuming
python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py \
  --resume --stages stage2_2call_agentic_openrouter --seed 43
```

## Contamination prevention

- The challenger is **never shown NESTFUL items** — only the synthetic tool
  registry and recipe feedback. Nothing to copy or paraphrase.
- `tool_schema_source_policy = aggregate_style_only`: tool signatures are
  synthetic, only aggregate NESTFUL statistics (naming style, arity ranges,
  `$varN.output_key$` convention) informed the registry design.
  `exact_tool_signatures_if_explicitly_approved` is defined but NOT used.
- Overlap gate per candidate + defense-in-depth re-check of the final corpus;
  the build **aborts** if NESTFUL reference files are missing (gate would be
  unauditable) or overlap rejections repeat.
- Report: `reports/AGENTIC_CONTAMINATION_REPORT.md`.

## Quality scoring — and why validation alone is not enough

`scripts/data/score_dataset_quality.py` writes `DATASET_QUALITY.md/.json`
with five sections: validity, contamination, distribution similarity (vs
v3.1 and NESTFUL; total-variation distance per dimension), solver gap, GRPO
signal (read from a stage-probe report; the scorer never launches the probe).

Verdict ladder (never skip a rung):

1. **technically acceptable** — validity + contamination hard gates pass,
   gold replay = 1.0. *Means only that nothing is broken.*
2. **training candidate** — also distributionally closer to NESTFUL than
   v3.1 on most dimensions, positive solver gap,
   `strong_exact_win_rate >= 0.95`, `weak_score_bucket_dominance <= 0.40`,
   failure-type diversity (>= 4 distinct weak failure types per stage, no
   type above 0.40), better probe signal than v3.1. The scorer reports
   `weak_score_entropy`, `failure_type_entropy`,
   `accepted_failure_type_diversity`, `weak_score_bucket_dominance` and
   `strong_exact_win_rate` in the solver-gap section. *Means only that
   training is worth trying.*
3. **actually useful** — ONLY if GRPO training on it improves same-batch
   official NESTFUL win rate. A dataset can pass every static check and still
   not transfer — v3.1 itself passed validation and did not reliably improve
   NESTFUL. Do not claim the dataset is good before step 3.

## Model configuration (hybrid — recommended)

| Role | Backend | Default model |
|------|---------|---------------|
| Challenger | OpenRouter | `deepseek/deepseek-v3.2` |
| **Weak solver** | **Local HF** | `Qwen/Qwen3-4B-Instruct-2507` |
| Strong solver | OpenRouter | `qwen/qwen3-235b-a22b-2507` |
| Verifier | Deterministic executor | (no LLM) |
| LLM judge | OpenRouter | `deepseek/deepseek-v3.2` |

The weak solver should be the **exact training target** (local Qwen3-4B), not an API
proxy. Set:

```bash
export WEAK_SOLVER_BACKEND=local
export LOCAL_WEAK_MODEL=Qwen/Qwen3-4B-Instruct-2507
export LOCAL_WEAK_4BIT=1          # fits 6 GB GPUs; set 0 on A100+

export OPENROUTER_CHALLENGER_MODEL=deepseek/deepseek-v3.2
export OPENROUTER_STRONG_MODEL=qwen/qwen3-235b-a22b-2507
export OPENROUTER_JUDGE_MODEL=deepseek/deepseek-v3.2
```

### GRPO-signal rollout gate (multi-turn)

The accept/reject gate samples `ROLLOUT_N` (default 8) **multi-turn** episodes via
`run_episode(mode="train")` — the same path as `probe_stage.py` and MT-GRPO
training — and scores them with `execution_aware_v3_2_dense` (or
`AGENTIC_REWARD_POLICY`). It does **not** use the legacy single-shot JSON weak-solver
prompt unless you explicitly set `AGENTIC_ROLLOUT_MODE=single_shot` (debug only).

```bash
export WEAK_SOLVER_BACKEND=local
export ROLLOUT_N=8
export ROLLOUT_TEMPERATURE=0.8
# optional per-turn cap (0 = training config stage_defaults)
export ROLLOUT_MAX_TOKENS=0
```

### Solver-gap (multi-turn, default when local weak)

Weak and strong solvers use the same ``run_episode(mode="train")`` path as the
rollout gate when ``WEAK_SOLVER_BACKEND=local`` (override with
``AGENTIC_SOLVER_GAP_MODE=single_shot`` for legacy JSON probing only).

```bash
export AGENTIC_SOLVER_GAP_MODE=multiturn   # default when local
export SOLVER_MT_WEAK_TEMPERATURE=0.2
export SOLVER_MT_STRONG_TEMPERATURE=0.7
```

### Executor mode for synthetic tools (CRITICAL — read before probing/training)

Agentic tool names (e.g. `units_per_box`, `percentage_of`) come from
`nestful_like_generator.TOOLS` — they are **not** entries in the real NESTFUL
IBM function registry. That real registry IS present in this repo (used for
genuine NESTFUL data), so any code path that resolves
`executor.mode="auto"` against it will pick `full` execution and try to run
predicted tool calls against the **real** IBM functions. On this corpus that
either:

1. hard-fails almost every episode on the first call with
   `exec:unknown_function:<name>` (most synthetic names aren't real IBM
   functions), or
2. **silently executes a different, real IBM function** when a synthetic
   name happens to collide with one (confirmed in practice: the synthetic
   `rectangle_area` tool coincidentally matches a real IBM function and
   returns a plausible-looking but uncontrolled result).

Either way the reward is corrupted, independent of how good the model's
completion is. This is fixed inside the generation gate itself
(`rollout_signal.load_rollout_config` / `multiturn_solver._context` always
force `executor.mode=gold_replay`), and `probe_stage.py` / `run_grpo.sh` now
**auto-detect** agentic dataset paths (or a `source` field containing
`"agentic"`) and force `executor.mode=gold_replay` too — but if you write a
NEW script against `nestful_mtgrpo_minimal/run.py` / `vllm_dp_pool.py`
directly, you MUST pass this override yourself:

```bash
# probe_stage.py — auto-forced when the dataset path/`source` looks agentic;
# override explicitly if you ever need to force a different mode:
python .../probe_stage.py --dataset .../stage2_2call_agentic_openrouter.jsonl \
  --override executor.mode=gold_replay ...

# run_grpo.sh — auto-forced via EXTRA_TRAIN_OVERRIDES_STR when any
# STAGE<N>_FILE_OVERRIDE path looks agentic; verify in the printed log line
# "[grpo] AGENTIC dataset detected — forcing executor.mode=gold_replay".
```

A live repro before this fix: probing `stage2_2call_agentic_openrouter.jsonl`
with the default config gave `dead_group_rate=0.33-0.88` even with a
STUB backend emitting the exact gold trace — the episode was failing on
`unknown_function`, not on genuine task difficulty. After forcing
`gold_replay`, the same stub probe gives `dead_group_rate=0.0`. Any prior
`PROBE_REPORT.json` for this dataset that does not show
`"executor_mode": "gold_replay"` should be treated as unreliable and
re-run.

Quick start (repo root):

```bash
export OPENROUTER_API_KEY="..."
export CONFIRM_FULL_AGENTIC_GENERATION=1
bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_hybrid.sh
```

Pilot: `PILOT=1 bash .../build_v4_agentic_hybrid.sh`

Local weak calls do **not** count toward `OPENROUTER_MAX_REQUESTS` / spend budget.

## Multi-GPU parallel generation (RunPod, N GPUs)

The generator is not multi-worker-safe on a **single** `--output-dir`
(`filtered/*.jsonl` is opened in write mode, and `sample_id` restarts at 1
inside every process). To parallelize across GPUs, run the SAME code N times
— one process per GPU, one `--output-dir` per worker, one `--seed` per
worker — then merge afterwards:

```bash
# repo root, inside the activated agentic venv, OPENROUTER_API_KEY exported
NUM_GPUS=4 TOTAL_PER_STAGE=60 BASE_SEED=45 \
  TOTAL_SPEND_USD=20 TOTAL_REQUESTS=4000 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/data/launch_multi_gpu_workers.sh
```

This launches one **tmux session per GPU** (`agentic_gpu0`..`agentic_gpuN-1`),
each running `build_curriculum_v4_agentic_openrouter.py` with:

- `CUDA_VISIBLE_DEVICES=$i`, `LOCAL_WEAK_DEVICE=cuda:0` (remapped device 0
  inside the process) — weak-solver inference pinned to its own GPU;
- `--seed BASE_SEED+i` — different challenger batches per worker;
- `--output-dir .../agentic_workers/gpu$i` — no file conflicts;
- `OPENROUTER_MAX_SPEND_USD` / `OPENROUTER_MAX_REQUESTS` split evenly across
  workers (shared OpenRouter account/rate-limit, so this is NOT a 4x budget);
- `--max-accepted-per-stage` = `ceil(TOTAL_PER_STAGE / NUM_GPUS)`.

Each worker independently runs the FULL pipeline (hard-trace validation,
semantic compatibility, GRPO-signal rollout probe, in-worker diversity caps,
LLM judge) — nothing about acceptance gating changes. Only in-corpus dedup
and diversity caps are **per-worker**, not global, until merge.

Monitor with `tmux attach -t agentic_gpu0` (detach: `Ctrl+b d`),
`tail -f .../agentic_workers/gpu*.log`, or `grep -l DONE .../gpu*.log` to
check completion.

### Merge step (required after all workers finish)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/data/merge_agentic_workers.py \
  --workers-glob "experiments/nestful_synthetic_curriculum_v3/data/agentic_workers/gpu*" \
  --output-dir experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter
```

`merge_agentic_workers.py` does NOT re-run any gate — every merged row was
already accepted by its worker's full orchestrator run. It only:

1. re-applies **cross-worker** dedup by `(question_hash, trace_hash)` (same
   hashes as the in-process `DedupIndex`), first worker wins on a duplicate;
2. renumbers `sample_id` sequentially per stage
   (`agentic_v4_<stage>_NNNNNN`) — worker-local ids collide by construction
   (every worker starts counting from 1);
3. writes merged `filtered/<stage>.jsonl` files to the canonical output dir;
4. writes `reports/MERGE_REPORT.md` / `.json`: per-worker loaded/kept/dropped
   counts, sample of dropped duplicates, and merged `corpus_stats`
   (dominance per motif/answer-type/tool-family/question-template) so global
   diversity caps can be checked post-merge.

Then re-run `score_dataset_quality.py` and the stage probe against the
**merged** canonical file exactly as in the single-GPU flow.

Regression tests: `tests/test_merge_agentic_workers.py`.

## Model configuration (OpenRouter-only legacy)

```bash
export OPENROUTER_CHALLENGER_MODEL=deepseek/deepseek-chat
export OPENROUTER_WEAK_MODEL=deepseek/deepseek-chat
export OPENROUTER_STRONG_MODEL=deepseek/deepseek-chat
export OPENROUTER_JUDGE_MODEL=deepseek/deepseek-chat
```

Set `WEAK_SOLVER_BACKEND=openrouter` (default) to use API for all roles.

```bash
cd <repo root>
export OPENROUTER_API_KEY="..."

# 0) offline smoke (no API, no cost; writes to ..._mock/, never real data)
MOCK=1 bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh

# 1) dry-run (print plan, nothing sent)
DRY_RUN=1 bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh

# 2) tiny pilot (default: 10 accepted/stage, <=200 requests, <=$5)
MAX_ACCEPTED_PER_STAGE=10 OPENROUTER_MAX_REQUESTS=200 OPENROUTER_MAX_SPEND_USD=5 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh

# 3) score (also runs automatically after the pilot)
bash experiments/nestful_synthetic_curriculum_v3/scripts/data/score_v4_agentic_dataset.sh

# 4) stage probe on the pod (GPU; NEVER launched automatically)
DATASET=experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter/filtered/stage2_2call_agentic_openrouter.jsonl \
  REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 SEED=42 BACKEND=vllm \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh
# compare with v3.1 stage2 and with REWARD_POLICY=execution_aware_v3_2_dense

# 5) FULL generation (mirrors deterministic v4 counts; real cost, hours)
CONFIRM_FULL_AGENTIC_GENERATION=1 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_full.sh

# 6) SALVAGE a stopped/crashed run (OFFLINE cache replay — zero API cost;
#    writes filtered/*.partial_salvaged.jsonl, archives the old manifest)
python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py --salvage

# 7) RESUME generation (continue toward target; loads existing rows first)
#    Reads filtered/<stage>.jsonl, or *.partial_salvaged.jsonl if canonical
#    is empty. Generates only the remaining gap (228 existing + target 800
#    -> 572 new). Dedup is seeded; sample_id continues from 229.
#    Raise iteration/request budgets for the remaining gap (~1000 iters).
CONFIRM_FULL_AGENTIC_GENERATION=1 \
OPENROUTER_MAX_ITERATIONS_PER_STAGE=1200 \
OPENROUTER_MAX_REQUESTS=20000 \
OPENROUTER_MAX_SPEND_USD=15 \
WEAK_SOLVER_MODE=handicapped \
python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py \
  --resume --stages stage2_2call_agentic_openrouter --seed 43
```

## Crash safety, partial datasets, count consistency

- Every accepted row is appended to `filtered/<stage>.jsonl` **immediately**
  (flush + fsync); early stops and crashes never lose accepted rows.
- A run that stops below target produces a **partial** dataset: still valid,
  still scoreable (`score_dataset_quality.py` reports per-stage
  rows/target/status), but `training_candidate` stays false until targets are
  met.
- The builder verifies accepted counts across memory, filtered files, the
  manifest and the solver-gap log after every run and exits with code 7 on
  any disagreement.
- Target resolution happens in ONE place; the printed FINAL table is written
  verbatim to the manifest (`extra.target_resolution`), including the
  explicit stage4 decision (det. v4 mirror 1600 → 800 used).
- `OPENROUTER_OFFLINE=1` (or `--salvage`) makes the client cache-only: any
  cache miss raises instead of spending money.
- `--resume` loads existing filtered rows (canonical file, or
  `*.partial_salvaged.jsonl` as fallback), seeds dedup, and generates only
  `target - len(existing)` new examples. Output always goes to the canonical
  `filtered/<stage>.jsonl` (partial_salvaged is migrated on first resume).
- Solver difficulty is configurable: `WEAK_SOLVER_MODE=minimal|handicapped`
  (handicapped = 400 tokens, no continuation-pressure hint),
  `STRONG_SOLVER_MODE=scaffolded|plain`. Strong pass requires a true
  executable win (partial-prefix scores are capped below the 0.80 threshold).

## Manual review before any training

1. Read 20+ accepted questions per stage — natural? unambiguous? no leakage?
2. `reports/AGENTIC_SOLVER_GAP_REPORT.md` — weak-fail/strong-pass must
   dominate; check the weak-failure statuses are the modes we want to train
   away (under-call, wrong args), not parse artifacts.
3. `rejected/rejection_reasons.csv` — a huge `too_easy_*` share means the
   recipe needs revision, not more budget.
4. `DATASET_QUALITY.md` — verdict must be at least `technically_acceptable`;
   distribution distance must beat v3.1.
5. Stage probe vs v3.1 (dead_group_rate lower, unique rewards/group higher).
   If the probe is bad: do **not** train; revise the recipe and regenerate.
6. `OPENROUTER_COST_REPORT.md` — sanity-check spend before scaling up.
