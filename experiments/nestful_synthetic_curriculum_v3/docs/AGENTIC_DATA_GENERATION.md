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
spend budget is exhausted; the acceptance rate is `< 2%` after 5 batches; the
contamination gate fires 10 times; the per-stage iteration budget runs out.
Cost report: `reports/OPENROUTER_COST_REPORT.md`.

## Acceptance gates (per example)

Hard gates, in cost order: valid challenger JSON → schema (call count for
stage, registry tools, motif) → no CoT leakage in `rationale` → deterministic
execution (gold replay, non-null answer, no unresolved `$var`, no metadata
leakage in the question, multi-call tasks must actually chain) → in-corpus
dedup (question hash, trace hash) → **NESTFUL overlap = 0** (question hash,
gold-trace hash, sample_id vs dev/test/full) → weak solver fails
(`<= 0.50`) → strong solver passes (`>= 0.80`, gap `>= 0.25`) → LLM judge
style check. Rejects land in `rejected/rejected_examples.jsonl` +
`rejected/rejection_reasons.csv` with one of the reason codes from
`lib/agentic_data/schema.py::REJECTION_REASONS`.

Deterministic solver scores: `1.0` executable win / solution-equivalent,
`0.5–0.8` correct prefix (partial credit by depth), `0.0–0.4` under-call /
wrong tool / wrong args / parse error.

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
   v3.1 on most dimensions, positive solver gap, better probe signal than
   v3.1. *Means only that training is worth trying.*
3. **actually useful** — ONLY if GRPO training on it improves same-batch
   official NESTFUL win rate. A dataset can pass every static check and still
   not transfer — v3.1 itself passed validation and did not reliably improve
   NESTFUL. Do not claim the dataset is good before step 3.

## How to run

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
```

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
