# Agentic OpenRouter implementation report

Date: 2026-07-09. Task: Autodata-style OpenRouter LLM synthetic curriculum for
NESTFUL-like data (Agentic Self-Instruct, arXiv:2606.25996).

Status: **implementation complete and smoke-tested offline (mock backend +
unit tests).** The real OpenRouter pilot was NOT run from this machine —
`OPENROUTER_API_KEY` is not set here; run it on the pod with the commands in
§Commands. No training was launched. No NESTFUL eval was run. No files were
deleted. No NESTFUL content was copied.

## 1. Files added

| file | purpose |
|---|---|
| `scripts/data/openrouter_client.py` | stdlib OpenRouter chat client: retry+backoff, JSON-mode w/ fallback, JSON extraction/repair, prompt-hash cache, request/spend budget guard, redacted raw dumps, per-role cost stats, dry-run + mock backends |
| `lib/agentic_data/__init__.py` | package doc |
| `lib/agentic_data/schema.py` | stage defs, rejection-reason vocabulary, candidate schema gates, leakage/CoT checks, final accepted-row assembly (spec §4 schema) |
| `lib/agentic_data/recipe.py` | versioned challenger recipe; batch rejection analysis → targeted prompt learnings (paper's challenger-prompt update) |
| `lib/agentic_data/challenger.py` | challenger prompt over the synthetic tool registry; candidate parsing/normalization |
| `lib/agentic_data/solvers.py` | weak (1 attempt, low budget) / strong (3 attempts, planning scaffold) prompts + deterministic execution-based scoring (1.0 win / 0.5–0.8 prefix / 0.0–0.4 failures) |
| `lib/agentic_data/verifier.py` | deterministic executor verification (source of truth for gold); secondary LLM style judge that can reject but never override execution |
| `lib/agentic_data/quality.py` | solver-gap acceptance policy (strong≥0.80, weak≤0.50, gap≥0.25) + in-corpus dedup |
| `lib/agentic_data/contamination.py` | NESTFUL overlap gate (question hash, trace hash, sample_id over dev/test/full); aborts if NESTFUL data missing |
| `lib/agentic_data/distribution.py` | corpus stats + total-variation distance to NESTFUL (same definitions as det. v4 audit) |
| `lib/agentic_data/orchestrator.py` | the Autodata loop: batches → gates in cost order → weak → strong (skipped when weak passes) → judge → accept; recipe revision; stop conditions; output writers |
| `lib/agentic_data/mock_llm.py` | offline mock backend (challenger/solvers/judge) for zero-cost smoke tests; injects flaws to exercise rejection paths |
| `scripts/data/build_curriculum_v4_agentic_openrouter.py` | CLI: target resolution (mirrors det. v4 counts), pilot/full/mock/dry-run, budget guards, final defense-in-depth validation, all reports + manifest |
| `scripts/data/score_dataset_quality.py` | 5-section quality report (validity / contamination / distribution / solver gap / GRPO signal) + verdict ladder |
| `scripts/data/build_v4_agentic_openrouter_pilot.sh` | pilot wrapper (10/stage, ≤200 req, ≤$5 defaults; DRY_RUN, MOCK) |
| `scripts/data/build_v4_agentic_openrouter_full.sh` | full wrapper — hard-requires `CONFIRM_FULL_AGENTIC_GENERATION=1` |
| `scripts/data/score_v4_agentic_dataset.sh` | scoring wrapper (read-only) |
| `tests/test_agentic_data.py` | 22 offline unit tests (JSON repair, verifier gates, solver scoring bands, gap policy) |
| `docs/AGENTIC_DATA_GENERATION.md` | full pipeline documentation |
| `data/curriculum_v4_nestful_like_agentic_openrouter/{raw,filtered,rejected,manifests,reports}/` | dataset directory skeleton (raw/ gitignored) |

## 2. Files changed

- `docs/DATASETS.md` — new "curriculum v4 deterministic + agentic" section.
- `docs/RUNBOOK.md` — new §5 agentic generation quick reference.
- `RESEARCH_FIX_PLAN.md` — new experiment E5b (hypothesis / mechanism /
  metrics / success criteria / failure interpretation).
- `.gitignore` — ignore agentic `raw/` (LLM dumps + cache) and `*_mock/`.

Intentionally untouched: MT-GRPO trainer, `lib/reward_v3_1.py`, eval runner,
deterministic v4 generator (reused read-only as executable tool registry).

## 3. OpenRouter client behavior

- `OPENROUTER_API_KEY` read from env per request, never stored on the object,
  never written to disk/logs; raw dumps contain only model/messages/response/
  token usage. Wrappers print `api_key = set (redacted)` at most.
- Retry: exponential backoff + jitter, honors `Retry-After`; 401/402/403 stop
  immediately; 400 triggers one retry without `response_format` (JSON-mode
  fallback for models that reject it).
- Cache: SHA256(model+messages+params) → `raw/cache/`; re-runs cost $0
  (verified: 68/68 cache hits on mock re-run).
- Cost: OpenRouter `usage.cost` when present, else configurable fallback
  prices; per-role accounting in `reports/OPENROUTER_COST_REPORT.md`.
- Budget guard raises BEFORE the request when `OPENROUTER_MAX_REQUESTS` or
  `OPENROUTER_MAX_SPEND_USD` would be exceeded.

## 4. Model configuration

All four roles configurable via `OPENROUTER_{CHALLENGER,WEAK,STRONG,JUDGE}_MODEL`
(default `deepseek/deepseek-chat`; slugs drift, so nothing is hardcoded).
Weak vs strong is a *mode* difference (paper §2.1): weak = 1 attempt, 700
tokens, temp 0.2, no scaffold; strong = best-of-3 attempts, 1400 tokens,
temp 0.7, planning scaffold.

## 5. Safety / budget controls

- pilot defaults: 10 accepted/stage, ≤200 requests, ≤$5;
- full generation refused without `CONFIRM_FULL_AGENTIC_GENERATION=1`
  (both in the wrapper, exit 3, and in the Python CLI for targets > 50);
- generation stops early on: request budget, spend budget, acceptance rate
  < 2 % after 5 batches, 10 contamination strikes, per-stage iteration cap —
  partial outputs and all reports are still written;
- contamination gate constructed BEFORE any API spend; build aborts if
  NESTFUL reference data is missing;
- builder never launches training or NESTFUL eval.

## 6. Offline smoke results (mock backend — NOT a real dataset)

30-row pilot (10×3 stages), `data/..._agentic_openrouter_mock/`:

- accepted 10/10 per stage; 20 rejections exercised 4 reason codes
  (weak_solver_passed 7, non_executable_gold_trace 5, invalid_schema 5,
  duplicate_question 3); 99 requests, cache re-run = 0 new requests;
- content-deterministic across re-runs at fixed seed (excluding `created_at`);
- unit tests: 22/22 pass; `compileall` clean.

Scoring of the mock pilot (mechanics check only):

- validity hard gates PASS (gold replay 1.0, schema 1.0, no nulls/dups);
- contamination hard gates PASS (0 overlap by question/trace/sample_id);
- distribution: closer to NESTFUL than v3.1 on **4/5** dimensions;
- solver gap: weak_fail_strong_pass rate 1.0, avg gap 0.5 (mock solvers);
- verdict: `technically_acceptable=True`, `training_candidate=False`
  (correct — no stage probe has been run yet), `actually_useful=None`.

Real-pilot acceptance/rejection statistics, dataset quality score and cost
must be filled in after running the pilot on the pod with a real key.

## 7. Contamination result

By construction + gate: the challenger never sees NESTFUL items (only the
from-scratch synthetic tool registry and recipe feedback);
`tool_schema_source_policy = aggregate_style_only`; per-candidate overlap
gate + final-corpus re-check → overlap 0 in all smoke runs. Report:
`reports/AGENTIC_CONTAMINATION_REPORT.md` (written per build).

## 8. Distribution comparison / stage probe

`reports/AGENTIC_DISTRIBUTION_REPORT.md` compares agentic v4 vs v3.1 vs
deterministic v4 vs NESTFUL per build. Stage probe: NOT run (GPU pod
required; never launched automatically) — `reports/AGENTIC_PROBE_REPORT.md`
is written as a stub with the exact probe commands and success targets
(dead_group_rate < v3.1, unique rewards/group > v3.1).

## 9. Remaining risks

1. **Real-LLM candidate validity is unknown** — DeepSeek may produce many
   non-executable traces; the recipe feedback should improve this, but the
   acceptance-rate stop (< 2 %) may fire. Budget is capped either way.
2. **Solver-gap calibration**: if weak DeepSeek is too strong for 2-call
   tasks, stage2 acceptance may be slow (mostly `weak_solver_passed`/
   `too_easy`). Mitigations: weak mode is already minimal-budget; recipe
   revision pushes difficulty up; consider a smaller weak model slug.
3. **The weak solver is not the actual training policy** (Qwen3-4B). The gap
   filter is a proxy; the stage probe with the real policy is the binding
   signal check before training.
4. Judge cost is ~1 call per surviving candidate; use `--no-judge` if budget
   is tight (execution gates still hold).
5. Mock smoke validates mechanics, not data quality — never ship `_mock/`.

## 10. Commands

```bash
cd /workspace/Tool-R0
export OPENROUTER_API_KEY="..."
export OPENROUTER_CHALLENGER_MODEL="deepseek/deepseek-chat"   # + WEAK/STRONG/JUDGE

# dry run (nothing sent)
DRY_RUN=1 bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh

# pilot generation (10/stage, <=200 requests, <=$5) — scoring runs automatically
MAX_ACCEPTED_PER_STAGE=10 OPENROUTER_MAX_REQUESTS=200 OPENROUTER_MAX_SPEND_USD=5 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh

# scoring (standalone, read-only)
bash experiments/nestful_synthetic_curriculum_v3/scripts/data/score_v4_agentic_dataset.sh

# stage probe (GPU pod; compare v3.1 vs agentic, reward v3.1 vs v3.2)
DATASET=experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter/filtered/stage2_2call_agentic_openrouter.jsonl \
  REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 SEED=42 BACKEND=vllm \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh

# full generation (mirrors deterministic v4 per-stage counts; real cost)
CONFIRM_FULL_AGENTIC_GENERATION=1 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_full.sh
```

Stop after pilot + scoring. Train only if: quality verdict ≥ technically
acceptable, distribution beats v3.1, solver gap positive, and the stage probe
beats v3.1 — and even then, "useful" is decided only by a same-batch official
NESTFUL eval.
