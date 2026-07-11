# Agentic OpenRouter dataset report

Generated 2026-07-10T09:51:24.497940+00:00 | seed 43 | backend openrouter
Models: challenger=deepseek/deepseek-chat weak=deepseek/deepseek-chat strong=deepseek/deepseek-chat judge=deepseek/deepseek-chat
Tool schema source policy: `aggregate_style_only` (synthetic registry, aggregate NESTFUL style only — no exact NESTFUL signatures).

## Counts

| stage | accepted | target | status |
|---|---|---|---|
| stage2_2call_agentic_openrouter | 228 (+0 new) | 800 | partial |

Accepted total: 228 | rejected: 21 | acceptance rate: 0.916
Dataset status: **partial** — a partial dataset is still valid and scoreable, but training_candidate stays false until targets are met.
Stop status: STOPPED EARLY: stage2_2call_agentic_openrouter: acceptance rate 0.000 < 0.02 after 5 batches (0 new accepted / 21 rejected this run) — revise the recipe manually

## Mean challenger rounds per accepted example

- stage2_2call_agentic_openrouter: None

## Top rejection reasons

| reason | count |
|---|---|
| weak_solver_passed | 6 |
| strong_solver_failed | 5 |
| duplicate_trace | 3 |
| invalid_schema | 3 |
| diversity_cap_weak_score | 2 |
| invalid_json | 1 |
| non_executable_gold_trace | 1 |
