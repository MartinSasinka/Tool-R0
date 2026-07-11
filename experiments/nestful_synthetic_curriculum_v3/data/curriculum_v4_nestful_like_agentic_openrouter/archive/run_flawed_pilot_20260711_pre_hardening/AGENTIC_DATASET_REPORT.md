# Agentic OpenRouter dataset report

Generated 2026-07-11T08:42:16.643555+00:00 | seed 44 | backend openrouter
Models: challenger=deepseek/deepseek-v3.2 weak=Qwen/Qwen3-4B-Instruct-2507 strong=qwen/qwen3-235b-a22b-2507 judge=deepseek/deepseek-v3.2
Tool schema source policy: `aggregate_style_only` (synthetic registry, aggregate NESTFUL style only — no exact NESTFUL signatures).

## Counts

| stage | accepted | target | status |
|---|---|---|---|
| stage2_2call_agentic_openrouter | 10 | 10 | complete |

Accepted total: 10 | rejected: 9 | acceptance rate: 0.526
Dataset status: **complete** — a partial dataset is still valid and scoreable, but training_candidate stays false until targets are met.
Stop status: completed

## Mean challenger rounds per accepted example

- stage2_2call_agentic_openrouter: 0.4

## Top rejection reasons

| reason | count |
|---|---|
| weak_solver_passed | 3 |
| unresolved_var | 2 |
| invalid_schema | 2 |
| not_nestful_like | 1 |
| metadata_leakage | 1 |
