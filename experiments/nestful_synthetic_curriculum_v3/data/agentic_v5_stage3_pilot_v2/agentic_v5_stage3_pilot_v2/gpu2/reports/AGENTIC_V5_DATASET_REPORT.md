# Agentic v5 OpenRouter dataset report

Generated 2026-07-16T17:57:01.887659+00:00 | seed 47 | backend openrouter
Registry: synthetic_tools_v5 version=5.0.0 hash=31a99c56b050281f | 163 tools
Executor: executor.mode=synthetic (REAL execution) everywhere — challenger verify, weak/strong solvers, rollout probe.
Models: challenger=deepseek/deepseek-v3.2 weak=Qwen/Qwen3-4B-Instruct-2507 strong=qwen/qwen3-235b-a22b-2507 judge=deepseek/deepseek-v3.2
Best-of-N: enabled=True max_accepts_per_batch=1
Tool schema source policy: `aggregate_style_only` (synthetic registry, aggregate NESTFUL style only — no exact NESTFUL signatures).

## Counts

| stage | accepted | target | status |
|---|---|---|---|
| stage3_3call_agentic_openrouter | 4 | 10 | partial |

Accepted total: 4 | rejected: 46 | acceptance rate: 0.080
Dataset status: **partial** — a partial dataset is still valid and scoreable, but training_candidate stays false until targets are met.
Stop status: STOPPED EARLY: stage3_3call_agentic_openrouter: iteration budget 10 exhausted at 4/10 accepted

## Mean challenger rounds per accepted example

- stage3_3call_agentic_openrouter: 2.5

## Top rejection reasons

| reason | count |
|---|---|
| low_grpo_signal_prediction | 43 |
| best_of_n_not_selected | 2 |
| not_nestful_like | 1 |

## Best-of-N candidate selection

- candidates that lost the batch ranking (best_of_n_not_selected): 2
- max accepted per batch: 1
