# OpenRouter cost report

Generated 2026-07-11T08:42:16.643555+00:00 | backend openrouter

- requests: 47 (budget 1000)
- cache hits: 1
- retries: 0 | json-mode fallbacks: 0
- prompt tokens: 27423 | completion tokens: 6411
- estimated spend: $0.0063 (budget $20.00)

## By role

| role | requests | cache hits | prompt toks | completion toks | spend USD |
|---|---|---|---|---|---|
| challenger | 3 | 1 | 3453 | 2972 | 0.001581 |
| weak_solver | 0 | 0 | 0 | 0 | 0.0 |
| strong_solver | 33 | 0 | 21471 | 2713 | 0.003925 |
| judge | 11 | 0 | 2499 | 726 | 0.000794 |

Spend uses OpenRouter `usage.cost` when present, otherwise fallback prices OPENROUTER_PRICE_PROMPT_PER_M / OPENROUTER_PRICE_COMPLETION_PER_M. API keys are never logged.
