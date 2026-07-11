# OpenRouter cost report

Generated 2026-07-10T09:51:24.497940+00:00 | backend openrouter

- requests: 39 (budget 20000)
- cache hits: 0
- retries: 0 | json-mode fallbacks: 0
- prompt tokens: 27807 | completion tokens: 9283
- estimated spend: $0.0149 (budget $15.00)

## By role

| role | requests | cache hits | prompt toks | completion toks | spend USD |
|---|---|---|---|---|---|
| challenger | 5 | 0 | 5739 | 5904 | 0.006212 |
| weak_solver | 13 | 0 | 7817 | 1253 | 0.003084 |
| strong_solver | 21 | 0 | 14251 | 2126 | 0.005591 |

Spend uses OpenRouter `usage.cost` when present, otherwise fallback prices OPENROUTER_PRICE_PROMPT_PER_M / OPENROUTER_PRICE_COMPLETION_PER_M. API keys are never logged.
