# COST REPORT — pilot3

Local CPU-only run. No GPU, no remote API.

| step | wall s | cpu s | peak python MB |
|---|---|---|---|
| export | 16.2 | 14.1 | 62 |
| generate | 118.8 | 114.8 | 493 |
| generate_expand | 58.0 | 56.3 | 310 |
| paraphrase | 8013.0 | 822.5 | 282 |
| probe | 2.4 | 2.3 | 44 |
| report | 3.3 | 3.1 | 59 |
| select | 20.2 | 19.2 | 272 |
| split | 7.2 | 7.1 | 45 |
| validate | 2098.4 | 2043.7 | 376 |

- total wall time: 10337.6 s
- total CPU time: 3083.1 s
- peak python-allocated RAM (tracemalloc, per step max): 493 MB (process RSS is higher; tracemalloc tracks python allocations only)
- outputs disk usage: 354.7 MB
- local model inference: none
- LLM call count: 0
- paid API cost: £0 (no remote endpoint enabled)
