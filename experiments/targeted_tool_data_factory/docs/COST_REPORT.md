# COST REPORT — pilot3

Local CPU-only run. No GPU, no remote API.

| step | wall s | cpu s | peak python MB |
|---|---|---|---|
| export | 15.4 | 14.9 | 62 |
| generate | 118.8 | 114.8 | 493 |
| generate_expand | 10.0 | 9.7 | 96 |
| paraphrase | 3.1 | 3.1 | 39 |
| probe | 4.5 | 4.5 | 44 |
| report | 2.5 | 2.1 | 16 |
| select | 17.0 | 16.3 | 140 |
| split | 7.9 | 7.7 | 45 |
| validate | 97.5 | 95.7 | 231 |

- total wall time: 276.7 s
- total CPU time: 268.8 s
- peak python-allocated RAM (tracemalloc, per step max): 493 MB (process RSS is higher; tracemalloc tracks python allocations only)
- outputs disk usage: 261.6 MB
- local model inference: none
- LLM call count: 0
- paid API cost: £0 (no remote endpoint enabled)
