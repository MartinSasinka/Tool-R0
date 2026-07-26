# COST REPORT — pilot2

Local CPU-only run. No GPU, no remote API.

| step | wall s | cpu s | peak python MB |
|---|---|---|---|
| export | 3.4 | 2.9 | 20 |
| generate | 22.7 | 21.5 | 151 |
| paraphrase | 2087.0 | 213.7 | 139 |
| probe | 0.9 | 0.9 | 14 |
| report | 1.1 | 0.9 | 27 |
| select | 15.1 | 12.5 | 150 |
| split | 1.9 | 1.8 | 14 |
| validate | 1052.7 | 981.9 | 141 |

- total wall time: 3184.9 s
- total CPU time: 1236.1 s
- peak python-allocated RAM (tracemalloc, per step max): 151 MB (process RSS is higher; tracemalloc tracks python allocations only)
- outputs disk usage: 134.4 MB
- local model inference: none
- LLM call count: 0
- paid API cost: £0 (no remote endpoint enabled)
