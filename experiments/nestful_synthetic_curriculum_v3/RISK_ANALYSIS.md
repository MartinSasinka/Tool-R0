# Risk Analysis

| risk | mitigation | status |
|------|------------|--------|
| Long-chain in stage1/2 breaks curriculum | v3.1 prefix decomposition | **fixed** |
| too_few_calls persistence | v3.1 stepwise reward + prefix training | **mitigated** (untested) |
| Thin stage2 | v3.1: 800 exact 2-call prefix tasks | **fixed** |
| Math-only tool shift | tool_registry_v3_1 (6 families) | **pilot_ready** |
| Stage2 dead_group | lower replay ratio (0.20), stepwise reward | planned |
| Stage3 non-scalar share below 30% | soft WARN only; pilot uses stage1–2 | **WARN 15.8%** |
| Synthetic ≠ NESTFUL tool overlap | prototype-only until dev gates | documented |

**Training started: NO** — next: pod dry-run, then stage1 pilot.
