# Uniqueness Improvement Report (v3.1)

| Stage | before_uq_ratio | after_uq_ratio | before_trace_dup | after_trace_dup | exact_dup_after | status |
|---|---:|---:|---:|---:|---:|---|
| stage1_1call_atomic | 1.0 | 1.0 | 0.0 | 0.0 | 0 | WARN |
| stage2_2call_dependency | 1.0 | 1.0 | 0.0013 | 0.0013 | 0 | WARN |
| stage3_3call_composition | 1.0 | 1.0 | 0.0 | 0.0 | 0 | WARN |
| stage4_4to6call_persistence | 1.0 | 1.0 | 0.0 | 0.0 | 0 | WARN |

- curriculum integrity preserved: stage counts={'stage1_1call_atomic': 800, 'stage2_2call_dependency': 800, 'stage3_3call_composition': 800, 'stage4_4to6call_persistence': 800}
- dedup warnings: 5744
- overall uniqueness status: WARN
