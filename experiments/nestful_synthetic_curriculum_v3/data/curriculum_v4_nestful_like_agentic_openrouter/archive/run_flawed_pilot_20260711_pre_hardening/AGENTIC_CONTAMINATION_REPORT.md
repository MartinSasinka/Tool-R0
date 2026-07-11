# Contamination report

Generated 2026-07-11T08:42:16.643555+00:00

- NESTFUL questions / gold traces / tool schemas copied: **NONE**. The challenger only sees the synthetic tool registry (written from scratch; aggregate NESTFUL naming/arity style only) and its own recipe feedback — it is never shown NESTFUL items.
- tool_schema_source_policy: `aggregate_style_only`.
- Overlap gate (question hash, trace hash, sample_id vs NESTFUL dev/test/full): checked per candidate AND re-checked over the final corpus — final overlap = **0** across 10 accepted rows.
- Candidates rejected for overlap during generation: 0.
- The build ABORTS if NESTFUL reference data is unavailable (gate cannot run) or if overlap rejections repeat (10 strikes).
