# Motif Schema

## Terms

| Term | Definition |
|------|------------|
| `num_calls` | Gold tool calls in trace |
| `dependency_graph` | DAG: nodes=calls, edges=reference→producer |
| `dependency_depth` | Longest reference chain length |
| `linear_chain` | Each call refs only immediate predecessor |
| `independent_calls` | At least one call with no refs |
| `fan_in` | Multiple preds feed one call |
| `fan_out` | One producer refs used by multiple successors |
| `reference_reuse` | Same producer referenced more than once |
| `nested_reference_depth` | Refs skipping immediate predecessor |
| `argument_complexity` | Count of reference arguments |
| `output_type` | Type of final tool output |
| `answer_type` | Type of gold_answer |
| `tool_family` | Coarse tool category (math/string/list/other) |
| `tool_sequence_bigram` | Adjacent tool name pairs |
| `tool_sequence_trigram` | Adjacent tool name triples |
| `distractor_tools` | Tools in schema not used in gold trace |
| `alternative_valid_trace` | Valid non-gold path to same answer |
| `baseline_failure_cluster` | Dev losses grouped by motif + failure mode |
| `motif_type` | Primary structural label for curriculum staging |
| `difficulty_score` | Normalized [0,1] structural complexity score |

## Task JSON schema

```json
{
  "task_id": "string",
  "question": "string",
  "tools": [],
  "gold_calls": [],
  "gold_answer": null,
  "num_calls": 0,
  "motif_type": "linear_dependency",
  "dependency_graph": {
    "nodes": [],
    "edges": []
  },
  "reference_pattern": {
    "num_references": 0,
    "fan_in_count": 0,
    "fan_out_count": 0,
    "reuse_count": 0,
    "nested_reference_depth": 0
  },
  "output_type": "scalar|string|list|object|array|boolean|mixed",
  "answer_type": "scalar|string|list|object|array|boolean|mixed",
  "difficulty_score": 0.0,
  "source_motif_cluster": null,
  "generation_seed": 0
}
```

Implementation: [`scripts/motif_lib.py`](scripts/motif_lib.py).
