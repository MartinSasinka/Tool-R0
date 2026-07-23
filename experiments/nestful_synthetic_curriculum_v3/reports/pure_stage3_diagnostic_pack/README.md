# Pure Stage-3 diagnostic pack

**Generated:** 2026-07-23T07:29:40.277095+00:00
**Cases:** 258

## Cohort counts

```json
{
  "C0_win_E2_loss": 93,
  "C0_loss_E2_win": 35,
  "official_win_reward_too_few": 55,
  "E2_executable_wrong_other": 35,
  "stable_win_control": 20,
  "stable_loss_control": 20
}
```

## Reward variant summary (pseudo-group C0/E1/E2 per task, n=1661)

| Variant | dead_group | C0win→E2loss wrong order | too_few on official win | exec_wrong reward≥0.52 |
|---------|----------:|-------------------------:|------------------------:|------------------------:|
| R0 | 0.630 | 22 | 1584 | 367 |
| R1 | 0.790 | 0 | 0 | 1182 |
| R2 | 0.626 | 0 | 0 | 1182 |
| R3 | 0.626 | 0 | 0 | 0 |

## Turn-2 confusion (case subset, script counts)

```json
{
  "n_rows": 115,
  "E2_wrong": 60,
  "confusion_type_counts": {
    "different_tool": 28,
    "E2_correct": 55,
    "missing_second_call": 32
  },
  "top_tool_pairs": [
    {
      "gold": "add",
      "E2": "divide",
      "count": 4
    },
    {
      "gold": "subtract",
      "E2": "multiply",
      "count": 4
    },
    {
      "gold": "divide",
      "E2": "multiply",
      "count": 3
    },
    {
      "gold": "multiply",
      "E2": "divide",
      "count": 3
    },
    {
      "gold": "divide",
      "E2": "add",
      "count": 3
    },
    {
      "gold": "subtract",
      "E2": "divide",
      "count": 2
    },
    {
      "gold": "subtract",
      "E2": "add",
      "count": 2
    },
    {
      "gold": "power",
      "E2": "square_edge_by_perimeter",
      "count": 1
    },
    {
      "gold": "delete_value",
      "E2": "parse_list_of_numbers",
      "count": 1
    },
    {
      "gold": "add",
      "E2": "multiply",
      "count": 1
    },
    {
      "gold": "add",
      "E2": "subtract",
      "count": 1
    },
    {
      "gold": "duration_to_string",
      "E2": "divide",
      "count": 1
    },
    {
      "gold": "multiply",
      "E2": "subtract",
      "count": 1
    },
    {
      "gold": "subtract",
      "E2": "floor",
      "count": 1
    }
  ]
}
```

## Next: weak-model annotation

1. Run `annotation_inputs/named/*.json` and `anonymized/*.json` with `annotation_prompt.txt`
2. Merge to `diagnostic_annotations.jsonl`
3. Re-run pack or fill `cluster_input_template.json`
4. Escalate disagreements / low confidence / C0_win_E2_loss to strong model