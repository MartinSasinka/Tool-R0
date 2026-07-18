# Distribution report — agentic v5 vs NESTFUL

Generated 2026-07-16T18:00:49.640127+00:00 | agentic rows: 8

## Total-variation distance to NESTFUL (lower = closer)

| dimension | agentic_v5 |
|---|---|
| call_count_dist | 0.7813 |
| offered_tools_dist | 0.596 |
| tool_arity_dist | 0.0788 |
| arg_type_dist | 0.0219 |
| answer_type_dist | 0.3861 |

**Mean distance:** agentic_v5=0.3728

## Corpus statistics

```json
{
  "agentic_v5": {
    "n_rows": 8,
    "call_count_dist": {
      "3": 8
    },
    "offered_tools_dist": {
      "9": 2,
      "10": 1,
      "12": 4,
      "15": 1
    },
    "tool_arity_dist": {
      "1": 39,
      "2": 44,
      "3": 8
    },
    "arg_type_dist": {
      "array": 1,
      "number": 23,
      "reference": 16
    },
    "answer_type_dist": {
      "boolean": 3,
      "scalar": 4,
      "string": 1
    },
    "motif_dist": {
      "long_chain": 2,
      "argument_binding": 2,
      "distractor_heavy": 2,
      "fan_in": 2
    },
    "tool_family_dist": {
      "math": 9,
      "conversion": 4,
      "text": 3,
      "comparison": 3,
      "scheduling": 1,
      "statistics": 1,
      "logistics": 1,
      "finance": 1,
      "geometry": 1
    },
    "question_template_dist": {
      "narrative_other": 5,
      "first_then": 2,
      "conditional": 1
    },
    "mean_question_words": 27.8,
    "dominance": {
      "motif": 0.25,
      "answer_type": 0.5,
      "tool_family": 0.375,
      "question_template": 0.625
    }
  },
  "nestful": {
    "n_rows": 1861,
    "call_count_dist": {
      "2": 609,
      "3": 407,
      "4": 250,
      "5": 173,
      "6": 134,
      "7": 75,
      "8": 213
    },
    "offered_tools_dist": {
      "7": 14,
      "8": 127,
      "9": 520,
      "10": 676,
      "11": 71,
      "12": 8,
      "13": 52,
      "14": 45,
      "15": 46,
      "16": 35,
      "17": 64,
      "18": 109,
      "19": 80,
      "20": 13,
      "21": 1
    },
    "tool_arity_dist": {
      "0": 79,
      "1": 10462,
      "2": 10047,
      "3": 218,
      "4": 28,
      "5": 4
    },
    "arg_type_dist": {
      "array": 180,
      "boolean": 5,
      "number": 8731,
      "object": 14,
      "reference": 6218,
      "string": 265
    },
    "answer_type_dist": {
      "boolean": 39,
      "list": 101,
      "object": 17,
      "scalar": 1531,
      "string": 173
    },
    "motif_dist": {},
    "tool_family_dist": {
      "unknown": 8091,
      "geometry": 12,
      "math": 1,
      "comparison": 1
    },
    "question_template_dist": {
      "conditional": 655,
      "interrogative": 618,
      "narrative_other": 548,
      "first_then": 31,
      "enumerated": 9
    },
    "mean_question_words": 33.3,
    "dominance": {
      "motif": null,
      "answer_type": 0.8227,
      "tool_family": 0.9983,
      "question_template": 0.352
    }
  }
}
```
