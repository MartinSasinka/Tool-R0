# Distribution report — agentic v5 vs NESTFUL

Generated 2026-07-16T18:02:38.760120+00:00 | agentic rows: 4

## Total-variation distance to NESTFUL (lower = closer)

| dimension | agentic_v5 |
|---|---|
| call_count_dist | 0.7813 |
| offered_tools_dist | 0.693 |
| tool_arity_dist | 0.1949 |
| arg_type_dist | 0.0383 |
| answer_type_dist | 0.1773 |

**Mean distance:** agentic_v5=0.377

## Corpus statistics

```json
{
  "agentic_v5": {
    "n_rows": 4,
    "call_count_dist": {
      "3": 4
    },
    "offered_tools_dist": {
      "10": 1,
      "11": 2,
      "16": 1
    },
    "tool_arity_dist": {
      "1": 15,
      "2": 28,
      "3": 5
    },
    "arg_type_dist": {
      "array": 1,
      "number": 11,
      "reference": 8
    },
    "answer_type_dist": {
      "scalar": 4
    },
    "motif_dist": {
      "long_chain": 2,
      "distractor_heavy": 1,
      "argument_binding": 1
    },
    "tool_family_dist": {
      "math": 6,
      "health": 1,
      "logistics": 1,
      "geometry": 1,
      "statistics": 1,
      "conversion": 1,
      "finance": 1
    },
    "question_template_dist": {
      "narrative_other": 4
    },
    "mean_question_words": 33.5,
    "dominance": {
      "motif": 0.5,
      "answer_type": 1.0,
      "tool_family": 0.5,
      "question_template": 1.0
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
