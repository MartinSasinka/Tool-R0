# Distribution report — agentic v4 vs v3.1 vs NESTFUL

Generated 2026-07-11T08:42:16.643555+00:00 | agentic rows: 10

## Total-variation distance to NESTFUL (lower = closer)

| dimension | agentic_v4 | v3_1 | v4_deterministic |
|---|---|---|---|
| call_count_dist | 0.6728 | 0.3045 | 0.232 |
| offered_tools_dist | 0.6738 | 0.8918 | 0.6387 |
| tool_arity_dist | 0.2098 | 0.1137 | 0.2244 |
| arg_type_dist | 0.1587 | 0.1455 | 0.0331 |
| answer_type_dist | 0.1773 | 0.2313 | 0.1236 |

**Mean distance:** agentic_v4=0.3785, v3_1=0.3374, v4_deterministic=0.2504

## Corpus statistics

```json
{
  "agentic_v4": {
    "n_rows": 10,
    "call_count_dist": {
      "2": 10
    },
    "offered_tools_dist": {
      "10": 1,
      "11": 1,
      "13": 1,
      "14": 1,
      "17": 1,
      "18": 2,
      "19": 1,
      "23": 1,
      "25": 1
    },
    "tool_arity_dist": {
      "1": 50,
      "2": 111,
      "3": 7
    },
    "arg_type_dist": {
      "array": 2,
      "number": 26,
      "reference": 10
    },
    "answer_type_dist": {
      "scalar": 10
    },
    "motif_dist": {
      "long_chain": 3,
      "argument_binding": 3,
      "reference_reuse": 2,
      "distractor_heavy": 2
    },
    "mean_question_words": 30.4
  },
  "v3_1": {
    "n_rows": 3200,
    "call_count_dist": {
      "1": 800,
      "2": 800,
      "3": 800,
      "4": 504,
      "5": 233,
      "6": 63
    },
    "offered_tools_dist": {
      "4": 3,
      "5": 135,
      "6": 373,
      "7": 639,
      "8": 1676,
      "9": 9,
      "12": 284,
      "15": 78,
      "17": 3
    },
    "tool_arity_dist": {
      "1": 10046,
      "2": 15039,
      "3": 431
    },
    "arg_type_dist": {
      "array": 711,
      "number": 7594,
      "object": 15,
      "reference": 5422,
      "string": 2041
    },
    "answer_type_dist": {
      "boolean": 373,
      "list": 487,
      "object": 150,
      "scalar": 1922,
      "string": 268
    },
    "motif_dist": {
      "baseline_failure_inspired": 611,
      "atomic_from_baseline_failure": 278,
      "two_call_baseline_failure_prefix": 262,
      "atomic_from_object_list": 222,
      "two_call_object_field_prefix": 197,
      "four_to_six_call_long_chain": 189,
      "three_call_long_chain_prefix": 173,
      "two_call_reference_passing": 167,
      "three_call_reference_reuse": 166,
      "atomic_from_linear": 155,
      "three_call_argument_transformation": 154,
      "atomic_from_long_chain": 145,
      "three_call_object_list": 129,
      "two_call_long_chain_prefix": 118,
      "three_call_fan_in": 112,
      "three_call_linear_chain": 66,
      "two_call_linear_prefix": 56
    },
    "mean_question_words": 32.1
  },
  "v4_deterministic": {
    "n_rows": 3200,
    "call_count_dist": {
      "2": 800,
      "3": 800,
      "4": 800,
      "5": 417,
      "6": 383
    },
    "offered_tools_dist": {
      "9": 1,
      "10": 254,
      "11": 247,
      "12": 250,
      "13": 252,
      "14": 229,
      "15": 223,
      "16": 310,
      "17": 322,
      "18": 305,
      "19": 328,
      "20": 89,
      "21": 89,
      "22": 92,
      "23": 74,
      "24": 64,
      "25": 71
    },
    "tool_arity_dist": {
      "1": 14414,
      "2": 34987,
      "3": 1534
    },
    "arg_type_dist": {
      "array": 424,
      "number": 11194,
      "reference": 8836,
      "string": 205
    },
    "answer_type_dist": {
      "boolean": 105,
      "scalar": 2990,
      "string": 105
    },
    "motif_dist": {
      "long_chain": 800,
      "argument_binding": 800,
      "reference_reuse": 800,
      "distractor_heavy": 800
    },
    "mean_question_words": 60.2
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
    "mean_question_words": 33.3
  }
}
```

Question length (mean words): agentic_v4=30.4, v3_1=32.1, v4_deterministic=60.2, nestful=33.3
