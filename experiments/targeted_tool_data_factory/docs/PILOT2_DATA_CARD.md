# Pilot2 data card — `pilot2`

Generated 2026-07-26 12:03 UTC.

## What this dataset is

320 synthetic nested-tool-use tasks produced program-first: a semantic program (a typed DAG over deterministic primitives) is sampled first, executed by the factory executor to obtain the oracle answer and every intermediate observation, and only then rendered into a natural-language question. No LLM ever decides what the answer is.

| field | value |
|---|---|
| version | `pilot2` |
| seed | `20260726` |
| selected tasks | 320 |
| train / structural held-out / reserve | 160 / 80 / 80 |
| target benchmark profile | NESTFUL dev |
| target student | `Qwen/Qwen3-4B-Instruct-2507` |
| oracle | executor-only, deterministic |
| replay rate | 100.0 % |

## Intended use

GRPO training data for the D1 arm of the D0-vs-D1 experiment, and a structural held-out set for in-domain measurement. The held-out split is **structural**: it holds out whole program families / generation cells, not random rows, so a model cannot pass it by memorising a template.

## Out-of-scope use

- This is not a benchmark. It is training data conditioned on a benchmark profile; scoring a model on it and reporting that as a capability number would be circular.
- The reserve split must stay unused until the train/held-out result is written down, otherwise it stops being a reserve.

## Composition

| answer_type | n | share |
|---|---|---|
| `float` | 237 | 74.1 % |
| `list` | 22 | 6.9 % |
| `string` | 21 | 6.6 % |
| `int` | 20 | 6.2 % |
| `bool` | 10 | 3.1 % |
| `numeric_string` | 10 | 3.1 % |

| motif | n | share |
|---|---|---|
| `linear` | 178 | 55.6 % |
| `fan_in` | 120 | 37.5 % |
| `branch_aggregate` | 22 | 6.9 % |

| plausibility | n | share |
|---|---|---|
| `abstract_coherent` | 239 | 74.7 % |
| `natural` | 81 | 25.3 % |

## Known limitations

- The domains are abstract-numeric by construction. `plausibility_class` records how far each task is from a naturally occurring scenario; `artificial_composition` is capped, not eliminated.
- Paraphrases come from one instruct model, so the surface distribution inherits that model's register. The deterministic-template fraction is kept deliberately non-zero as a hedge.
- Difficulty is calibrated against a proxy (a local 4-bit Qwen3-4B), not against the exact training-time policy.
