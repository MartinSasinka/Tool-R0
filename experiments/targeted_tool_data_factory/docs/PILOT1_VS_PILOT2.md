# pilot1 vs pilot2

Generated 2026-07-26 12:03 UTC.

Both pools are 320 and 320 selected tasks respectively. The pipeline, gates and oracle are the same; what changed is the semantic core (typed answer kinds, unit propagation), the motif mix, the surface layer (paraphrasing) and the distractor policy.

## Answer types

The single biggest pilot1 gap: near-total float dominance against a NESTFUL dev profile that is much more mixed.

| answer_type | pilot1 | pilot2 | delta |
|---|---|---|---|
| `float` | 97.2 % | 74.1 % | -23.1 pp |
| `list` | 0.0 % | 6.9 % | +6.9 pp |
| `string` | 0.0 % | 6.6 % | +6.6 pp |
| `int` | 0.0 % | 6.2 % | +6.2 pp |
| `numeric_string` | 2.8 % | 3.1 % | +0.3 pp |
| `bool` | 0.0 % | 3.1 % | +3.1 pp |

## Graph motifs

Fan-in was the second gap: pilot1 was chain-heavy.

| motif | pilot1 | pilot2 | delta |
|---|---|---|---|
| `linear` | 70.3 % | 55.6 % | -14.7 pp |
| `fan_in` | 21.9 % | 37.5 % | +15.6 pp |
| `branch_aggregate` | 7.8 % | 6.9 % | -0.9 pp |

## Call counts

| calls | pilot1 | pilot2 | delta |
|---|---|---|---|
| `2` | 37.8 % | 37.5 % | -0.3 pp |
| `3` | 17.2 % | 17.5 % | +0.3 pp |
| `4` | 13.4 % | 14.1 % | +0.6 pp |
| `6` | 11.6 % | 12.5 % | +0.9 pp |
| `5` | 9.4 % | 9.7 % | +0.3 pp |
| `7` | 6.9 % | 3.1 % | -3.8 pp |
| `8` | 3.8 % | 5.6 % | +1.9 pp |

## Track

| track | pilot1 | pilot2 | delta |
|---|---|---|---|
| `A` | 59.7 % | 59.7 % | +0.0 pp |
| `G` | 40.3 % | 40.3 % | +0.0 pp |

## Distractors and offered sets

| metric | pilot1 | pilot2 |
|---|---|---|
| mean offered tools | 10.71 | 11.31 |
| mean hard distractors | 2.44 | 2.13 |
| mean easy distractors | 4.81 | 5.67 |
| tasks with hard distractors | 100.0 % | 80.3 % |

pilot1 hard-coded hard distractors onto effectively every task, which teaches the model only the maximally adversarial regime. pilot2 makes the share configurable and leaves a deliberate fraction with ordinary offered sets.

## Surface diversity

- **pilot1**: 20 distinct templates, largest share 5.3 % (`indirect_v1`)
- **pilot2**: 27 distinct templates, largest share 4.1 % (`sequence_v1`)

- **pilot2** query provenance: `openrouter_paraphrase` 57.2 %, `template` 42.8 %

## Semantic plausibility

`plausibility_class` does not exist in pilot1 — it is a pilot2 concept. In pilot2:

| class | n | share |
|---|---|---|
| `abstract_coherent` | 239 | 74.7 % |
| `natural` | 81 | 25.3 % |

## What did NOT change

- executor-only oracle and deterministic replay;
- the V1-V6 validation ladder and its hard gates;
- structural (family-level) splitting and the leakage audit;
- contamination and dedup against the target benchmark;
- the exported GRPO record contract.

## Why pilot1 is not simply patched

pilot1 is left byte-identical on disk. It is the control condition: if pilot2 is re-derived by editing pilot1 in place, the comparison stops being reproducible.
