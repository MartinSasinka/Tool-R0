# PILOT2 REPORT — targeted tool-data factory

Verdict: **READY**

Frozen dataset: 320 tasks (train 160 / structural held-out 80 /
reserve 80), seed 20260726, generator engine v2.

## 1. Counts

| stage | count |
|---|---|
| candidates generated | 2495 |
| validated (V1-V6 pass, deduped, uncontaminated) | 2175 |
| rejected | 320 |
| deduplicated | 4 |
| contaminated | 4 |
| selected (frozen) | 320 |

## 2. Answer types (gap 1)

| answer type | pilot2 | pilot1 | NESTFUL dev |
|---|---|---|---|
| bool | 3.1 % | 0.0 % | 2.0 % |
| float | 74.1 % | 97.2 % | 77.0 % |
| int | 6.2 % | 0.0 % | 5.0 % |
| list | 6.9 % | 0.0 % | 7.0 % |
| numeric_string | 3.1 % | 2.8 % | 2.0 % |
| string | 6.6 % | 0.0 % | 7.0 % |

- float answer share 0.741 vs NESTFUL dev 0.770 (requested band 0.78-0.82)
- answer-type L1 distance to NESTFUL dev: pilot2=0.070 vs pilot1=0.420

## 3. Graph motifs (gap 2)

| motif | pilot2 | pilot1 |
|---|---|---|
| branch_aggregate | 6.9 % | 7.8 % |
| fan_in | 37.5 % | 21.9 % |
| linear | 55.6 % | 70.3 % |

## 4. Semantic plausibility (gap 3)

| class | share |
|---|---|
| abstract_coherent | 74.7 % |
| natural | 25.3 % |

Engine v2 propagates units through the DAG and refuses to emit a chain that
feeds an incompatible unit into a typed operation, so `artificial_composition`
is structurally impossible rather than merely rare (cap was 15 %).

## 5. Surface diversity (gap 4)

| metric | pilot2 | pilot1 |
|---|---|---|
| distinct templates | 27 | 20 |
| largest template share | 4.1 % | 5.3 % |
| largest cell share | 4.4 % | 5.9 % |
| mean query length | 197.0 | 190.1 |
| LLM-paraphrased share | 57.2 % | 0.0 % |

## 6. Hard distractors (gap 5)

| metric | pilot2 | pilot1 |
|---|---|---|
| tasks with >=1 hard distractor | 80.3 % | 100.0 % |
| mean offered tools | 11.31 | 10.71 |

## 7. Call counts and references

| calls | pilot2 | pilot1 |
|---|---|---|
| 2 | 37.5 % | 37.8 % |
| 3 | 17.5 % | 17.2 % |
| 4 | 14.1 % | 13.4 % |
| 5 | 9.7 % | 9.4 % |
| 6 | 12.5 % | 11.6 % |
| 7 | 3.1 % | 6.9 % |
| 8 | 5.6 % | 3.8 % |

Mean reference-argument share: 0.393
(pilot1 0.381).

## 8. Profile match vs NESTFUL dev

| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| pilot2 | 0.0030 | 0.0457 | 0.0253 | 0.0026 | 0.69 | 29.7 | 0.525 |
| pilot1 | 0.0033 | 0.0644 | 0.0331 | 0.1140 | 0.71 | 23.2 | 0.550 |
| stage3 (old) | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## 9. OpenRouter paraphrasing

| field | value |
|---|---|
| model | `mistralai/mistral-small-24b-instruct-2501` |
| date (UTC) | 2026-07-26T11:50:32Z |
| requests sent | 899 (cap 2000) |
| prompt tokens | 318258 |
| completion tokens | 104060 |
| measured cost | $0.0242 (cap $2.0) |
| shortlisted tasks | 1800 |
| accepted paraphrases | 755 |
| fell back to template | 1045 |
| reverted at re-validation | 0 |
| cache hits | 901 |

Top rejection reasons:

| reason | count |
|---|---|
| numeric tokens changed | 2063 |
| dependency references dropped | 432 |
| operation 2 (ratio_of) missing or reordered | 35 |
| operation 1 (floor_divide) missing or reordered | 31 |
| V3 | 27 |
| too long | 26 |
| operation 2 (floor_divide) missing or reordered | 25 |
| operation 1 (inverse) missing or reordered | 20 |
| operation 3 (ratio_of) missing or reordered | 18 |
| operation 3 (floor_divide) missing or reordered | 16 |
| operation 4 (ratio_of) missing or reordered | 13 |
| operation 1 (ratio_of) missing or reordered | 11 |

## 10. Hard gates

| gate | status |
|---|---|
| deterministic replay | 100.0 % |
| schema / oracle / reference errors | 0 |
| exact or near target contamination (selected) | 0 |
| split leakage | none |
| accepted single-tool shortcuts | 0 |
| minimal call count == metadata | yes |
| template share <= 5 % | 4.1 % |
| generation cell share <= 10 % | 4.4 % |
| references inside array arguments | 0 |

## 11. Trainer gold-replay preflight

| dataset | replayed | reference args | status |
|---|---|---|---|
| `train_grpo_pilot2.jsonl` | 160/160 | 440 | PASS |
| `heldout_grpo_pilot2.jsonl` | 80/80 | 216 | PASS |

Adapter registry hash: `55cd6806a0da4603c69ab8856adee646a56b810910bae036025d5488df8a1141`

## 12. Local student probe

Status: **NOT_RUN_LOCAL** — no OpenAI-compatible endpoint answered; see docs/LOCAL_PROBE_REPORT.md for the exact command

## 13. Artifacts

| artifact | sha256 |
|---|---|
| `analysis_csv` | `72c7edd8c317d4b00cfee830e22cb6a8…` |
| `canonical` | `e066f6becf2bc3c4e30352d6ffb959cd…` |
| `grpo_train_ready` | `2232f844da08ba2ad65ea79497d2d3db…` |
| `heldout_grpo` | `8d77581eeaa152936de5d44339489381…` |
| `heldout_nestful` | `4da21e55921970149ebadcc9b8534a9d…` |
| `nestful_compat` | `005245a1392c80111f9741f4b071b6af…` |
| `reserve_grpo` | `ee40fac8860ad91c3c8d7f1168a45e22…` |
| `reserve_nestful` | `ee2e72bc9f4b6251b8a7fc5ee8f5a637…` |
| `train_grpo` | `51481495a3f27deb4ea20e9ab4a4c831…` |
| `train_nestful` | `32a9c99e292ae1fe571f4966a007e601…` |

## 14. Examples

- **ttdf_1000941721cb** (A, 2 calls, linear, float, template): A school tracks these figures for its yearbook. Step 1: compute the ratio of 409 to 83. Step 2: decrease that result by 51 percent.
- **ttdf_068257a109d7** (A, 2 calls, linear, float, template): A shop reviews its figures. Step 1: round 897.2837 to 3 decimal places. Step 2: divide 6039 by that result. (Note: the team has 7 members, which is not needed here.)
- **ttdf_01441b6413d8** (A, 4 calls, fan_in, float, openrouter_paraphrase): Find the remainder of 445 divided by 50. Next, calculate how many whole times 56 fits into the remainder. Then, decrease 1164 by 39 percent. Average the result of the step 2 and that value. Only repor
- **ttdf_4720ea0a3695** (G, 4 calls, branch_aggregate, float, template): I need the final value after the following steps: round 32.95 down to a whole number; round 698.5903 to 3 decimal places; multiply 222 by 29; find the spread between the largest and smallest of the re
- **ttdf_25c7a9dcff69** (A, 6 calls, branch_aggregate, float, openrouter_paraphrase): First, determine 25 percent of 1874, then find the average of 515 and that result. Next, multiply 302 by 11, then calculate that result as a percentage of 560. After that, raise 417 by 39 percent. Las
- **ttdf_1383fff20ea7** (G, 2 calls, linear, float, template): A school tracks these figures for its yearbook. Step 1: average 469 and 56. Step 2: find the remainder of that result divided by 6. What is the final result?
- **ttdf_045f760dd7ad** (G, 2 calls, linear, float, openrouter_paraphrase): First, round 485.61 up to the nearest whole number. Then, find the remainder of the result of the rounding when divided by 41.
- **ttdf_04e494f5a70b** (A, 2 calls, linear, bool, template): A hiking club recorded these values. Step 1: compute 95 percent of 3541. Step 2: check whether that result lies between 2013 and 4715.

## 15. Verdict

**READY**

- no failing gate

