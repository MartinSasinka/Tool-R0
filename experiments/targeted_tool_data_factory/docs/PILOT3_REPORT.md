# PILOT3 REPORT — targeted tool-data factory

Verdict: **NOT_READY**

Frozen dataset: 1000 tasks (train 600 /
structural held-out 200 / reserve 200),
seed 20260727, generator engine v2. Baseline comparison: `pilot2`.

## 1. Counts

| stage | count |
|---|---|
| candidates generated | 5671 |
| validated (V1-V6 pass, deduped, uncontaminated) | 5243 |
| rejected | 428 |
| deduplicated | 16 |
| contaminated | 7 |
| selected (frozen) | 1000 |

## 2. Answer types (gap 1)

| answer type | pilot3 | pilot2 | NESTFUL dev |
|---|---|---|---|
| bool | 4.2 % | 3.1 % | 2.0 % |
| float | 71.6 % | 74.1 % | 77.0 % |
| int | 6.2 % | 6.2 % | 5.0 % |
| list | 6.8 % | 6.9 % | 7.0 % |
| numeric_string | 3.5 % | 3.1 % | 2.0 % |
| string | 7.7 % | 6.6 % | 7.0 % |

- float answer share 0.716 vs NESTFUL dev 0.770 (requested band 0.78-0.82)
- answer-type L1 distance to NESTFUL dev: pilot2=0.112 vs pilot1=0.070

## 3. Graph motifs (gap 2)

| motif | pilot3 | pilot2 |
|---|---|---|
| branch_aggregate | 7.6 % | 6.9 % |
| fan_in | 41.6 % | 37.5 % |
| linear | 50.8 % | 55.6 % |

## 4. Semantic plausibility (gap 3)

| class | share |
|---|---|
| abstract_coherent | 71.3 % |
| natural | 28.7 % |

Engine v2 propagates units through the DAG and refuses to emit a chain that
feeds an incompatible unit into a typed operation, so `artificial_composition`
is structurally impossible rather than merely rare (cap was 15 %).

## 5. Surface diversity (gap 4)

| metric | pilot2 | pilot1 |
|---|---|---|
| distinct templates | 27 | 27 |
| largest template share | 4.0 % | 4.1 % |
| largest cell share | 4.3 % | 4.4 % |
| mean query length | 209.3 | 197.0 |
| LLM-paraphrased share | 54.9 % | 0.0 % |

## 6. Hard distractors (gap 5)

| metric | pilot2 | pilot1 |
|---|---|---|
| tasks with >=1 hard distractor | 79.0 % | 80.3 % |
| mean offered tools | 11.44 | 11.31 |

## 7. Call counts and references

| calls | pilot2 | pilot1 |
|---|---|---|
| 2 | 31.4 % | 37.5 % |
| 3 | 17.5 % | 17.5 % |
| 4 | 13.4 % | 14.1 % |
| 5 | 12.1 % | 9.7 % |
| 6 | 13.3 % | 12.5 % |
| 7 | 5.5 % | 3.1 % |
| 8 | 6.8 % | 5.6 % |

Mean reference-argument share: 0.404
(pilot1 0.393).

## 8. Profile match vs NESTFUL dev

| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| pilot3 | 0.0041 | 0.0487 | 0.0231 | 0.0057 | 0.76 | 42.0 | 0.545 |
| pilot2 | 0.0030 | 0.0457 | 0.0253 | 0.0026 | 0.69 | 29.7 | 0.525 |
| stage3 (old) | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## 9. OpenRouter paraphrasing

| field | value |
|---|---|
| model | `mistralai/mistral-small-24b-instruct-2501` |
| date (UTC) | 2026-07-27T03:13:42Z |
| requests sent | 2658 (cap 4500) |
| prompt tokens | 977352 |
| completion tokens | 343806 |
| measured cost | $0.0764 (cap $5.0) |
| shortlisted tasks | 4500 |
| accepted paraphrases | 1793 |
| fell back to template | 2698 |
| reverted at re-validation | 0 |
| cache hits | 1833 |

Top rejection reasons:

| reason | count |
|---|---|
| numeric tokens changed | 5262 |
| dependency references dropped | 1232 |
| operation 2 (ratio_of) missing or reordered | 88 |
| operation 1 (floor_divide) missing or reordered | 82 |
| V3 | 65 |
| operation 2 (floor_divide) missing or reordered | 60 |
| too long | 60 |
| operation 1 (inverse) missing or reordered | 57 |
| operation 3 (ratio_of) missing or reordered | 48 |
| operation 1 (ratio_of) missing or reordered | 38 |
| operation 4 (ratio_of) missing or reordered | 35 |
| operation 3 (floor_divide) missing or reordered | 28 |

## 10. Hard gates

| gate | status |
|---|---|
| deterministic replay | 100.0 % |
| schema / oracle / reference errors | 0 |
| exact or near target contamination (selected) | 0 |
| split leakage | none |
| accepted single-tool shortcuts | 0 |
| minimal call count == metadata | yes |
| template share <= 5 % | 4.0 % |
| generation cell share <= 10 % | 4.3 % |
| references inside array arguments | 0 |

## 11. Trainer gold-replay preflight

| dataset | replayed | reference args | status |
|---|---|---|---|
| not run | — | — | — |

Adapter registry hash: `n/a`

## 12. Local student probe

Status: **NOT_RUN_LOCAL** — no OpenAI-compatible endpoint answered; see docs/LOCAL_PROBE_REPORT.md for the exact command

## 13. Artifacts

| artifact | sha256 |
|---|---|
| `analysis_csv` | `6e81936836bfd7ccaa45c0777e200a81…` |
| `canonical` | `5deb55092d27e16209126c0f0144bd11…` |
| `grpo_train_ready` | `ae7042b23dfb1b50ee998ea08cbf9cd3…` |
| `heldout_grpo` | `dce7a610153f347e1590241a4f41aaac…` |
| `heldout_nestful` | `c0f1c3ae0d9a5bfaac2efdb5fc36b608…` |
| `nestful_compat` | `12a97bdf22ef47568e09611150275820…` |
| `reserve_grpo` | `711d7093114255b156bf8362471fb09b…` |
| `reserve_nestful` | `6f78e4e8a2a4ac4f6ca15af656e893e5…` |
| `train_grpo` | `77d7e2bf51acd9a998d4cd202f6b7add…` |
| `train_nestful` | `d8b3960e7460218d39b6b8db99bf773b…` |

## 14. Examples

- **ttdf_2134f49bd3ad** (G, 5 calls, branch_aggregate, float, template): An engineer checks a report. Step 1: subtract 11 from 230. Step 2: compute how many whole times 31 fits into that result. Step 3: round 151.78 up to a whole number. Step 4: negate 863. Step 5: add up 
- **ttdf_a59ae8939796** (G, 5 calls, fan_in, float, openrouter_paraphrase): You will need to find the ratio of 524 to 17, then square that result, then increase the result by 9 percent, then increase 1422 by 5 percent first, and lastly, for the final goal, find the percent of
- **ttdf_562e14e68ec4** (A, 2 calls, linear, float, template): Determine the outcome of this procedure: round 562.55 to the nearest whole number; subtract 30 from that result. (Note: the team has 7 members, which is not needed here.)
- **ttdf_4753fd527f32** (G, 4 calls, fan_in, string, template): Find the remainder of 2051 divided by 30, then convert 34 degrees Fahrenheit to Celsius, then compute the ratio of the result of step 1 to that result, and finally build an identifier from the prefix 
- **ttdf_a67dd02aea22** (A, 3 calls, fan_in, float, openrouter_paraphrase): First, take the negative of 459. Then, increase the value of 581 by 28 percent, that result, and finally, divide the result of step 1 by the previous value.
- **ttdf_2930e4ea8e12** (G, 7 calls, linear, float, openrouter_paraphrase): First, determine the ratio of 657 to 11, then round the previous value down to the nearest whole number. Next, take the ratio of that result to 81, then divide that result by 28. After that, work out 
- **ttdf_5142b25ba487** (A, 8 calls, fan_in, float, template): An engineer checks a report. Step 1: add 650 and 78. Step 2: multiply 197 by that result. Step 3: compute 87 percent of that result. Step 4: subtract 76 from that result. Step 5: divide that result by
- **ttdf_3890bd39e665** (A, 2 calls, linear, bool, openrouter_paraphrase): First, find the reciprocal of 7, and then, determine if that result is greater than -3.

## 15. Verdict

**NOT_READY**

- FAIL: answer-type match not better than baseline
- WARN: float answer share 0.716 more than 5 pp from the NESTFUL dev share 0.770
- WARN: paraphrase share 0.549 outside the 0.60-0.70 target band
