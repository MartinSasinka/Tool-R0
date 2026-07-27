# PILOT3 REPORT — targeted tool-data factory

Verdict: **CONDITIONAL**

Frozen dataset: 1000 tasks (train 600 /
structural held-out 200 / reserve 200),
seed 20260727, generator engine v2. Baseline comparison: `pilot2`.

## 1. Counts

| stage | count |
|---|---|
| candidates generated | 2213 |
| validated (V1-V6 pass, deduped, uncontaminated) | 2210 |
| rejected | 3 |
| deduplicated | 0 |
| contaminated | 0 |
| selected (frozen) | 1000 |

## 2. Answer types (gap 1)

| answer type | pilot3 | pilot2 | NESTFUL dev |
|---|---|---|---|
| bool | 3.1 % | 3.1 % | 2.0 % |
| float | 74.0 % | 74.1 % | 77.0 % |
| int | 5.9 % | 6.2 % | 5.0 % |
| list | 6.6 % | 6.9 % | 7.0 % |
| numeric_string | 2.8 % | 3.1 % | 2.0 % |
| string | 7.6 % | 6.6 % | 7.0 % |

- float answer share 0.740 vs NESTFUL dev 0.770 (requested band 0.78-0.82)
- answer-type L1 distance to NESTFUL dev: pilot2=0.068 vs pilot1=0.070

## 3. Graph motifs (gap 2)

| motif | pilot3 | pilot2 |
|---|---|---|
| branch_aggregate | 8.2 % | 6.9 % |
| fan_in | 39.9 % | 37.5 % |
| linear | 51.9 % | 55.6 % |

## 4. Semantic plausibility (gap 3)

| class | share |
|---|---|
| abstract_coherent | 73.0 % |
| natural | 27.0 % |

Engine v2 propagates units through the DAG and refuses to emit a chain that
feeds an incompatible unit into a typed operation, so `artificial_composition`
is structurally impossible rather than merely rare (cap was 15 %).

## 5. Surface diversity (gap 4)

| metric | pilot2 | pilot1 |
|---|---|---|
| distinct templates | 27 | 27 |
| largest template share | 3.8 % | 4.1 % |
| largest cell share | 4.4 % | 4.4 % |
| mean query length | 212.5 | 197.0 |
| LLM-paraphrased share | 39.3 % | 0.0 % |

## 6. Hard distractors (gap 5)

| metric | pilot2 | pilot1 |
|---|---|---|
| tasks with >=1 hard distractor | 81.0 % | 80.3 % |
| mean offered tools | 11.37 | 11.31 |

## 7. Call counts and references

| calls | pilot2 | pilot1 |
|---|---|---|
| 2 | 32.1 % | 37.5 % |
| 3 | 15.3 % | 17.5 % |
| 4 | 14.0 % | 14.1 % |
| 5 | 12.4 % | 9.7 % |
| 6 | 13.4 % | 12.5 % |
| 7 | 7.3 % | 3.1 % |
| 8 | 5.5 % | 5.6 % |

Mean reference-argument share: 0.402
(pilot1 0.393).

## 8. Profile match vs NESTFUL dev

| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| pilot3 | 0.0072 | 0.0517 | 0.0234 | 0.0020 | 0.67 | 45.2 | 0.549 |
| pilot2 | 0.0030 | 0.0457 | 0.0253 | 0.0026 | 0.69 | 29.7 | 0.525 |
| stage3 (old) | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## 9. OpenRouter paraphrasing

| field | value |
|---|---|
| model | `mistralai/mistral-small-24b-instruct-2501` |
| date (UTC) | 2026-07-26T23:56:47Z |
| requests sent | 0 (cap 4500) |
| prompt tokens | 0 |
| completion tokens | 0 |
| measured cost | $0.0000 (cap $5.0) |
| shortlisted tasks | 0 |
| accepted paraphrases | 0 |
| fell back to template | 0 |
| reverted at re-validation | 0 |
| cache hits | 0 |

Top rejection reasons:

| reason | count |
|---|---|
| — | 0 |

## 10. Hard gates

| gate | status |
|---|---|
| deterministic replay | 100.0 % |
| schema / oracle / reference errors | 0 |
| exact or near target contamination (selected) | 0 |
| split leakage | none |
| accepted single-tool shortcuts | 0 |
| minimal call count == metadata | yes |
| template share <= 5 % | 3.8 % |
| generation cell share <= 10 % | 4.4 % |
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
| `analysis_csv` | `6e2a827166717b6328a0ebe26d71028c…` |
| `canonical` | `0f01f8d124f6429735ee684115aa8f10…` |
| `grpo_train_ready` | `dd7bac86b02cc2a0bcf5381733f094ee…` |
| `heldout_grpo` | `99c6a1cf2e8d50c5bcb2dfc6548f43f0…` |
| `heldout_nestful` | `27115cd8822aa52558070c449b62213a…` |
| `nestful_compat` | `badcce161e37dd3365f4f0aa2c80b23c…` |
| `reserve_grpo` | `cdb63ad453d049873c870127f0a7b130…` |
| `reserve_nestful` | `662c932550b685a4abebe9727743d997…` |
| `train_grpo` | `b1bf1d7e24e71521fa6fe34540f75723…` |
| `train_nestful` | `00178b0279d27cfd5d640cf646cec3a5…` |

## 14. Examples

- **ttdf_0982b2624c98** (G, 3 calls, linear, float, openrouter_paraphrase): Calculate the average of 23, 47, 1, 42, 31. Then, increase 670 by that result percent. Finally, express the increase as a percent of 549 and report the result.
- **ttdf_a1f4522870c7** (G, 4 calls, fan_in, string, template): First, round 746.71 up to a whole number. Then negate 231. Then compute the result of step 1 percent of that result. Finally, build an identifier from the prefix 'run' and that result.
- **ttdf_88d02f070bad** (G, 2 calls, linear, float, openrouter_paraphrase): First, round 395.09 down to the nearest whole number. Then, use that result to find the remainder of 2247 divided by it.
- **ttdf_acf64105ab0f** (G, 5 calls, linear, float, template): A school tracks these figures for its yearbook. Step 1: increase 820 by 39 percent. Step 2: average 459 and that result. Step 3: increase 484 by that result percent. Step 4: average 725 and that resul
- **ttdf_52e02b0074e9** (A, 2 calls, linear, float, openrouter_paraphrase): First, determine how many times 10 fits into 424 as a whole number. Then, report the square of that result.
- **ttdf_28600db1173f** (A, 8 calls, fan_in, float, template): Tell me what comes out when I compute how many whole times 12 fits into 80; take the square root of that result; negate 191; subtract 83 from that result; average that result and 27; round that result
- **ttdf_4bcc0bcb441d** (A, 5 calls, fan_in, string, openrouter_paraphrase): Calculate the absolute difference between 175 and 77. Afterwards, 1843 by 57 percent and then square that value. Then, determine the result of step 1 percent of the squared value. Lastly, format that 
- **ttdf_c03e76b84f89** (G, 4 calls, fan_in, string, template): Determine the outcome of this procedure: compute the ratio of 36 to 10; find how many whole minutes fit into 13563 seconds; compute the result of step 1 percent of that result; label that result with 

## 15. Verdict

**CONDITIONAL**

- no failing gate
- WARN: paraphrase share 0.393 outside the 0.60-0.70 target band
