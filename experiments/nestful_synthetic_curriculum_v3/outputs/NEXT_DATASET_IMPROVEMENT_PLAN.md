# Next Dataset Improvement Plan (v3.1)

Updated for prefix/motif-aware call-count curriculum. **Do not** add full 7–9 call long-chain tasks to stage1/2.

## A. Prefix distribution by failure motif

| failure motif | stage1 prefix | stage2 prefix | stage3 prefix | stage4 task |
|---|---|---|---|---|
| long_chain / too_few_calls | first tool call | 2-call ref | 3-call continuation | 4–6 call persistence |
| fan_in / wrong_argument | atomic input calls | paired inputs | call1+call2→call3 | fan-in with distractors |
| object/list wrong extraction | get_field/get_item | object→field | field→transform→answer | nested object/list |
| reference_reuse invalid ref | atomic producer | producer→consumer | reuse across two consumers | reuse with distractors |

## B. Generation strategy (implemented in v3.1)

+ long-chain-derived prefix tasks distributed by exact call count:
  - stage1: 1-call atomic first-step tasks from long-chain motifs
  - stage2: 2-call long-chain prefixes with reference passing
  - stage3: 3-call long-chain prefixes
  - stage4: 4–6 call long-chain tasks after gates
  - **no full 7–9 call long-chain in stage1/2**

Pipeline: full trajectories → prefix decomposition → process filter → integrity check.

## C. Stage targets (v3.1 built)

| stage | target | built |
|---|---:|---:|
| stage1_1call_atomic | 800 | **800** |
| stage2_2call_dependency | 800 | **800** |
| stage3_3call_composition | 800 | **800** |
| stage4_4to6call_persistence | 800 | **882** |

## D. Tool/output realism (v3.1)

- 6 tool families (math, string, list, object, boolean, lookup)
- 33 distinct tools — status **pilot_ready**
- TODO: IBM/NESTFUL name-pattern registry for final_experiment_ready

## E. Reward (v3.1)

- `execution_aware_v3_1_stepwise` — stage-aware, hard caps on too_few_calls and premature final
- See `outputs/curriculum_v3_1/REWARD_V3_1_DESIGN.md`

## F. Gates

- Stage3/4 training blocked until dev gates pass (see `configs/curriculum_v3_1.yaml`)
- Replay ratios: s2=0.20, s3=0.25, s4=0.30
