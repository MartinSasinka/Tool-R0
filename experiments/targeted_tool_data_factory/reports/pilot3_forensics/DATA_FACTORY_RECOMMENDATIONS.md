# DATA_FACTORY_RECOMMENDATIONS

## Modes

- `PROFILE_SAFE`: TargetProfile / factory metadata only.
- `DIAGNOSTIC_INFORMED_EXPLORATORY`: uses diagnostic-500 deeply; must not be confirmed on the same 500.

### REC_001 (PROFILE_SAFE / P0)

- quota: increase unique topologies ≥2×; cap top-1 topology share ≤10%
- mechanism: Reduce shape collapse and force surface-invariant dependency skill.
- evidence: ['train300 top1 topology share=0.300', 'diagnostic unseen topology rate vs train300=0.240', 'exact namespace overlap vs diagnostic=0.013']
- confidence: high

### REC_002 (PROFILE_SAFE / P0)

- quota: increase fan-in/reuse cells 1.5–2× vs current linear share
- mechanism: Teach non-linear dependency / fan-in that diagnostic programs use.
- evidence: ['linear chains dominate many factory curricula historically', 'unseen topology rate=0.240']
- confidence: medium

### REC_003 (PROFILE_SAFE / P0)

- quota: reserve 20–30% cells for mid-difficulty; reduce trivially saturated cells
- mechanism: Increase fraction of mixed-reward groups so GRPO updates are informative.
- evidence: ['dead_group_rate=0.51', 'mean_unique_rewards low in train log aggregates', 'subset shuffle_interpretation=likely_interleaved_or_shuffled']
- confidence: high

### REC_004 (PROFILE_SAFE / P1)

- quota: ≥50% of A-track with NESTFUL-like keys
- mechanism: Close surface/schema gap that blocks transfer of dependency skill.
- evidence: ['reference syntax audit differences train vs diagnostic', 'exact overlap=0.013']
- confidence: high

### REC_005 (DIAGNOSTIC_INFORMED_EXPLORATORY / P1)

- quota: allocate 15–25% exploratory mass to top unmet joint cells
- mechanism: Directly fill joint OOD cells associated with unchanged failures.
- evidence: ['coverage×outcome tables', 'gained/lost pattern table', 'joint unseen combination rate']
- confidence: medium
- disclaimer: This recommendation makes diagnostic-500 part of the development loop. It must not be validated as a final confirmatory claim on the same 500 tasks.

### REC_006 (PROFILE_SAFE / P1)

- quota: cap easy 2-call all-success-prone cells; keep discrimination-focused 2/3-call
- mechanism: Improve first-tool selection under realistic confusion sets.
- evidence: ['wrong_first_tool transitions in failure taxonomy', 'distractor hardness train vs diag']
- confidence: medium

### REC_007 (PROFILE_SAFE / P2)

- quota: cap same question skeleton share below 5%
- mechanism: Prevent reward hacking via question template cues.
- evidence: ['anti-shortcut audit template concentration']
- confidence: medium
