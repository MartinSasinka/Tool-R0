# Synthetic vs NESTFUL Distribution Gaps

Synthetic source: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\clean_curriculum` (2157 tasks)
NESTFUL source: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\NESTFUL-main\data_v2\nestful_data.jsonl` (1861 tasks)
Motif-type KL(nestful||synthetic): 0.5302

## Covered by synthetic (within 5pp of NESTFUL share)
- long_chain

## Underrepresented in synthetic
- fan_in (15.2% nestful vs 1.0% syn)

## Overrepresented in synthetic
- linear_dependency (70.7% syn vs 51.4% nestful)

## Missing entirely from synthetic
- fan_out
- independent_calls

## Stage coverage issue
Old N-call stages (epoch_N_Ncall.jsonl) conflate depth with structure — e.g. stage 2
includes both linear chains and independent calls, but rarely fan-in/fan-out patterns
that dominate harder NESTFUL tasks.
