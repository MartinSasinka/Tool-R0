# Process Filter Report (v3.1)

- pass_rate: **1.0000**
- total_input: 3200
- total_passed: 3200
- total_failed: 0
- duplicate_ids: 0

## Stage stats
- stage1_1call_atomic: 800/800 passed
- stage2_2call_dependency: 800/800 passed
- stage3_3call_composition: 800/800 passed
- stage4_4to6call_persistence: 800/800 passed

## Notes
- Optional LLM judge for ambiguous alternative traces — TODO.

## Hard fail gates
- pass_rate == 1.0: PASS
- duplicate_ids == 0: PASS
