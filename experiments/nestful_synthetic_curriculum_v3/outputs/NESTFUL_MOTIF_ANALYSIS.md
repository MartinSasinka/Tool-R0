# NESTFUL Motif Analysis

Input: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\NESTFUL-main\data_v2\nestful_data.jsonl` (1861 tasks)

## num_calls
- min=2, max=53, mean=4.36, median=3.0

### buckets
- 2: 609 (32.7%)
- 3: 407 (21.9%)
- 4: 250 (13.4%)
- 5-8: 442 (23.8%)
- 9+: 153 (8.2%)

## Top motif types
- linear_dependency: 956
- long_chain: 595
- fan_in: 282
- independent_calls: 27
- fan_out: 1

## Dependency depth distribution
- depth 2: 748
- depth 3: 510
- depth 4: 295
- depth 5: 175
- depth 6: 74
- depth 7: 34
- depth 8: 16
- depth 9: 5
- depth 10: 2
- depth 12: 2

## Most complex tasks (top 5 difficulty)
- 2e70b1d4-a123-4c7a-b4a7-3e31ac5cb464: difficulty=0.9953, calls=53, motif=long_chain, seq=divide->divide->divide->divide->divide->divide->add->add->divide->divide->divide->divide->add->divide->divide->divide->divide->divide->add->divide->divide->divide->divide->divide->divide->add->divide->divide->divide->divide->divide->divide->divide->add->divide->divide->divide->divide->divide->divide->divide->divide->add->divide->divide->divide->divide->divide->divide->divide->divide->divide->add
- 07df2ad6-7ac0-4c6a-b79a-0b02782d0235: difficulty=0.9931, calls=36, motif=long_chain, seq=add->add->subtract->add->add->divide->multiply->add->add->add->subtract->add->add->divide->add->add->add->subtract->add->add->divide->add->add->add->subtract->add->add->divide->add->add->add->subtract->add->add->divide->add
- e7f6a43e-b1f7-4c00-9f38-46a53c6fa8a6: difficulty=0.9904, calls=26, motif=long_chain, seq=add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add->add
- 6a580394-2cd8-4a80-bac8-d81c88244059: difficulty=0.99, calls=25, motif=long_chain, seq=multiply->add->multiply->add->multiply->add->divide->subtract->add->add->add->multiply->add->multiply->add->multiply->add->divide->subtract->add->add->add->add->divide->multiply
- 0335ed3e-e613-420b-a07a-9d481b2a791e: difficulty=0.965, calls=25, motif=long_chain, seq=multiply->multiply->divide->multiply->multiply->divide->multiply->divide->add->multiply->divide->multiply->multiply->divide->multiply->multiply->divide->multiply->divide->add->add->multiply->multiply->divide->subtract

## Implications for synthetic generation
- Match call-count buckets but prioritize fan-in/fan-out and reference reuse (underrepresented in N-call curriculum).
- Include object/list output types — common in real NESTFUL answers.
- Stage by structural motif, not call count alone.
- Mine baseline failures on dev to oversample hard motif clusters.
