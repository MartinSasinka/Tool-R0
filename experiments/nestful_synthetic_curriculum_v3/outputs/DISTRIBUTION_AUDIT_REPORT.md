# Distribution Audit Report

Status: **PASS**

## NESTFUL motif distribution
- tasks analyzed: 1861
- top motifs: [('linear_dependency', 956), ('long_chain', 595), ('fan_in', 282), ('independent_calls', 27), ('fan_out', 1)]

## Old synthetic vs NESTFUL gaps
- KL (from compare script): 0.5302136844772503
- missing motifs: ['fan_out', 'independent_calls']

## New synthetic v3 vs NESTFUL
- v3 tasks: 1030
- motif KL(nestful||v3): 0.6322
- coverage (v3 >= 50% nestful share): 80.0%

## Baseline failure motif coverage
- rate: 100.0%
- uncovered: (none)

## Warnings
- (none)

## Recommendations
- Use weighted NESTFUL sampling (not equal per-family split).
- Ensure independent_calls generator is active.
- Run baseline dev eval before training to refresh failure specs.
- Do NOT use nestful_test.jsonl for generation or validation inputs.
