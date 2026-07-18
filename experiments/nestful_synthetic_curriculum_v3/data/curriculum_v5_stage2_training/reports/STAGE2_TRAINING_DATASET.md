# Stage 2 v5 training dataset (rematerialized)

- **Rows:** 496 / 496 input
- **Registry:** 5.0.2 `f945b18ccdc260b1…`
- **Source field:** `curriculum_v5_synthetic_tools_agentic_openrouter`
- **Input:** `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\data\curriculum_v4_nestful_like_agentic_openrouter\filtered\stage2_2call_agentic_openrouter.jsonl`
- **Output:** `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\data\curriculum_v5_stage2_training\filtered\stage2_2call_agentic_openrouter.jsonl`
- **Manifest:** `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\data\curriculum_v5_stage2_training\manifests\curriculum_v5_stage2_training_manifest.json`

## Schema drift fixed

- `character_count`: required `['text']` → `['text']`; outputs `['output_0']` → `['output_0']`
- `format_as_currency`: required `['amount', 'currency_symbol']` → `['amount']`; outputs `['output_0']` → `['output_0']`
- `hours_to_minutes`: required `['hours']` → `['hours']`; outputs `['value']` → `['output_0']`
- `monthly_installment`: required `['loan_amount', 'num_months']` → `['loan_amount', 'num_months']`; outputs `['output_0']` → `['output_0']`
- `remaining_stock`: required `['initial_stock', 'units_sold']` → `['initial_stock', 'units_sold']`; outputs `['output_0']` → `['output_0']`
- `repeat_word`: required `['times', 'word']` → `['times', 'word']`; outputs `['output_0']` → `['output_0']`
- `split_bill_evenly`: required `['num_people', 'total_amount']` → `['num_people', 'total_amount']`; outputs `['output_0']` → `['output_0']`

## Tier mix

- frontier: 339 (68.3%)
- partial_frontier: 157 (31.7%)

## Semantic warnings (kept, 44)

These rows replay correctly but chain semantically incompatible quantity families (legacy v4 acceptances). Pass `--apply-semantic-filter` to exclude them.

## Diversity

- motif dominance: 0.2782
- tool_family dominance: 0.2157
