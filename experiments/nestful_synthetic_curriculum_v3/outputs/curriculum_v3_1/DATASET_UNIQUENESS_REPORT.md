# Dataset Uniqueness Report (v3.1)

Overall status: **WARN**
- exact_duplicate_count (all stages): 0
- mean_unique_question_ratio: 1.0
- mean_trace_duplicate_ratio: 0.0003

## Per-stage summary

| Stage | Status | N | Unique Q ratio | Exact dup | Trace dup ratio | Tools |
|---|---|---:|---:|---:|---:|---:|
| stage1 1call atomic | WARN | 800 | 1.000 | 0 | 0.000 | 15 |
| stage2 2call dependency | WARN | 800 | 1.000 | 0 | 0.001 | 14 |
| stage3 3call composition | WARN | 800 | 1.000 | 0 | 0.000 | 18 |
| stage4 4to6call persistence | WARN | 800 | 1.000 | 0 | 0.000 | 23 |

## Most duplicated signature types

### stage1_1call_atomic
- Top question-template duplicates:
  - count=16: `task: find <NUM> plus <NUM>. return the outcome.`
  - count=16: `please compute the sum of <NUM> and <NUM> and report the answer.`
  - count=16: `using the tools, add <NUM> and <NUM>. what is the result?`
  - count=16: `complete this single-step task: use the addition tool to combine <NUM> and <NUM>`
  - count=16: `your job is to calculate <NUM> + <NUM>. provide the answer.`
- Top tool-sequence duplicates:
  - greater_than (count=132)
  - add (count=80)
  - filter_greater_than (count=80)
  - sort_list (count=76)
  - concat (count=74)

### stage2_2call_dependency
- Top trace duplicates:
  - count=2 hash=c09e495d12aa...
- Top question-template duplicates:
  - count=12: `complete two steps: (<NUM>) check whether <NUM> is greater than <NUM>; (<NUM>) c`
  - count=12: `complete two steps: (<NUM>) compare <NUM> with <NUM> using greater-than; (<NUM>)`
  - count=12: `first, join the strings <STR> and <STR>. then, convert the previous result to lo`
  - count=12: `first, filter the list <LIST> to values greater than <NUM>. then, sort the list `
  - count=12: `perform two operations in order. step one: filter the list <LIST> to values grea`
- Top tool-sequence duplicates:
  - add->multiply (count=139)
  - add->add (count=134)
  - greater_than->less_than (count=128)
  - filter_greater_than->sort_list (count=128)
  - make_object->get_field (count=69)

### stage3_3call_composition
- Top question-template duplicates:
  - count=16: `follow three steps: first check whether <NUM> is greater than <NUM>; then check `
  - count=16: `follow three steps: first create an object with field <STR> set to <STR>; then u`
  - count=16: `follow three steps: first join the strings <STR> and <STR>; then take the output`
  - count=16: `complete a three-step task: return whether <NUM> exceeds <NUM>; next check wheth`
  - count=16: `step <NUM>: create an object with field <STR> set to <STR>. step <NUM>: read fie`
- Top tool-sequence duplicates:
  - add->multiply->multiply (count=123)
  - add->multiply->add (count=123)
  - add->add->multiply (count=119)
  - make_object->get_field->filter_greater_than (count=78)
  - filter_greater_than->sort_list->get_item (count=51)

### stage4_4to6call_persistence
- Top question-template duplicates:
  - count=16: `carry out the following <NUM> steps: step <NUM>: use the addition tool to combin`
  - count=16: `perform these <NUM> operations in order: step <NUM>: compute the sum of <NUM> an`
  - count=16: `work through all <NUM> steps sequentially: step <NUM>: find <NUM> plus <NUM>; st`
  - count=16: `complete each of the <NUM> tool calls: step <NUM>: calculate <NUM> + <NUM>; step`
  - count=16: `execute this <NUM>-step workflow: step <NUM>: add <NUM> and <NUM>; step <NUM>: m`
- Top tool-sequence duplicates:
  - make_object->get_field->filter_greater_than->sort_list (count=125)
  - make_object->get_field->filter_greater_than->sort_list->update_field (count=81)
  - add->multiply->multiply->multiply (count=80)
  - add->add->multiply->add (count=74)
  - add->multiply->add->multiply (count=63)

## Recommendations
- Regenerate upsampled stage2/3/4 slots with dedup-aware trajectory generation.
- Allow skill repetition (motif/tool sequence) but reject exact and excessive trace duplicates.
- Increase question-template variants per tool family.

## Gates
- Hard fail: exact_duplicate_count > 0, unique_question_ratio < 0.40, stage count < 800
- Soft warn: trace_duplicate_ratio > 0.05, template_duplicate_ratio > 0.30
