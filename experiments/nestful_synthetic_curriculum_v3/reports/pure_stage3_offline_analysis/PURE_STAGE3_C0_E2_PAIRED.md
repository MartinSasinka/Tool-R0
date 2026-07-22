# Pure Stage 3 — C0 vs E2 Paired (NESTFUL test, n=1661)

Generated: 2026-07-22T14:03:48.878904+00:00
Run: `pure_stage3_smoke_20260719_213722` (smoke pipeline; **8-task train**, full test eval)
E1 test eval **not available** — analysis is C0 → E2 only.

## 1. Eval parity

**Parity OK:** True

| Check | OK |
|---|---|
| same_1661_task_ids | True |
| paired_by_task_id | True |
| same_eval_set_path | True |
| same_temperature | True |
| same_top_p | True |
| same_num_rollouts | True |
| same_paradigm | True |
| same_base_model_revision | cdbee75f17c01a7cc42f958dc650907174af0554 |
| same_vllm_tp | True |
| same_parser_pipeline | True |
| same_official_scorer_pipeline | True |

### Provenance hashes

| Arm | adapter hash | task set sha | prompt sha | scorer sha |
|---|---|---|---|---|
| C0 | `none` | `917ce6ec8686c97f…` | `aeaae71414b47acc…` | `c1a850ab56a840dc…` |
| E2 | `bbc8cbb17c3354fa7959857730140d235a50b1327681d2127de8af3b24410776` | `917ce6ec8686c97f…` | `aeaae71414b47acc…` | `c1a850ab56a840dc…` |

## 2. Headline metrics

| Metric | C0 | E2 | Δ |
|---|---:|---:|---:|
| win_rate | 0.5382 | 0.5340 | -0.0042 |
| f1_func_mean | 0.6208 | 0.6225 | +0.0017 |
| f1_param_mean | 0.2974 | 0.2965 | -0.0008 |
| first_tool_accuracy | 0.5707 | 0.5738 | +0.0031 |
| full_sequence_accuracy | 0.0217 | 0.0223 | +0.0006 |
| executability | 0.7899 | 0.7887 | -0.0012 |
| final_answer_accuracy | 0.5876 | 0.5882 | +0.0006 |
| under_calling | 0.6051 | 0.6057 | +0.0006 |
| over_calling | 0.0566 | 0.0608 | +0.0042 |
| avg_pred_calls | 2.2613 | 2.2721 | +0.0108 |

**Paired:** gained 79 / lost 86 / net -7
McNemar p=0.640428787412791

### Transitions (no E1)

| Category | count |
|---|---:|
| stable_win | 808 |
| stable_loss | 688 |
| lost_after_E2 | 86 |
| gained_after_E2 | 79 |

## Key question: wrong values ↓ but wrong tool / exec-wrong ↑?

- wrong argument values: C0 8.61% → E2 9.09% (+0.48 pp)
- wrong tool: C0 9.99% → E2 9.87% (-0.12 pp)
- executable wrong result: C0 10.96% → E2 10.96% (+0.00 pp)

**Pattern present:** False

## By gold call count (win rate)

| bucket | n | C0 | E2 | Δ |
|---|---:|---:|---:|---:|
| 2 | 543 | 0.4567 | 0.4604 | +0.0037 |
| 3 | 363 | 0.6198 | 0.5978 | -0.0220 |
| 4 | 223 | 0.6009 | 0.6009 | +0.0000 |
| 5 | 154 | 0.5649 | 0.5714 | +0.0065 |
| 6+ | 378 | 0.5291 | 0.5238 | -0.0053 |

## By motif (win rate)

| motif | n | C0 | E2 | Δ |
|---|---:|---:|---:|---:|
| fan_in | 255 | 0.6392 | 0.6314 | -0.0078 |
| fan_out | 1 | 0.0000 | 1.0000 | +1.0000 |
| independent_calls | 23 | 0.2609 | 0.2609 | +0.0000 |
| linear_dependency | 850 | 0.5153 | 0.5094 | -0.0059 |
| long_chain | 532 | 0.5395 | 0.5376 | -0.0019 |