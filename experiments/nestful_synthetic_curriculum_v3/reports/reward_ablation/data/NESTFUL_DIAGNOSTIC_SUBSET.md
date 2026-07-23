# NESTFUL_DIAGNOSTIC_SUBSET — 500-task fixed diagnostic eval subset

Generated: 2026-07-23T14:07:39.765270+00:00

## Disclosure

This exact NESTFUL test split has already been used in prior diagnostics (pure_stage3_2ep_20260719_221918 C0/E1/E2 evals and the reward-variant audit). This 500-task subset is an INTERNAL ablation/development evaluation set for comparing reward arms under identical conditions — it is NOT a held-out/untouched final test. Round 3's final internal confirmation evaluates on the full untrimmed n=1661 NESTFUL test set.

## Source

- Path: `experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl`
- Rows: 1661
- SHA-256: `917ce6ec8686c97f2c54da662b57438e584fa51eb5b7985d5aeb30160a2915a4`
- C0 baseline used ONLY for stratification metadata (never for arm selection): `experiments/nestful_synthetic_curriculum_v3/outputs/runs/pure_stage3_2ep_20260719_221918/eval/C0_test/final_eval_trajectories.jsonl`
  (sha256 `ca7f49a535d24cc76d924ca936f7e62de72a111b0f42e33cecfbd9c714ae8b3d`)

## Subset

- Seed: 20260724
- Selected: 500 / target 500
- IDs file SHA-256: `90e018f2b7ce106ddc4f9654bcd982b4c87e714f320d95c215d7ede1fa35c8e9`
- Identical IDs frozen for C0 and every reward arm: True

## Primary axis — gold call-count bucket (target 100/bucket)

| bucket | available | target | selected |
|---|---:|---:|---:|
| 2 | 543 | 100 | 100 |
| 3 | 363 | 100 | 100 |
| 4 | 223 | 100 | 100 |
| 5 | 154 | 100 | 100 |
| 6+ | 378 | 100 | 100 |

## motif_type distribution

| value | source (1661) | source % | subset (500) | subset % |
|---|---:|---:|---:|---:|
| fan_in | 255 | 15.4% | 98 | 19.6% |
| fan_out | 1 | 0.1% | 0 | 0.0% |
| independent_calls | 23 | 1.4% | 6 | 1.2% |
| linear_dependency | 850 | 51.2% | 196 | 39.2% |
| long_chain | 532 | 32.0% | 200 | 40.0% |

## c0_official_win distribution (secondary stratification axis)

| value | source (1661) | source % | subset (500) | subset % |
|---|---:|---:|---:|---:|
| False | 757 | 45.6% | 218 | 43.6% |
| True | 904 | 54.4% | 282 | 56.4% |

## c0_failure_primary taxonomy (reported, not a sampling axis)

| value | source (1661) | source % | subset (500) | subset % |
|---|---:|---:|---:|---:|
| correct keys, wrong argument values | 149 | 9.0% | 56 | 11.2% |
| correct tool, wrong argument keys | 43 | 2.6% | 9 | 1.8% |
| correct trajectory, wrong final answer | 43 | 2.6% | 11 | 2.2% |
| executable trajectory ending wrong result | 169 | 10.2% | 37 | 7.4% |
| no tool call | 109 | 6.6% | 36 | 7.2% |
| parse/format error | 72 | 4.3% | 13 | 2.6% |
| too few calls | 13 | 0.8% | 1 | 0.2% |
| too many calls | 1 | 0.1% | 0 | 0.0% |
| win | 904 | 54.4% | 282 | 56.4% |
| wrong tool | 158 | 9.5% | 55 | 11.0% |

## tool_family distribution (reported, not a sampling axis)

| value | source (1661) | source % | subset (500) | subset % |
|---|---:|---:|---:|---:|
| math | 1189 | 71.6% | 404 | 80.8% |
| other | 472 | 28.4% | 96 | 19.2% |

## Notes / limits

- Primary stratification is the gold call-count bucket (2, 3, 4, 5, 6+); all five buckets have
  >=100 available tasks in the full 1661-task test set, so the ~100/bucket target is met exactly
  (500 = 5 x 100).
- Secondary stratification inside each bucket is (`motif_type`, `c0_official_win`) via the
  largest-remainder method, so both motif mix and C0 success/failure mix are preserved within each
  call-count bucket, not just in aggregate.
- `c0_failure_primary` / `tool_family` / argument-reference complexity are reported for
  transparency but are not independent sampling axes (crossing all of them with call-count and
  motif would produce too many near-empty cells for a stable, deterministic allocation).
- Selection depends only on dataset metadata (gold calls -> motif/call-count) and the C0 baseline;
  no reward-arm rollout or evaluation result is used anywhere in this selection.
