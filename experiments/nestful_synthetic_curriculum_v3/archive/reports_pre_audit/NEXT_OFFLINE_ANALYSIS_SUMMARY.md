# Next Offline Analysis Summary

Date: 2026-07-02 (from 50-task prototype run)

## 1. NESTFUL motif distribution (n=1861)

| motif_type | count | share |
|---|---|---|
| linear_dependency | 956 | 51.4% |
| long_chain | 595 | 32.0% |
| fan_in | 282 | 15.2% |
| independent_calls | 27 | 1.5% |
| fan_out | 1 | 0.05% |

Call depth: median 3, mean 4.36, max 53. Buckets: 2-call 32.7%, 3-call 21.9%, 4-call 13.4%, 5-8 23.8%, 9+ 8.2%.

## 2. Synthetic v3 prototype coverage (50 tasks)

Generator families (10): linear_dependency, reference_reuse, fan_in, fan_out, object_or_list_output, argument_transformation, distractor_tools, long_chain, alternative_valid_traces, baseline_failure_inspired — **5 tasks each**.

Mapped to NESTFUL motif types in curriculum:
- linear_dependency: present (~10%)
- long_chain: present (~10%)
- fan_in: present (~10%)
- fan_out: present (~10%)
- independent_calls: **absent (0%)**

Additional v3-only labels (reference_reuse, object_or_list_output, etc.) are training motifs but do not appear in NESTFUL classifier taxonomy.

## 3. Missing motif types (vs NESTFUL)

- **independent_calls** — no generator template
- Old clean_curriculum also missing: fan_out, independent_calls

## 4. Underrepresented (vs NESTFUL share, 50-task equal split)

| NESTFUL motif | nestful share | v3 share (50) | status |
|---|---|---|---|
| linear_dependency | 51.4% | ~10% | under |
| long_chain | 32.0% | ~10% | under |
| fan_in | 15.2% | ~10% | borderline |
| independent_calls | 1.5% | 0% | missing |
| fan_out | 0.05% | ~10% | over |

## 5. Baseline failure clusters (dev overlap with full-test trajectories)

| cluster | n | recipe focus |
|---|---|---|
| linear_dependency__too_few_calls | 43 | 2-call linear |
| long_chain__too_few_calls | 29 | ~7-call chains |
| fan_in__too_few_calls | 12 | fan-in refs |
| independent_calls__too_few_calls | 3 | parallel independent calls |

3/4 failure motif types covered; **independent_calls** uncovered.

## 6. Is 50-task prototype sufficient for training?

**No.** Suitable only as **pipeline sanity check** (validation, audit, tests). Reasons:
- Motif coverage 40% (threshold 80%)
- Equal 5-per-family split ignores NESTFUL proportions (51% linear, 32% long_chain)
- Math-only tool registry — tool-family shifted vs real NESTFUL
- No gold replay gate yet at time of first run

## 7. Why coverage is only 40%

Coverage = fraction of NESTFUL motif types where v3 share ≥ 50% of NESTFUL share.

With equal 10% per declared family mapped to 5 NESTFUL types:
- **PASS:** fan_in (10% ≥ 7.6%), fan_out (10% ≥ 0.025%)
- **FAIL:** linear_dependency (10% < 25.7%), long_chain (10% < 16%), independent_calls (0%)

→ 2/5 = **40%**. Root cause: **equal family allocation**, not sample size alone.

## 8. Generator must dogenerate

1. **Weighted sampling** from `nestful_motif_distribution.json` (not equal 5/family)
2. **independent_calls** template (parallel unrelated calls)
3. More **long_chain** (32% NESTFUL) and **linear_dependency** (51%)
4. **Gold replay** executor for math registry
5. Scale to **500+** tasks after weighting fix
6. Tool-family realism report before claiming NESTFUL transfer

---

## Post scale-up update (500-task weighted run)

After weighted sampling + generator fixes:
- Motif coverage: **100%**
- Baseline-failure motif coverage: **100%**
- Gold replay: **100%**
- Preflight: **PASS_PROTOTYPE_ONLY** (math-only tools)
- See `GENERATION_SCALEUP_REPORT.md` and `GENERATOR_GAP_DIAGNOSIS.md`
