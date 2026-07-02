# NESTFUL Coverage Definition

Date: 2026-07-02

Correct NESTFUL coverage is **not** a single metric. Pilot readiness requires four layers.

---

## A. Structural motif coverage

**Measures:**
- `motif_type` distribution vs NESTFUL
- `num_calls_bucket` distribution
- `dependency_depth` distribution
- fan_in / fan_out / reference_reuse rates
- `baseline_failure_motif_coverage`
- motif KL / L1 distance (synthetic vs NESTFUL)

**Minimum gate (pre-training):**
| gate | threshold |
|------|-----------|
| motif coverage | ≥ 80% |
| baseline-failure motif coverage | ≥ 80% |
| validation failures | 0 |
| invalid references | 0 |
| gold replay | 100% |

**Current (1030-task pilot dataset):** motif coverage **80%**, bf coverage **100%**, gold replay **100%**.

Coverage formula: fraction of NESTFUL motif types where v3 share ≥ 50% of NESTFUL share.

---

## B. Tool / output realism

**Measures:**
- tool_family_distribution
- tool name diversity
- tool bigram / trigram overlap with NESTFUL
- argument schema diversity
- output_type / answer_type distribution
- distractor_tool_count distribution

**Current:**
- tool name diversity: NESTFUL 4202 / v3 **25**
- family overlap (Jaccard): **0.323** (was 0.194)
- bigram overlap: **0.009** (still low)
- scalar output share: **83.2%** (was 100%)
- non-scalar output share: **16.8%**

**Status:** `partial_tool_realism` — improved from `math_only`, **not final-ready**.

Final transfer claims blocked until IBM-tool registry integration.

---

## C. Execution validity

**Measures:**
- synthetic gold replay success rate
- executor pass rate
- answer match rate
- dependency graph consistency

**Gate:** gold replay success rate = **1.0**

**Current:** **100%** (1030/1030 tasks)

---

## D. Behavioral transfer (post-pilot only)

**Measures after stage1–2 pilot on real NESTFUL dev:**
- dev ReAct Win vs baseline dev Win
- motif-level delta vs baseline
- regression on baseline-win tasks
- gain on baseline-fail tasks
- avg_call_count drop
- strict_trace retention
- dead_group_rate

**Conclusion:** Structural + execution gates can pass while behavioral transfer is unknown. Pilot decides whether motif reward + GRPO signal exist before investing in full IBM tool generator.

---

## Readiness summary

| layer | status |
|-------|--------|
| A structural | PASS (80% motif, prototype mode) |
| B tool realism | partial_tool_realism — pilot OK, not final |
| C execution | PASS |
| D behavioral | **not measured yet** |

**Training started: NO**
