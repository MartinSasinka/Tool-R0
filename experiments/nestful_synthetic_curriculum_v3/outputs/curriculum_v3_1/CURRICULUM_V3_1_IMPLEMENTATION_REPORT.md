# Curriculum v3.1 Implementation Report

Generated: 2026-07-03 (uniqueness + dedup pass)

## Stage counts

| Stage | Count |
|---|---:|
| full trajectories | **888** |
| stage1_1call_atomic | **800** |
| stage2_2call_dependency | **800** |
| stage3_3call_composition | **800** |
| stage4_4to6call_persistence | **800** |

## Call-count integrity

**PASS** — exact num_calls per stage preserved.

## Gold replay success

**1.0**

## Process filter pass rate

**1.0**

## Question–trace alignment

**PASS** — failures=0, unresolved=0, constant/reference mismatch=0

## Uniqueness (dedup-aware generation)

| Metric | Value |
|---|---:|
| exact_duplicate_count | **0** |
| mean_unique_question_ratio | **1.0** (3200/3200) |
| mean_trace_duplicate_ratio | **0.0003** |
| stage1 unique Q ratio | **1.0** |
| stage2 unique Q ratio | **1.0** |
| stage3 unique Q ratio | **1.0** |
| stage4 unique Q ratio | **1.0** |
| stage1 trace dup ratio | **0.0** |
| stage2 trace dup ratio | **0.0013** |
| stage3 trace dup ratio | **0.0** |
| stage4 trace dup ratio | **0.0** |

Soft WARN: question-template duplicate ratio high on stage1–3 (expected — same skill, different literals/questions). Skill repetition is allowed; exact/trace repetition is not.

## Used tool diversity

| Metric | Value |
|---|---:|
| offered tools | **37** |
| used tools (global gold_calls) | **25** |
| used tool families | **6** |
| non-scalar stage2+ share | **≥30%** |

## Output type distribution (filtered)

scalar 1900, list 438, string 330, boolean 282, object 250

## Preflight

**PASS_PILOT_READY** (hard fails=0, soft warns on per-stage template concentration)

## Pytest

**41/41 passed**

## Pod dry-run allowed?

**Yes** — all hard gates pass.

## Stage1–2 pilot allowed?

**Yes** — with `ALLOW_PROTOTYPE_TRAINING=1`, reward `execution_aware_v3_1_stepwise`.

Training / GPU eval / test eval: **not started** (by policy).

## Key changes

- `uniqueness_utils_v3_1.py` — signatures + stage-aware `StageDedupRegistry`
- `analyze_dataset_uniqueness_v3_1.py` — uniqueness analyzer + reports
- Dedup-aware prefix build — no blind upsample clones; synth fill for gaps
- Question template variants per tool family (trace-aligned)
- Wider argument ranges in trajectory generator
