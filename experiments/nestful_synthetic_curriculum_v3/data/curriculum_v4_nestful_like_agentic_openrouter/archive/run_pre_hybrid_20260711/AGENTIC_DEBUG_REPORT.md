# Agentic generation debug report (overnight run 2026-07-09/10)

Incident: builder printed `iteration budget 400 exhausted at 228/800 accepted`,
but `filtered/` was empty, the manifest said `accepted: {}`,
AGENTIC_DATASET_REPORT.md said `Accepted total: 0`, the distribution report
said `agentic rows: 0`, the scorer said `0 rows` — while
AGENTIC_SOLVER_GAP_REPORT.md said `weak_fail_strong_pass (accepted): 234`.

## 1. Why accepted examples were not written

`Orchestrator.generate_stage()` collected accepted rows in a **local list**
and only returned it on normal completion. When the stage hit the 400-iteration
budget, `StageBudgetStop` was raised **before the return**, so:

- the 228 accepted rows were garbage-collected with the stack frame;
- `main()` caught the exception, but `accepted_by_stage` never received the
  stage2 key;
- `write_outputs()` iterated an empty dict → nothing written to `filtered/`;
- the manifest and all row-based reports were correctly computed from that
  (empty) dict — hence `0` everywhere.

Nothing was wrong with the rows themselves; they were simply never persisted.

## 2. Where the 234 vs 228 mismatch came from

The solver-gap log lives on the orchestrator object (it survived the
exception) and marked `accepted=True` when a candidate **passed the solver-gap
gate** — *before* the LLM style judge. The judge then rejected 6 gap-passing
candidates (`not_nestful_like`: 4, `ambiguous_question`: 2), so:

- gap-gate passes: 234
- finally accepted: 228 (= 234 − 6)

The report labeled the 234 as "accepted", which was wrong. It now reports
`gap_passed` and `finally accepted` separately, and `accepted` is only set at
the final accept step.

## 3. Target-count inconsistency (stage4 800 vs 1600)

`resolve_targets()` printed the raw deterministic-v4 mirror. Deterministic v4
has four stages of 800; both `v4_stage3_4call` and `v4_stage4_5to6call` map to
agentic `stage4_4to6call`, so the raw mirror printed stage4=1600. The
`OPENROUTER_MAX_ACCEPTED_PER_STAGE=800` cap was then applied silently
afterward — the printed table was not the used table.

Now resolved in one function with an explicit, recorded decision:

- `stage4_decision: {mirrored_sum: 1600, used: 800, reason: "two det. stages
  map to agentic stage4; uniform 800/stage kept"}`
- the single FINAL table is printed once and written verbatim to the manifest.

## 4. Salvage result

All 2,840 LLM responses were cached under `raw/cache/`, and the loop is
deterministic given the seed. The run was **fully salvaged with zero API
spend** by replaying it in offline cache-only mode
(`--salvage`, raises `OfflineCacheMiss` rather than ever touching the API):

- salvaged rows: **228** → `filtered/stage2_2call_agentic_openrouter.partial_salvaged.jsonl`
- replay matched the overnight run exactly (1,749 rejections, identical
  reason counts, 228 accepted)
- validation: gold replay pass rate 1.0, schema pass 1.0, 0 duplicates,
  0 NESTFUL overlap → **technically_acceptable = True**
- status: **partial** (228/800 stage2; stages 3-4 never started)
- training_candidate: **False** (targets not met; no probe yet)
- original (buggy) manifest preserved as
  `manifests/curriculum_v4_agentic_openrouter_manifest.pre_salvage.json`

## 5. Rejection analysis (1,749 rejections, 2,000 candidates proposed)

| reason | count | share of candidates |
|---|---|---|
| weak_solver_passed | 690 | 34.5% |
| non_executable_gold_trace | 437 | 21.9% |
| invalid_schema | 264 | 13.2% |
| strong_solver_failed | 118 | 5.9% |
| duplicate_trace | 99 | 5.0% |
| unresolved_var | 87 | 4.4% |
| too_hard_both_solvers_fail | 27 | 1.4% |
| other (leakage/json/judge) | 27 | 1.4% |

**Is the weak solver too strong? Yes.** 690 of 1,069 candidates that reached
the solvers (64.5%) were solved outright by the weak solver (avg weak score
0.813). The weak and strong solver are the same model; a single attempt at
temperature 0.2 is enough for most 2-call tasks the challenger writes.

**Is challenger validity too low? Yes.** 795 of 2,000 candidates (39.8%)
failed the free deterministic gates before any solver ran
(non-executable trace 437 + invalid schema 264 + unresolved var 87 + leakage
7/10). Roughly 40% of challenger tokens produced nothing scoreable.

**Is the strong threshold brittle? No.** The strong-score distribution has
**zero** candidates in [0.70, 0.80); partial-prefix scores are mathematically
capped below 0.8, so `strong >= 0.80` is exactly "true executable win /
solution-equivalent (1.0)". This policy is now stated explicitly in the
solver-gap report.

## 6. Recipe changes needed (implemented / recommended)

Implemented in code (affect the NEXT paid run, not the salvage):

1. `WEAK_SOLVER_MODE=handicapped` — 400-token budget, continuation-pressure
   hint removed from the weak prompt (widens the weak/strong gap).
   Default `minimal` keeps historical behavior. `STRONG_SOLVER_MODE`
   (`scaffolded` default / `plain`) is likewise explicit.
2. Challenger prompt hardened: exact-argument-key rule, "avoid trivial
   single-operation arithmetic; every call after the first consumes an
   earlier result".
3. `repair_candidate()` — deterministic, zero-cost repair of unambiguous
   argument-key mistakes (case/underscore variants, single unknown → single
   missing) BEFORE the executor gate, cutting `non_executable_gold_trace` /
   `invalid_schema` waste.
4. Deterministic gates already run before solver calls (cost order kept).

Recommended for the next paid run:

- set `OPENROUTER_WEAK_MODEL` to a genuinely weaker/cheaper model than the
  strong solver, rather than relying on mode handicaps alone;
- raise per-stage iteration budget only after the acceptance rate improves
  (at the observed 0.57 accepted/iteration, stage2 alone needs ~1,400
  iterations — improve validity + gap first);
- keep `duplicate_trace` in check by widening the per-batch tool subset if it
  grows past ~10%.
