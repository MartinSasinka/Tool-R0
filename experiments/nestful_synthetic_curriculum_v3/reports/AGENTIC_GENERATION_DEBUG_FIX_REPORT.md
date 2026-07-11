# Agentic OpenRouter generation — debug & fix report

Date: 2026-07-10. Scope: the "228 accepted / 0 written" inconsistency from the
first real overnight OpenRouter run. **Zero API money was spent during this
debug** (mock backend + offline cache-only replay). No training, no NESTFUL
eval was launched.

## Root cause

1. **Accepted rows were held only in a local variable.**
   `Orchestrator.generate_stage()` returned the accepted list on success; a
   `StageBudgetStop` at iteration 400 propagated before the return, so the
   228 accepted stage2 rows were lost with the stack frame. `write_outputs`
   then wrote an empty dict → empty `filtered/`, `accepted: {}` manifest,
   `Accepted total: 0` report, `0 rows` scorer.
2. **Solver-gap report counted a different event.** Its "accepted" flag was
   set at the solver-gap gate, before the LLM judge; the judge later rejected
   6 of the 234 gap-passers → true accepted 228. Reports were computed from
   two different sources (orchestrator log vs. accepted dict).
3. **Target table printed ≠ target table used.** The deterministic-v4 mirror
   prints stage4=1600 (two det. stages map to agentic stage4), then the
   `OPENROUTER_MAX_ACCEPTED_PER_STAGE=800` cap was applied silently.

Full forensic analysis: `data/curriculum_v4_nestful_like_agentic_openrouter/reports/AGENTIC_DEBUG_REPORT.md`.

## Fixes implemented

### Crash-safe persistence (lib/agentic_data/orchestrator.py)
- `StageWriter`: every accepted row is appended to the stage-level
  `filtered/*.jsonl` **immediately** with flush + fsync. A hard kill can no
  longer lose accepted rows.
- Accepted rows also live on `orch.accepted_by_stage[stage]` (assigned at
  stage start), so exceptions cannot discard them; `main()` reads that state,
  not return values.
- Stage summaries are written in a `finally:` block with
  `status: partial|complete`.
- `write_outputs` finalizes files atomically (`.tmp` → fsync → `os.replace`)
  and raises on any memory/disk row-count disagreement.

### Count consistency (builder)
- After writing, the builder re-reads the manifest and compares:
  memory count == filtered-file rows == manifest `accepted` ==
  manifest `stage_files.rows` == stage summary == solver-gap-log accepted.
  Any disagreement prints all numbers and exits with code 7.
- The solver-gap log now records `gap_passed` and `final_status`
  (`accepted` / `judge_rejected:<reason>`); `accepted` is set only at final
  accept.

### Target resolution (builder)
- `resolve_targets()` resolves mirror → pilot → CLI override → env cap in one
  place, prints exactly one FINAL table, and writes the full decision
  (including the explicit `stage4_decision: 1600 mirrored → 800 used`) to the
  manifest. Fallback without a det. manifest: 800/800/800.

### Partial-dataset support (score_dataset_quality.py)
- New completeness section (per-stage rows vs manifest target, status
  complete/partial); partial datasets score normally (exit 0);
  `training_candidate` is false until targets are complete; 0 rows writes a
  minimal report and exits 1 with a clear message instead of just erroring.
- Salvaged files (`*.partial_salvaged.jsonl`) are skipped when the base file
  also has rows (no double counting).

### Offline cache-only mode + salvage
- `OpenRouterClient(offline=True)` / `OPENROUTER_OFFLINE=1`: serves ONLY from
  the prompt-hash cache and raises `OfflineCacheMiss` otherwise — zero-spend
  guarantee.
- Builder `--salvage`: deterministic offline replay of a previous run from
  `raw/cache/`, writing `filtered/*.partial_salvaged.jsonl`; archives the
  pre-salvage manifest as `*.pre_salvage.json`.

### Calibration (affects future paid runs)
- `WEAK_SOLVER_MODE` (`minimal` default = historical behavior;
  `handicapped` = 400 tokens, no continuation-pressure hint) and
  `STRONG_SOLVER_MODE` (`scaffolded` default; `plain`).
- Strong-pass policy made explicit + strong-score distribution with a
  near-threshold band in the solver-gap report: 0 candidates in [0.70, 0.80),
  so `strong >= 0.80` ≡ true executable win (1.0) — no threshold edge bug.
- Challenger prompt hardened (exact arg keys, no trivial arithmetic, chaining
  required) and `repair_candidate()` deterministically fixes unambiguous
  argument-key mistakes before the executor gate (which already runs before
  any solver spend).

## Salvage result

- **228/228 rows recovered** to
  `filtered/stage2_2call_agentic_openrouter.partial_salvaged.jsonl` via
  offline replay (cache had all 2,840 responses; replay matched the overnight
  run exactly: 1,749 rejections, identical reason counts).
- Scoring: gold replay 1.0, schema 1.0, 0 duplicates, 0 NESTFUL overlap →
  `technically_acceptable=True`, `status=partial`,
  `training_candidate=False` (targets not met, no probe yet).

## Files changed / added

| file | change |
|---|---|
| `lib/agentic_data/orchestrator.py` | StageWriter, accepted state on orchestrator, finally-summaries, gap_passed/final_status, atomic write_outputs, count checks |
| `scripts/data/build_curriculum_v4_agentic_openrouter.py` | single-source resolve_targets, source-of-truth from orchestrator state, `--salvage`, consistency check (exit 7), report fixes, completion in manifest |
| `scripts/data/score_dataset_quality.py` | completeness section, partial/empty handling, salvage-file dedup, verdict gating |
| `scripts/data/openrouter_client.py` | `offline` mode + `OfflineCacheMiss` |
| `lib/agentic_data/solvers.py` | WEAK_SOLVER_MODE / STRONG_SOLVER_MODE, solver_params |
| `lib/agentic_data/challenger.py` | hardened prompt, `repair_candidate` |
| `tests/test_agentic_persistence.py` | NEW — 24 regression tests |
| `data/.../reports/AGENTIC_DEBUG_REPORT.md` | NEW — forensic incident analysis |

## Tests run

- `tests/test_agentic_persistence.py` — 24/24 PASS (early-stop persistence,
  builder/manifest/report/file count consistency, target-table consistency,
  offline cache-miss, partial + empty scoring, repair_candidate).
- `tests/test_agentic_data.py` — 22/22 PASS (no regressions).
- Mock end-to-end builder (5/stage): exit 0, `count consistency OK (15 accepted)`.
- Offline salvage replay of the overnight run: 228 rows recovered, count
  consistency OK, scorer exit 0 on the partial dataset.

## Remaining risks

- The salvage replay depends on loop determinism; any prompt-affecting code
  change invalidates the cache for replay (the challenger prompt has since
  been hardened, so THIS salvage cannot be re-run — its outputs are final).
- Acceptance rate is still low (~11.5%): weak solver solves 64.5% of solver-
  stage candidates and ~40% of challenger output fails deterministic gates.
  Expect ~2.5x cost per accepted row until the calibration changes prove out
  in a small pilot.
- `stage3/stage4` are untouched (0 rows) — the overnight run never reached them.

## Next small pilot (real API, tiny budget)

```bash
cd <repo root>
export OPENROUTER_API_KEY="..."
export WEAK_SOLVER_MODE=handicapped         # widen weak/strong gap
export OPENROUTER_MAX_REQUESTS=200
export OPENROUTER_MAX_SPEND_USD=2
python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py \
  --max-accepted-per-stage 5 --seed 43
python experiments/nestful_synthetic_curriculum_v3/scripts/data/score_dataset_quality.py
```

Watch: acceptance rate (target > 0.2), `weak_solver_passed` share (target
< 50%), `non_executable_gold_trace + invalid_schema` share (target < 25%),
and that the count-consistency line prints OK.
