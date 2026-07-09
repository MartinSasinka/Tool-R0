# MASTER AUDIT REPORT — NESTFUL synthetic curriculum (v3/v3.1)

Date: 2026-07-09 · Read-only audit (no code modified, no files deleted, no runs started).

Companion documents in this folder: `DATASET_AUDIT.md/.json`, `METRIC_AUDIT.md`,
`METRIC_STANDARD_PROPOSAL.md`, `RUN_AUDIT.md/.csv/.json`, `REWARD_AUDIT.md`,
`FAILURE_MODE_AUDIT.md`, `STAGE3_AUDIT.md`, `CLEANUP_PLAN.md`, `IMPLEMENTATION_PLAN.md`.
Reproducible extraction scripts: `audits/tools/`.

---

## Q1. Which dataset is canonical?

**`outputs/curriculum_v3_1/filtered/stage{1..4}_*.jsonl` (dataset A, 4×800 rows) plus
`curriculum_v3_1_manifest.json`.** All v3.1 training runs (July 7–8) verifiably trained on
exactly these rows (run `data_base` files match A row-for-row, 800/800 canonical-JSON
identical). Dataset B and the pre-v3.1 `curriculum_v3` corpus are legacy.

## Q2. Are the two synthetic dataset folders equivalent or different?

**Completely different — zero overlap.** Across all 28 A×B file pairs: 0 shared question
hashes, 0 shared gold-trace hashes, 0 shared sample ids. Different generators, different
schemas (A: 21 fields with motif/cluster metadata and precomputed observations; B: 5-field
NESTFUL-shaped records), different tool registries. Neither derives from the other. B
additionally has ~4% unscoreable gold answers (unresolved `$var…$` placeholders), one null
answer, and a misnamed `curriculum_toolr0_all.jsonl` (contains only the 268 6-call rows).
B remains the *default* in `nestful_mtgrpo_{minimal,partial}/config.yaml` paths — a silent
wrong-dataset footgun, though the v3 launcher overrides it.

## Q3. Which metric should be the main paper metric?

**`official_nestful_win_rate_temp0`** — the official NESTFUL `calculate_win_score`
(re-execution of predicted calls with the IBM executable functions), temperature 0.0, on
nestful_test (1,661), with a same-batch baseline, binomial CI, and paired gained/regressed
counts. Everything else (internal win, final_answer_pass, solution_equivalent, trace
matches, F1s, behavioral rates) is secondary diagnostics. Full standard in
`METRIC_STANDARD_PROPOSAL.md`.

## Q4. Why do internal and official win rates differ?

Systematically +6–7 pp (internal higher) in every measured cell. The internal
`internal_metrics_diagnostic.win_rate` scores the **agent's own executed trajectory** with
the internal executor's tolerant answer matching (`matches_gold` fallback, numeric/string
normalization, recovery turns allowed). The official scorer **re-executes the extracted call
list from scratch** with stricter decimal-aware comparison. Borderline formats and messy
traces pass internally, fail officially. The code labels the internal number "diagnostic";
prose and summaries have not always respected that.

## Q5. Are any previous comparisons invalid because of different baselines/eval batches?

Yes, three classes:

1. **Stage-3 full-NESTFUL claims** — the temp0 Stage-3 batch (2026-07-09) has no baseline
   cell; the nearest baseline is from a different batch a day earlier and has no official
   score at all. Any "Stage 3 improved on full NESTFUL" statement is invalid.
2. **Cross-batch sampled-vs-temp0** — the 2026-07-07 batch used sampled decoding, the
   2026-07-08 batch temp0; temp0 shifted every cell ~−1 pp. Deltas across those batches
   conflate decoding with checkpoints.
3. **Cross-era comparisons** — July-2 runs used a different corpus (curriculum_v3), a
   different reward (v2.1 motif, effectively binary) and n_gen=4; not comparable to v3.1 runs.

Within-run dev comparisons and within-batch full-NESTFUL comparisons are valid (baselines
were present).

## Q6. Did any checkpoint truly improve over same-batch baseline?

**No.** On dev (200, official scorer, same-run baseline): every epoch of every run is at or
below its baseline (best case Stage-2 152750: 0.545 vs 0.570; Stage-3: 0.545 vs 0.565).
On full NESTFUL at temp0 (only valid same-batch comparison, internal metric): best checkpoint
+0.38 pp over baseline (0.6024 vs 0.5986) ≈ 7/1,861 tasks — inside noise (±1.1 pp, 1σ). The
one nominally positive result (July-2 stage-1 epoch-2, +2 pp dev official) is legacy-era and
within noise.

## Q7. Is Stage 3 interpretable without a same-batch baseline?

**Not as an improvement claim.** What can be said: training was healthy-ish (17 fractional
reward values, dead-group rate improved 0.71→0.65, avg predicted calls 2.19/3), but the
stage gate failed on position-artifact rate (0.35–0.41 > 0.2), and on dev (same-run
baseline) Stage 3 is −2 to −3 pp official. `STAGE3_AUDIT.md` contains the exact one-batch
command (baseline + s3_e1 + s3_e2, temp0, official scorer) to settle full-NESTFUL ranking.

## Q8. What is the main reason GRPO does not reliably improve NESTFUL?

Two compounding causes, in order:

1. **Reward-band quantization starves GRPO of within-group variance.** The stepwise reward
   collapses diverse completions into a few flat bands (0.3 too-few / 0.6 wrong-args / 1.0).
   65–88% of groups have zero between-completion reward std even at 8 generations, T=1.0
   (unique episode rewards per group: 1.09–1.33). Only 12–35% of groups produce any
   gradient; with lr 5e-7 the surviving signal moves nothing measurable (KL ≈ 0 throughout).
2. **Synthetic→NESTFUL transfer gap.** Even where training reward improves on the synthetic
   stages, the learned skill (2–3-call chains over ~20 toy math/string tools) does not
   intersect NESTFUL's failure mass (heterogeneous real APIs, long chains, argument
   formats). Zero data overlap (good for contamination) also means zero distributional
   anchor. Dominant behavioral failure — under-calling (too_few_calls 0.42–0.56,
   wrong_tool 0.39–0.56) — persists essentially unchanged through training.

Secondary aggravators: internal-metric inflation repeatedly suggested phantom progress;
Stage 1 was fully saturated (dead rate 1.0, 0 optimizer steps — wasted stage); missing
same-batch baselines made several conclusions unverifiable.

## Q9. What should be fixed first?

In order (details in `IMPLEMENTATION_PLAN.md`, reward specifics in `REWARD_AUDIT.md` §4):

1. **Evaluation credibility (cheap, immediate):** deterministic eval runner that hard-fails
   without a same-batch baseline; official scorer always on; unified metric JSON with
   decoding block; re-run baseline+s3_e1+s3_e2 in one temp0 batch (command ready in
   `STAGE3_AUDIT.md`).
2. **Reward densification (the actual training fix):** continuous per-call credit inside the
   too-few band, argument-binding partial credit inside the wrong-args band, distinct step
   between "no calls" and "one correct call then stop". Validate with a forward-only stage
   probe (target: dead-group rate < 0.5) *before* any full run.
3. **Config hygiene:** repoint `config.yaml` defaults away from legacy dataset B; export
   run manifests (dataset SHA, git commit, decoding) with every run.
4. Only then spend GPU time on new curriculum stages.

## Q10. What should be archived or cleaned later?

Per `CLEANUP_PLAN.md` (proposal only, nothing executed):

- **Archive:** dataset B (`filtered_toolr0_synthetic/`), the pre-v3.1 `curriculum_v3`
  corpus, July-2/3 runs (legacy dataset/reward era; fix the `0260703…` id typo on the way),
  ~20 stale pilot-era reports under `outputs/`, v3-only generator scripts.
- **Stop committing:** adapter safetensors, tokenizer copies, trajectory/prediction JSONLs,
  `data_base/` copies (the current git status is about to commit a full checkpoint —
  recommend adding `.gitignore` rules first).
- **Restructure:** move dataset A from `outputs/` to `data/`; flatten double-nested eval
  batch dirs; unify eval outputs under one `outputs/evals/<batch_id>/` convention (also
  fixes the report generator's path bug that produced "No scored cells found").

---

### One-paragraph executive summary

The v3.1 pipeline is instrumented well enough to answer its own question: MT-GRPO on the
synthetic curriculum does not reliably improve NESTFUL because (a) the banded reward erases
within-group reward variance, so GRPO trains on 12–35% of groups with a tiny lr, and
(b) what it does teach lies outside NESTFUL's failure distribution. No checkpoint beat a
same-batch baseline on any scorer; apparent gains came from comparing across batches,
decodings or the lenient internal metric. The datasets are clean (no leakage, no eval
overlap) but split across three confusable corpora with legacy defaults still live. Fix the
evaluation protocol first (it's cheap), then the reward's within-group discrimination, and
only then buy more GPU hours.
