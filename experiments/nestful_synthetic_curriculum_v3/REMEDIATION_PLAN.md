# REMEDIATION PLAN — from audit to a working pipeline

Date: 2026-07-09. Source: the full read-only audit in `audits/` (MASTER_AUDIT_REPORT.md and
its 12 companions). Every claim below cites an audit finding, not an assumption.

---

## 1. Root causes (from MASTER_AUDIT_REPORT §Q8, FAILURE_MODE_AUDIT §3)

Ranked:

1. **Reward-band quantization starves GRPO.** The v3.1 stepwise reward collapses diverse
   completions into a few flat bands (0.25/0.3, 0.6/0.65, 1.0). 65–88 % of groups have zero
   between-completion reward variance even at 8 generations, T=1.0; only 12–35 % of groups
   produce gradient; KL ≈ 0 throughout. (REWARD_AUDIT §3–4, FAILURE_MODE_AUDIT §1)
2. **Synthetic→NESTFUL transfer gap.** The v3.1 corpus is clean but toy-like (~20 math/string
   tools, `arg_0/arg_1` schemas); zero overlap with NESTFUL also means zero distributional
   anchor. Training reward improves on synthetic stages while dev win never moves.
   (DATASET_AUDIT §0, FAILURE_MODE_AUDIT §2)
3. **Evaluation protocol allowed phantom progress.** Internal win inflates official win by
   ~6–7 pp systematically; two final-eval batches have no official scores; the Stage-3 batch
   has no baseline cell; sampled and temp0 batches were mixed. (METRIC_AUDIT §3, RUN_AUDIT §2)
4. **Dominant behavioral failure is under-calling** (too_few_calls 0.41–0.56, wrong_tool
   0.39–0.56), under-penalized relative to its frequency. (FAILURE_MODE_AUDIT §2)
5. **Repo hygiene erodes provenance**: legacy dataset B is still the config default, three
   confusable corpora share the `epoch_N_Ncall.jsonl` naming, checkpoints are about to be
   committed, report generator has a path bug. (DATASET_AUDIT §3, RUN_AUDIT §4, CLEANUP_PLAN)

## 2. Problem classification

### Evaluation / reporting problems
- Internal `win_rate` confusable with official win (+6–7 pp inflation) — METRIC_AUDIT §3.1
- Final-eval batches without `metrics_official.json` (pre-2026-07-09) — RUN_AUDIT §2
- Stage-3 temp0 batch has no baseline cell → unrankable — STAGE3_AUDIT §2
- Decoding not pinned across batches (sampled vs temp0, ~1 pp shift) — RUN_AUDIT §2
- Three eval populations share metric names (dev-200, NESTFUL call-count slices, full-1861) —
  METRIC_AUDIT §3.3
- Full NESTFUL (1,861) includes the 200 dev tasks used for selection — DATASET_AUDIT §1
- `CHECKPOINT_REEVAL_REPORT` generator looks in the wrong directory ("No scored cells found")
  — RUN_AUDIT §2
- Official win silently skipped on Windows / missing IBM dir — METRIC_AUDIT §3.8
- `strict_gold_trace_pass` in train_log is actually mean reward (misnamed) — METRIC_AUDIT §3.10

### Training / reward problems
- Dead-group rate 0.65–0.88; unique episode rewards per group 1.09–1.33 of 8 — FAILURE_MODE §1
- Band caps (too_few ≤ 0.30, wrong_args ≤ 0.60) flatten within-group differences — REWARD_AUDIT §4
- Under-calling weakly penalized; no gradient step between "no calls" and "call 1 then stop"
- Stage 1 fully saturated (dead rate 1.0, 0 optimizer steps in run 0260703) — RUN_AUDIT §1
- Stage 3 position-artifact rate 0.35–0.41 > 0.2 gate — STAGE3_AUDIT §1
- Motif reward's r_seq is all zeros (no turn-level credit); July-2/3 runs binary-reward era

### Dataset / distribution problems
- v3.1 corpus valid but toy-like vs NESTFUL (tool realism, phrasing, answer types)
- Legacy dataset B: ~4 % unresolved `$var…$` gold answers, 1 null answer, misnamed
  `curriculum_toolr0_all.jsonl` (epoch-6-only), `tools` as string — DATASET_AUDIT §1B
- Three corpora with same file naming; provenance recoverable only via id prefixes
- `nestful_mtgrpo_{minimal,partial}/config.yaml` defaults still point at B — DATASET_AUDIT §D2

### Repo hygiene problems
- Untracked checkpoint (safetensors + tokenizer) staged for commit right now
- Run-id typo `0260703_145219_v3_1`; double-nested eval batch dirs
- ~20 stale pilot-era reports; duplicate v3/v3.1 scripts; empty `curriculum_summary.jsonl`
- No run manifests (git commit / dataset SHA / decoding not recorded with results)

## 3. What MUST be fixed before any new training (P0)

Any GPU-hour spent before these is wasted because its results cannot be trusted or compared:

1. Deterministic eval batch runner with **mandatory same-batch baseline** and official scorer.
2. Unambiguous metric naming + unified metrics JSON (decoding block, dataset SHA).
3. Run manifest (git commit, config hash, seed) attached to every eval output.
4. `.gitignore` so heavy artifacts stop entering git.
5. Guardrail against silently using dataset B (runner-level check; config repoint is P1).

## 4. What can wait

- Reward redesign, stage probe, SFT-warmup+GRPO, NESTFUL-like data — research work (P1),
  gated behind the fixed evaluation protocol and a probe-verified signal check.
- W&B standardization, training-side manifests, multi-GPU launcher formalization (P1–P2).
- Archive moves, renames, doc-tree pruning (P2–P3) — valuable but not blocking.

## 5. Priority table

| Priority | Area | Problem | Proposed fix | Expected impact | Risk | Effort | Acceptance criteria |
|---|---|---|---|---|---|---|---|
| **P0** | repo hygiene | checkpoints/trajectories about to be committed | `.gitignore` for `outputs/**` heavy artifacts | stops repo bloat immediately | none | XS | `git status` no longer lists safetensors/tokenizer/trajectory files |
| **P0** | evaluation unification | comparisons across batches/decodings; missing baselines | `scripts/eval/run_eval_batch.py` + `eval_batch_temp0.sh`: one batch dir, temp0 default, baseline cell required, official scorer verified per cell | every future comparison is valid by construction | low (wraps existing `run.py final_eval`) | S | `--dry-run` prints correct commands; refuses to run without baseline; smoke `--max-tasks 5` produces per-cell dirs + BATCH_REPORT.md |
| **P0** | metric schema | internal win confusable with official win | `scripts/lib/metrics_schema.py` emits `metrics_unified.json` (primary = `official_nestful_win_rate`; internal renamed `internal_final_answer_win`) | headline numbers cannot be the inflated metric | none (additive) | S | unified JSON validates for an existing eval cell (Stage-3 batch) |
| **P0** | run manifests | no provenance on results | `scripts/lib/run_manifest.py` (git commit+dirty, dataset SHA256, config hash, seed, decoding, host) written by eval runner | reproducibility of every number | none | S | manifest present in each batch dir; SHAs match `audits/DATASET_AUDIT.json` |
| **P0** | dataset guardrail | silent legacy-B usage | runner + `check_env.sh` detect `filtered_toolr0_synthetic` in resolved paths and refuse unless `--allow-legacy-dataset` | no accidental wrong-corpus runs | low | XS | running with a B path fails loudly |
| **P0** | official scorer | eval batches without official scores | runner asserts `metrics_official.json` exists per cell, fails batch otherwise (docs cover the Windows/IBM-dir gate) | no more internal-only batches | low | XS | missing official file → nonzero exit + report note |
| **P0** | docs | tribal knowledge only in audits | `README.md` + `docs/{EVALUATION,DATASETS,TRAINING,REWARD,RUNBOOK}.md` | onboarding + fewer protocol mistakes | none | S | docs answer the 11 questions in the task spec |
| **P1** | config defaults | `config.yaml` paths → dataset B | repoint `nestful_mtgrpo_partial/config.yaml` to canonical A/dev paths; add loud comment in minimal (shared file — needs approval) | removes footgun at the source | **medium** (shared with other experiments) | S | v3 dry-run passes; grep shows no B default in partial config |
| **P1** | baseline re-eval | Stage 3 unranked officially | run baseline+s3_e1+s3_e2 in ONE temp0 batch via new runner (GPU pod, user-launched) | settles the Stage-3 question | low | S (pod time) | one batch dir with 3 cells, official win + paired counts |
| **P1** | stage probe | stages trained blind (Stage-1 saturation wasted a run) | `scripts/probe/probe_stage.sh` + Python: forward-only N-task probe, reward-band histogram, predicted dead-group rate | no more dead stages; cheap reward-design iteration | low | M | probe on stage1 reproduces saturation finding without training |
| **P1** | reward redesign | band quantization kills GRPO signal | densify within bands: per-call credit inside too_few band, arg-binding partial credit inside 0.6 band, step between "no calls" and "1 correct call" (new policy name `execution_aware_v3_2_dense`, never edit v3_1 in place) | dead-group rate target < 0.5 at probe time | **medium** (changes training science — experiment, not fix) | M | probe shows ≥2 distinct episode rewards in ≥60 % of groups; RESEARCH_FIX_PLAN E1 criteria |
| **P1** | SFT warmup + GRPO | base model under-calls; GRPO can't bootstrap | reuse existing Stage-2 SFT scripts; add `run_sft_plus_grpo.sh` chaining SFT adapter → GRPO `INIT_FROM=checkpoint` | tests whether initialization unlocks GRPO | medium | M | RESEARCH_FIX_PLAN E3 criteria; no GRPO code changes |
| **P1** | signal filtering | 65–88 % rollout compute wasted on dead groups | curriculum-side: drop tasks probed as saturated/hopeless from stage files (data filter, not trainer change) | more optimizer steps per pod-hour | low | S | filtered stage file + manifest; trainer untouched |
| **P1** | W&B logging | inconsistent projects/runs | one project, run naming = run_id, eval batches as linked runs, unified JSON as artifact | comparable dashboards | low | S | new runs appear under the standard project with tags |
| **P2** | NESTFUL-like synthetic data | toy-tool gap | new generator pass with realistic tool schemas/param names mined from NESTFUL train-side only (no eval questions/traces copied) | attacks transfer gap | **medium** (contamination discipline required) | L | RESEARCH_FIX_PLAN E5 criteria; overlap check = 0 vs eval |
| **P2** | multi-GPU runner | env-var choreography (`ROLLOUT_DP_GPUS` etc.) fragile | single `scripts/training/run_grpo.sh` validating topology up front | fewer mid-run crashes | medium | M | dry-run prints resolved topology; bad topology fails fast |
| **P2** | training manifests | train runs lack manifests | wire `run_manifest.py` into `run_curriculum.sh` launch path | full provenance | low | S | manifest in each `OUTPUT_ROOT` |
| **P3** | archive | legacy corpora/runs/reports clutter | `git mv` per CLEANUP_PLAN §2 (dataset B, curriculum_v3, July-2/3 runs, stale reports) + archive README | navigable repo | medium (path breakage on pod) | M | CI/dry-runs pass after moves; archive README maps old→new |
| **P3** | naming | run-id typo, double-nested batch dirs, `epoch_N` alias | renames per CLEANUP_PLAN §2 | less confusion | low | S | no double-nesting; ids sort correctly |

## 6. Safest order of implementation

1. **P0 wave (this change-set):** `.gitignore` → eval runner + schema + manifest + guardrail →
   docs. All additive; nothing existing modified except adding new files.
2. **P1 gate 1 (needs one pod session, user-launched):** same-batch baseline+s3 re-eval with
   the new runner. This is the first real use of the P0 infra.
3. **P1 gate 2 (approval needed):** config repoint (shared files), stage probe, W&B standard.
4. **P1 research (only after probe exists):** reward v3_2 experiment → probe → tiny GRPO run →
   full protocol per RESEARCH_FIX_PLAN. SFT+GRPO chain in parallel (independent infra).
5. **P2:** NESTFUL-like data generation; multi-GPU runner; training manifests.
6. **P3:** archive moves and renames, last — after nothing active depends on old paths.

## 7. What should NOT be touched yet

- `experiments/nestful_mtgrpo_minimal/grpo_train.py`, `group_stats.py`, `rollout.py`,
  `executor.py`, `metrics.py`, `reward.py` — the trainer and scorers are audited and working;
  all P0 work wraps them.
- `lib/reward_v3_1.py` — keep frozen as the audited baseline policy; new reward work goes in
  a NEW module (`reward_v3_2_dense.py`) selected by config, so A/B remains possible.
- `outputs/runs/**` — no renames/moves until P3 (pod resume paths, audit reproducibility).
- `data/NESTFUL-main/**` — official benchmark code/data, never modified.
- The `audits/` folder — historical record; new analysis goes to new files.
- Scientific framing: no claim of improvement anywhere until a same-batch official win with
  CI and paired counts exists (METRIC_STANDARD_PROPOSAL rules).

## 8. Acceptance criteria per phase

| Phase | Acceptance |
|---|---|
| P0 infra | `bash -n` + `python -m compileall` clean; eval runner `--dry-run` prints resolved per-cell commands incl. baseline; runner exits nonzero without baseline cell or with legacy-B path; unified metrics JSON produced for an existing cell; `.gitignore` verified with `git status`; docs present and answer the required questions |
| P1 re-eval | one temp0 batch dir with baseline/s3_e1/s3_e2 cells, each with `metrics_official.json` + `metrics_unified.json` + manifest; BATCH_REPORT with paired counts and CI |
| P1 probe/reward | probe runs on ≤100 tasks without training; reward v3_2 probe shows dead-group rate < 0.5 before any GRPO time is spent |
| P1 SFT+GRPO | chain runs end-to-end on a smoke subset; evaluation only through the batch runner |
| P2 data | new corpus passes the dataset audit script with 0 overlap vs NESTFUL eval; tool-schema realism metrics reported |
| P3 archive | all launchers dry-run green after moves; archive README complete |
