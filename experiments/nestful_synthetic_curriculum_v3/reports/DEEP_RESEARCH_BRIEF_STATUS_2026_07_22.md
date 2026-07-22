# Tool-R0 / NESTFUL Synthetic Curriculum — Deep Research Brief

**Date:** 2026-07-22  
**Purpose:** Self-contained status brief for external deep research on *how to continue*.  
**Project:** Improve `Qwen/Qwen3-4B-Instruct-2507` on IBM NESTFUL nested tool-calling via synthetic curriculum + MT-GRPO (multi-turn GRPO), without contaminating eval.  
**Headline metric (only claimable metric):** `official_nestful_win_rate` at temperature 0 on `nestful_test` (n=1661), same-batch baseline, paired gained/regressed counts. Internal train “win” (reward ≥ 0.99) is diagnostic only and systematically ~5–7 pp higher than official win.

---

## 0. One-paragraph verdict

After fixing evaluation credibility and densifying the reward, a full two-phase GRPO run on NESTFUL-like synthetic Stage-2/3 data moved official Nestful win only **+0.42 pp** (53.52% → 53.94%), which is **not statistically significant**. Format/parse is largely resolved; failures are semantic (wrong tool / wrong values / executable-wrong-final). Training still suffers sparse GRPO signal on Stage-2 (78% dead groups). A subsequent pure-Stage-3 overnight train completed on W&B with better train dynamics (dead groups 28%→15%), but **official Nestful transfer for those checkpoints was never measured** because the RunPod pod died mid-eval. Smoke pure-Stage-3 (8 tasks) slightly *hurt* full test (−0.42 pp). We are at a decision gate: stop “more epochs of the same Stage-3 GRPO” unless overnight E2 proves ≥~1 pp gain; otherwise change objective (credit assignment / reward / data mix / init), not scale the current recipe.

---

## 1. Goal and constraints

### Goal
Raise official NESTFUL win rate of a 4B instruct model on nested multi-tool trajectories (2–6+ calls, variable refs `$varN.field$`, IBM executable functions), using synthetic curriculum that is:

- executable and gold-replayable,
- contamination-free vs Nestful dev/test,
- trainable with GRPO on 4× GPU RunPod (vLLM DP rollouts + QLoRA learner).

### Hard rules already enforced
1. Canonical headline = official Nestful win @ temp0 with same-batch baseline.  
2. No Nestful eval questions/gold traces in training.  
3. Executor mode for train = `synthetic`; reward = `execution_aware_v3_2_dense`; registry **v5.0.2**.  
4. No `gold_replay` shortcuts in training asserts.

### Base setup (current production recipe)
| Item | Value |
|------|-------|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Method | MT-GRPO / QLoRA, `num_generations=8`, LR `3e-7`, `kl_beta=0.15` |
| Train T / top_p | 1.0 / 0.95 |
| Eval | T=0, top_p=1, 1 rollout, IBM official scorer |
| GPUs | 0 = HF learner; 1–3 = vLLM DP; eval TP=4 after unload |
| Reward | `lib/reward_v3_2_dense.py` (dense within bands) |
| Prompt | ReAct + `<tool_call_answer>` single-call gate |

---

## 2. Timeline of what was built (chronological)

### Phase A — Early pilots + audit (≈ July 2–9, 2026)
- Built synthetic curriculum **v3.1**: prefix-decomposed stages (Stage 1–4), ~800 rows/stage, 100% gold replay, toy math/string tools.
- Ran Stage 1–3 GRPO pilots. **Audit conclusion:** no checkpoint beat same-batch baseline on official Nestful; many prior “gains” were invalid (missing baseline, internal metric inflation, cross-batch compare).
- Root causes ranked in audit:
  1. Banded reward → dead GRPO groups (65–88%).
  2. Synthetic→Nestful distributional transfer gap (toy tools vs real APIs).
  3. Broken eval protocol (phantom progress).
  4. Dominant failure: under-calling + wrong tool.
- P0 remediation: eval batch runner with mandatory baseline, manifests, dataset-B guardrails, hygiene.

### Phase B — Research plan (falsifiable experiments E0–E5b)
Documented in `RESEARCH_FIX_PLAN.md`:
- **E0** stage probe (dead-group instrument)
- **E1** reward densification (`v3_2_dense`)
- **E2** signal-positive filtering (drop dead tasks)
- **E3** SFT warmup → GRPO
- **E4** under-call-targeted weighting
- **E5** Nestful-like synthetic v4 (realistic schemas)
- **E5b** Agentic Autodata-style generation via OpenRouter (weak-fail / strong-pass)

Decision flow intended: calibrate probe → densify reward → if still no transfer, SFT or Nestful-like data; don’t scale dead GRPO.

### Phase C — Nestful-like data + agentic generation (≈ July 9–16)
- Implemented deterministic Nestful-like generator path + **agentic OpenRouter** pipeline (challenger / weak / strong / judge, contamination gates, quality scoring). Offline mock smoke OK.
- Real agentic Stage-3 pilots on pod:
  - Broken registry → 0 accepts.
  - `win=1` gate too hard → ~2.7% accept.
  - `win=0` + GRPO-signal gate → **~10% accept** (target 5–10%), but dominant reject = `all_same_reward` (model rollouts too homogeneous on 3-call).
  - Some accepted rows had micro reward_range &lt; 0.01 (gate later hardened).
- Stage-3 syntax audit vs Nestful: **NO_MISMATCH** (refs already Tool-R0 canonical).

### Phase D — Training-ready v5 + two-phase continuous GRPO (≈ July 18)
- Materialized **training_ready_v5**:
  - Phase 1: **429** Stage-2 (2-call)
  - Phase 2: **466** = 326 Stage-3 (3-call) + 140 Stage-2 replay
- Built continuous in-process two-phase session (shared AdamW + monotonic `global_step`, deferred C1/C2 eval so EVAL_TP=4 doesn’t kill optimizer state).
- Full run: `two_phase_20260718_192902`.

### Phase E — Analysis + pure Stage-3 overnight (≈ July 19–20)
- Root-cause + format audits on C0/C1/C2.
- Hypothesis shift: skip Stage-2 (too many dead groups); train **pure Stage-3** 2 epochs on 326 tasks.
- Smoke (8 tasks) + overnight train completed on W&B; **post-train Nestful eval for overnight lost** (pod death / vLLM teardown fragility). Eval unload fix later landed in code.

---

## 3. What worked

### Infrastructure / science hygiene (clear wins)
- **Official Nestful scoring path** with manifests, same-batch baseline discipline, paired analysis.
- **Continuous two-phase GRPO** (same process, same optimizer, atomic C1/C2 checkpoints, rollout worker sync) — engineering works; optimizer continuity verified (`optimizer_id` unchanged, `global_step` 0→24→105).
- **Dense reward `v3_2_dense`:** Phase-2 dead-group rate dropped to **31%** (Stage-3-only slice **17%**); GRPO ordering violations = 0 (reward ordering coherent).
- **Format largely resolved:** parse_fail 74→63 on test; `no_tool_call_rate` during pure-S3 train ≈ 0.04%; reference syntax mismatch ruled out.
- **Agentic data loop** is operational at ~10% accept for Stage-3 with cost controls / contamination gates.
- **W&B logging** sufficient to recover overnight train metrics after local pod loss.
- **Eval teardown fix** (unload learner + util cap) identified after overnight eval crash.

### Partial scientific signal (real but small / localized)
- Two-phase C2 vs C0:
  - **6+ call bucket:** +5.3 pp win
  - **long_chain motif:** +3.2 pp
  - Wrong-arg-value failures: −16 tasks
  - Parse/format errors: −11
- Pure Stage-3 train dynamics improved vs Phase-1: dead groups **28%→15%** across epochs; epoch train win_rate ~0.28–0.29 (synthetic threshold), mean reward ~0.63.
- Curriculum coverage diagnosis is clear: Nestful test has large 4–6+ mass; train was mostly 2–3 call.

---

## 4. What did not work (or underperformed)

### Headline outcome failures
| Experiment | Result | Interpretation |
|------------|--------|----------------|
| Early v3.1 Stage 1–3 GRPO | No same-batch official win gain | Reward sparsity + toy transfer |
| Stage 1 training | Dead rate ~1.0, ~0 useful steps | Saturated / useless stage |
| Two-phase v5 GRPO full | +0.42 pp C2−C0; McNemar n.s. | Not meaningful transfer |
| Function F1 | −1.0 pp C2 vs C0 | Mild tool-choice regression |
| 4-call / 5-call buckets | −3.6 / −2.0 pp | Mid-length regression |
| 2-call bucket after Phase 2 | −0.7 pp | Partial forgetting despite replay |
| Pure-S3 smoke E2 @ full test | **−0.42 pp** vs C0 | Stage3-only can hurt if underpowered/unstable |
| Pure-S3 overnight Nestful E1/E2 | **Unknown** (eval crashed / not logged) | Missing critical decision data |

### Mechanisms that failed or remain blocked
1. **Dead groups still dominate Stage-2** (78% Phase 1) — most GRPO steps were null. Stage-2 replay in Phase 2 still **63% dead**.
2. **Train–eval reward gap:** synthetic dense success ≠ IBM official win (final_answer_pass ~59% vs official ~54%).
3. **`executable_wrong_result` increased** (+15) even as wrong-arg-value fell — model learns executable but wrong trajectories.
4. **Agentic Stage-3 bottleneck:** rejects mostly `all_same_reward` — generation finds tasks where the policy is homogeneous, not discriminative.
5. **Infra fragility:** train→eval TP=4 after learner still on GPU caused overnight loss of Nestful scores; local checkpoints may be gone with pod.
6. **Not the bottleneck anymore:** raw format / reference syntax / “add format reward” — audits say **don’t**.

### Explicitly ruled-out or low-priority next steps
- More format reward / format SFT as primary lever.
- Claiming progress from internal train win_rate alone.
- More Stage-1.
- Interpreting official `parser_errors=0` as format solved (official scorer sees pre-extracted JSON, not raw ReAct).

---

## 5. Best measured numbers (Nestful test, n=1661)

### Two-phase run `two_phase_20260718_192902`

| Arm | Win | F1 func | F1 param | Notes |
|-----|----:|--------:|---------:|-------|
| C0 baseline | 0.5352 | 0.898 | 0.439 | Base model |
| C1 (after Stage2) | 0.5370 | 0.889 | 0.442 | +0.18 pp |
| C2 (after Stage3+replay) | 0.5394 | 0.888 | 0.442 | +0.42 pp vs C0 |

Paired C2 vs C0: gained 88 / lost 81 / net +7; McNemar p≈0.81–0.88 (n.s.).

Train diagnostics:
- Phase1: dead_group **78.3%**, mean reward 0.533, train win 0.031  
- Phase2: dead_group **31.0%**, mean reward 0.588, train win 0.197  
- Phase2 Stage3-only dead **17.2%**; Stage2-replay dead **62.9%**

Failure mix among non-wins: ~72% semantic-dominant.

### Pure Stage-3 overnight `pure_stage3_2ep_20260719_221918` (train only)

| Epoch | Dead-group | Mean reward | Epoch win_rate |
|------:|-----------:|------------:|---------------:|
| E1 | 27.9% | 0.626 | 0.283 |
| E2 | 15.0% | 0.634 | 0.294 |

- `too_few_calls_rate` still ~14–15% vs gold 3-call during train.  
- Nestful E1/E2 official scores: **missing**.

### Pure Stage-3 smoke (underpowered)

| Eval | Win |
|------|----:|
| C0 test | 0.5382 |
| E2 test | 0.5340 (−0.42 pp) |

---

## 6. Current state (as of 2026-07-22)

### Known
- Evaluation science is trustworthy enough to say current two-phase recipe **does not** deliver meaningful Nestful gains.
- Format is not the main problem; semantics + credit assignment + transfer are.
- Pure Stage-3 improves *train-time* GRPO signal vs Stage-2, but Nestful transfer for the overnight model is **unmeasured**.
- RunPod overnight artifacts may be lost; W&B has train curves; need credit + re-eval or retrain with artifact sync.

### Unknown / blocking
1. Does overnight pure-S3 E2 beat C0 by ≥~1 pp on nestful_test?  
2. Does it help 3-call and 6+ without killing 2-call?  
3. Is the binding constraint now **credit assignment on long chains**, **distribution mismatch**, **model capability floor (4B)**, or **reward–official gap**?

### Open decision gate (from WANDB_STATUS_OVERVIEW)
After recovering overnight Nestful E2 (or retrain):
- if win ≤ C0 + ~1 pp → **stop** “more Stage3 epochs”; change objective (credit / reward / data mix / init);
- if win up on 3-call and 6+ without killing 2-call → continue that lane carefully.

---

## 7. Hypotheses still in play (for deep research)

Ranked by how much evidence currently supports them:

### H1 — Sparse / misaligned credit (strong evidence)
Even with denser rewards, many groups are flat; success credit may not distinguish “almost correct Nestful trajectories.” Mid-length (4–5) regressions + rising `executable_wrong_result` suggest wrong local optima.

**Research ask:** Better process/outcome credit for multi-turn tool RL (step-level, trajectory shaping, outcome-only with verifier, GRPO variants, filtering dead groups online).

### H2 — Distribution / transfer gap (strong evidence historically; partially addressed)
Toy v3.1 clearly mismatched Nestful. v5 Nestful-like registry + agentic data may still miss Nestful’s long-tail (4–6+ calls, API heterogeneity, answer equality strictness).

**Research ask:** What synthetic→real tool-use transfer recipes actually work at 4B? Schema matching vs on-policy Nestful-train (careful leakage) vs teacher distillation vs retrieval of tool docs.

### H3 — Init / exploration insufficiency (moderate)
Train win on Phase1 was 3%; under-calling persists (~14% too_few on pure-S3 train; ~60% under-calling on Nestful eval historically). SFT warmup (E3) was planned but **not shown to win Nestful yet** in this brief’s measured headline runs.

**Research ask:** SFT-on-gold-continuations then GRPO vs pure GRPO; whether imitation is required before RL on sparse tool tasks.

### H4 — Data mix / curriculum design (moderate)
Phase2 replay insufficient to protect 2-call; 4–5 call mass absent from train; Stage1 useless.

**Research ask:** Optimal call-length mix; whether to skip Stage2 entirely; online hard-example mining; longer-chain synthetic prefixes.

### H5 — Capability floor (open)
4B may be near Nestful ceiling for this prompt/executor setup (~53.5% base). Gains of &lt;1 pp may be noise around a capability wall.

**Research ask:** Expected headroom for 4B on Nestful; whether larger teacher / stronger base is prerequisite.

### H6 — Infra / measurement (mostly fixed, residual risk)
Missing overnight eval is an ops failure, not a scientific negative on pure-S3. Must not confuse “unknown” with “failed.”

---

## 8. Candidate next experiments (already proposed internally)

From `C0_C1_C2_ROOT_CAUSE_ANALYSIS.md` + WANDB overview — each should stay falsifiable:

1. **Recover / re-run pure-S3 Nestful eval** (cheap if checkpoints exist; else retrain + immediate artifact download). Decision gate.  
2. **Adaptive dead-group filtering** during Phase1/Stage2 (or skip Stage2). Expect dead_group &lt;0.4 and Nestful +≥1 pp or abort.  
3. **Increase Stage2 replay** in Phase2 (protect 2-call).  
4. **Terminal-outcome reward ablation** (widen fully_correct vs executable_wrong_final gap) to close official-win alignment.  
5. **Stage 4/5 prefix curriculum** (200× 4–5 call) for mid/long Nestful buckets.  
6. **On-policy Nestful-dev mini-loop** (50 held-out train-side Nestful tasks, IBM executor) as transfer probe — watch for overfit.  
7. **SFT warmup then GRPO (E3)** — still under-tested as headline Nestful experiment.  
8. **Do not** add format reward as primary lever.

---

## 9. What deep research should answer

Please propose a **ranked continuation plan** for the next 1–2 GPU-weeks that:

1. States whether the overnight pure-S3 missing eval is the single highest-EV next action.  
2. Given a pessimistic prior that pure-S3 also lands ≤ +1 pp, recommends the **best alternative lever** among: reward/credit redesign, SFT→GRPO, longer-chain data, Nestful on-policy fine-tune (leak-safe), agentic data scale-up, or model scale / teacher.  
3. Specifies success/abort criteria in official Nestful pp and which secondary diagnostics (dead_group, too_few, F1_func, per-bucket win) must move together.  
4. Warns against known invalid comparisons (internal win, cross-batch, no baseline).  
5. Accounts for 4×GPU RunPod ops: always sync `checkpoints/`, `eval/`, `console.log` before teardown.

### Key references inside the repo
- `reports/WANDB_STATUS_OVERVIEW.md` — latest W&B recovery  
- `reports/C0_C1_C2_ROOT_CAUSE_ANALYSIS.md` — full paired Nestful analysis  
- `reports/FORMAT_STATUS_C0_C1_C2.md` — format vs semantics verdict  
- `audits/MASTER_AUDIT_REPORT.md` — early negative result + root causes  
- `RESEARCH_FIX_PLAN.md` — E0–E5b hypotheses  
- `scripts/training/RUNBOOK_TWO_PHASE_GRPO.md` — two-phase ops  
- `scripts/training/RUNBOOK_PURE_STAGE3_TWO_EPOCH.md` — pure-S3 ops  

### W&B projects
- `sasinka-martin/nestful-v5-curriculum` (two-phase)  
- `sasinka-martin/nestful_v5_pure_stage3` (pure Stage-3)

---

## 10. Non-goals / do-not-repeat

- Do not optimize for internal train win_rate as headline.  
- Do not spend cycles on format reward first.  
- Do not claim Stage-1 curriculum value without new evidence.  
- Do not interpret missing overnight Nestful scores as proof that pure-S3 failed.  
- Do not train without artifact sync / checkpoint export on ephemeral pods.

---

*End of brief.*
)
