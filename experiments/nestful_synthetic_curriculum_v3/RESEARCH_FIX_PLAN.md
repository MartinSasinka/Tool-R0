# RESEARCH FIX PLAN

Date: 2026-07-09. These are **experiments, not fixes** — none of them is claimed to improve
NESTFUL win rate. Each is falsifiable, cheap to abort, and measured under the fixed protocol
(same-batch temp0 official win via `scripts/eval/run_eval_batch.py`, paired counts, manifest).

Shared measurement standard for every experiment below:
- **Primary:** `official_nestful_win_rate` on the full NESTFUL set, temp0, baseline in the
  same batch, with binomial CI and paired gained/regressed counts.
- **Selection:** dev-200 only, never reported as headline.
- **Training-signal diagnostics:** probe-based dead-group rate, unique rewards/group,
  too_few_calls rate, avg_predicted_calls, KL.
- **Guardrails:** no NESTFUL eval questions/gold traces in any training data (checked with
  `audits/tools/dataset_audit.py` overlap section); dataset SHAs in run manifests.

---

## E0 — Stage probe as an instrument (prerequisite, not an experiment)

Forward-only rollout of `num_generations` completions on N≈50–100 stage tasks + reward
scoring, reporting reward-band histogram and predicted dead-group rate.
**Calibration:** must reproduce audited values (Stage 1 ≈ 1.0 dead; Stage 2 ≈ 0.65–0.88)
before it is trusted to gate anything else.

## E1 — Reward densification (`execution_aware_v3_2_dense`)

- **Hypothesis:** GRPO fails mainly because the v3.1 band structure gives identical rewards to
  behaviorally different completions; densifying within-band credit will raise within-group
  variance and produce optimizer steps that track task success.
- **Mechanism:** group-relative advantages need reward differences *within* a group; today
  65–88 % of groups are flat (FAILURE_MODE_AUDIT §1). Adding graded credit — per-correct-call
  fraction inside the too_few band, per-correct-argument fraction inside the wrong-args band,
  and a nonzero step between "no calls" and "one correct call" — creates orderings among
  completions that currently tie.
- **Dataset:** canonical v3.1 Stage 2 (then Stage 3), unchanged.
- **Reward:** new module `lib/reward_v3_2_dense.py`; v3_1 kept as control. Band *ceilings*
  preserved (a too-few completion can never outscore a complete correct one), only within-band
  resolution increases — this bounds the behavior change.
- **Initialization:** base model (same as pilots) for the primary comparison.
- **Metrics:** probe dead-group rate before/after (gate); then a short GRPO run (1 stage,
  same budget as the July-8 pilot) → same-batch official eval vs base.
- **Success criteria:** (gate) probe dead-group rate < 0.5 and mean unique rewards/group ≥ 2;
  (outcome) paired net gain > 0 with CI excluding a ≥1 pp regression, plus reduced
  too_few_calls rate on eval.
- **Failure interpretation:** if the probe gate passes but official win still doesn't move,
  reward sparsity was not the binding constraint — the transfer gap (E5) or capability floor
  (E3) dominates; stop iterating on reward bands.

## E2 — Signal-positive filtering

- **Hypothesis:** concentrating rollout budget on tasks that produce reward variance
  increases effective optimizer steps per pod-hour without changing the objective.
- **Mechanism:** dead groups contribute zero gradient but full rollout cost; filtering
  probed-dead tasks (saturated or hopeless under the current policy) reallocates compute.
- **Dataset:** probe-filtered copy of Stage 2/3 (filter manifest records what was dropped and
  why); original files untouched.
- **Reward:** whatever E1 selects (works with v3_1 too).
- **Initialization:** base model.
- **Metrics:** optimizer steps per wall-hour, dead-group rate during training, then the
  standard same-batch eval.
- **Success criteria:** ≥ 2× non-dead group throughput at equal wall time, no eval regression.
- **Failure interpretation:** if filtered training degrades eval, the "dead" tasks were
  contributing implicit regularization/coverage — revert to full stage files and treat
  filtering as a diagnostics-only tool. Beware selection drift: re-probe periodically since
  deadness is policy-dependent.

## E3 — SFT warmup, then GRPO from the SFT adapter

- **Hypothesis:** the base model under-calls so severely (too_few_calls 0.41–0.56) that GRPO
  rarely samples complete traces to reinforce; SFT on gold continuations moves the policy into
  a region where rollout groups contain successes, and GRPO can then optimize.
- **Mechanism:** RL from a policy with near-zero success rate is signal-starved regardless of
  reward shape; imitation first raises the success base rate (standard RLHF-style pipeline).
- **Dataset:** Stage-2 continuation SFT view (already built: 640/160, exact derived view of
  the GRPO Stage-2 file) for warmup; canonical stages for GRPO.
- **Reward:** unchanged during SFT (none); v3_1 or v3_2 during GRPO.
- **Initialization:** GRPO starts from the SFT adapter (`run_sft_plus_grpo.sh`).
- **Metrics:** (a) SFT alone vs base — continuation-conditioned and free-ReAct evals plus
  same-batch NESTFUL; (b) SFT+GRPO vs SFT alone (the incremental-GRPO question); probe
  dead-group rate at GRPO start.
- **Success criteria:** SFT reduces too_few_calls on eval; probe-at-init dead-group rate
  drops vs base; SFT+GRPO > SFT alone on paired official win.
- **Failure interpretation:** if SFT helps but GRPO adds nothing on top, GRPO's marginal value
  at this scale/reward is nil — the paper story becomes "SFT on synthetic continuations";
  if SFT itself doesn't transfer to NESTFUL, the corpus is too toy-like → E5 first.

## E4 — Under-call-targeted curriculum weighting

- **Hypothesis:** oversampling motifs/positions where under-calling concentrates (call-2+
  continuation failures) attacks the dominant failure mode faster than uniform sampling.
- **Mechanism:** loss/rollout exposure proportional to observed failure mass.
- **Dataset:** canonical stages with a sampling-weight sidecar (files unchanged).
- **Reward:** unchanged. **Initialization:** best of E1/E3.
- **Metrics:** too_few_calls and call-position failure curves on dev-200 during training;
  standard same-batch final eval.
- **Success criteria:** too_few_calls rate drops ≥ 10 pp on eval without wrong_tool rising.
- **Failure interpretation:** under-calling is a decoding/format behavior, not a data-mix
  behavior → address via prompt/stop-criterion work instead of data.

## E5 — NESTFUL-like synthetic curriculum (v4)

- **Hypothesis:** the transfer gap is distributional: ~20 toy math/string tools with
  `arg_0/arg_1` schemas teach dependency-passing that doesn't map onto NESTFUL's realistic
  APIs; a corpus with realistic tool schemas will transfer.
- **Mechanism:** matching tool-schema statistics (name/param realism, arity, type mix,
  nesting patterns) while keeping synthetic gold traces executable and verifiable.
- **Dataset:** new generator (v4). Tool schemas mined from NESTFUL **train-side executable
  specs only**; **no eval questions, no gold traces, no paraphrases of eval items**. Overlap
  gate (question hash, trace hash, id) must be exactly 0 vs dev and test; generator + seed +
  manifest committed.
- **Reward:** best policy from E1. **Initialization:** best pipeline from E3.
- **Metrics:** standard same-batch eval; plus corpus-level distribution distance (tool arity,
  arg-type mix, calls-per-task) between v4 and NESTFUL vs the v3.1 baseline distance.
- **Success criteria:** v4-trained model beats both base and v3.1-trained model on paired
  official win in one batch.
- **Failure interpretation:** if v4 matches distributions but still doesn't transfer, the gap
  is capability (model scale / reasoning), not data — a scaling or teacher-distillation
  question, and the curriculum line of work should be reported as negative.

## E5b — Agentic (Autodata-style) NESTFUL-like curriculum via OpenRouter

- **Hypothesis:** LLM-composed tasks filtered by weak-fail/strong-pass are harder
  and more discriminative than template-generated v4, concentrating training
  signal exactly where the (weak-solver-like) policy fails — per Autodata /
  Agentic Self-Instruct (arXiv:2606.25996), where the agentic loop lowered
  weak-solver scores by 22 pts while raising strong-solver scores.
- **Mechanism:** challenger LLM proposes tasks over the same executable tool
  registry as deterministic v4; a deterministic executor computes gold answers
  (LLM answers never trusted); acceptance requires weak solver ≤ 0.50,
  strong solver ≥ 0.80, gap ≥ 0.25; the challenger recipe is revised from
  batch-level rejection analysis.
- **Dataset:** `data/curriculum_v4_nestful_like_agentic_openrouter/` (mirrors
  deterministic v4 per-stage counts). Contamination: challenger never sees
  NESTFUL items; `aggregate_style_only` tool policy; zero-overlap gate per
  candidate + final corpus. Pipeline: `docs/AGENTIC_DATA_GENERATION.md`.
- **Reward:** best policy from E1. **Initialization:** best pipeline from E3.
- **Metrics:** `score_dataset_quality.py` (validity / contamination /
  distribution distance / solver gap), stage probe vs v3.1 stage2 (dead-group
  rate, unique rewards/group), then standard same-batch eval after training.
- **Success criteria:** probe signal better than v3.1 AND (after training)
  paired official win over both base and the v3.1-trained model in one batch;
  weak-fail/strong-pass examples must dominate the accepted set.
- **Failure interpretation:** if solver-gap filtering accepts too few examples
  (acceptance rate < 2 %), the challenger/model pairing is miscalibrated —
  revise recipe or models before spending more budget. If the probe is good
  but transfer still fails, same interpretation as E5 (capability gap, not
  data), and the LLM-generation cost is not justified.

## Decision flow

```
E0 probe calibrated
  └─ E1 reward: probe gate passes? ──no──► skip GRPO; go E3/E5
        └─ yes ► short GRPO ► same-batch eval
E3 SFT: transfer to NESTFUL? ──no──► E5 (data first)
        └─ yes ► SFT+GRPO increment test (with E1 reward)
E2/E4 are efficiency/targeting add-ons, run only inside a lane that already shows signal
E5 is the fallback if both reward- and init-side lanes stall
```

Reporting rule for all of the above: results go in a batch report generated by the eval
runner; "improvement" may be claimed only for paired same-batch official win with CI, never
from internal metrics or cross-batch deltas.
