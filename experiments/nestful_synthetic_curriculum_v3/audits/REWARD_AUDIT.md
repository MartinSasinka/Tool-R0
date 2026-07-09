# REWARD AUDIT — MT-GRPO training rewards

Date: 2026-07-09 · Read-only audit. Files:
`experiments/nestful_synthetic_curriculum_v3/lib/reward_v3_1.py` (current),
`lib/reward_motif.py` (v2.1 ablation), `experiments/nestful_mtgrpo_minimal/reward.py`
(strict), dispatch in `experiments/nestful_synthetic_curriculum_v3/run.py`
(`_hook_select_train_reward`), trainer coupling in
`experiments/nestful_mtgrpo_minimal/grpo_train.py`.

## 1. Reward policies

### `execution_aware_v3_1_stepwise` (canonical for all v3.1 runs)

`lib/reward_v3_1.py::execution_aware_v3_1_stepwise` → adapter
`episode_turn_reward_seq` supplies BOTH `episode_reward` and per-turn `r_seq` from the same
definition (fixing an earlier bug where r_seq silently came from execution_aware_v2).

Weighted components (multi-call stages): format 0.10, tool_sequence_match 0.20,
argument_value_match 0.15, valid_references 0.15, dependency_use 0.10,
executable_trajectory 0.10, expected_num_calls 0.10, final_answer_match 0.10.
Stage 1 uses a 5-component variant.

**Bands (hard caps / floor), in evaluation order:**

| condition | reward | notes |
|---|---|---|
| parse error | **0.0** | hard cap |
| clipped completion | **0.0** | also masked from update in trainer |
| no tool call | **0.0** | |
| premature final on non-terminal (prefix) task | **0.0** | non-terminal tasks reward *not* answering |
| invalid variable reference | ≤ **0.10** | |
| too few calls | ≤ **0.30** | dominant observed band (0.25/0.3 clusters in logs) |
| wrong tool (any gold position mismatched) | ≤ **0.35** | |
| correct tools, wrong args | ≤ **0.60** | second dominant band (0.6/0.65) |
| too many calls | ≤ **0.70** | |
| executable, right trace, wrong final answer | ≤ **0.75** | |
| fully correct | ≥ **0.90** (floor) | logs show mass at 1.0 |

**Fallback / error handling:** stage must be inferable (task metadata → `train_stage` arg →
`TRAIN_STAGE` env → num_calls) else `RewardError` is raised — no silent default. Predicate
computation failure returns 0.0 **with** a `predicates_error` diagnostic and a loud log line
(never fakes a band). Dispatch fallback to the strict reward is env-gated:
`ALLOW_STRICT_REWARD_FALLBACK=0` in the launcher, and `train_summary.json` logs
`reward_policy_configured/resolved/fallback_used` — all audited runs show
`fallback_used=false` with `lib.reward_v3_1.episode_turn_reward_seq` resolved. The stage
gate `resolved_reward_matches_configured` re-checks this per stage.

**Execution semantics:** uses real trajectory execution (executor mode `full`; predicates from
`nestful_core.rewards`: executable_fraction, valid_references_fraction, final-answer pass).
Argument matching is value-normalized (`_values_match`: 1e-6 relative numeric tolerance,
stripped strings, element-wise containers); gold reference args accept any well-formed
`$var…$` reference (label-agnostic), literal-instead-of-reference gets half credit.

### `execution_aware_v2_1_motif` (July-2 run, v3 dataset)

`lib/reward_motif.py`; selected when `REWARD_POLICY=execution_aware_v2_1_motif` or curriculum
version v3. Weighted sum (final_answer 0.30, executable 0.20, valid_refs 0.15,
motif_consistency 0.15, completeness 0.10, gold_progress 0.10) with caps
(parse/clipped/no_tool/terminal-before-first-tool → 0.0; not_executable / invalid_ref → 0.25;
too_few_without_final → 0.20; final-pass-but-low-motif → 0.75) and floor 0.85 for
final_pass ∧ executable ∧ motif ≥ 0.75. **Its `r_seq` is all zeros** (episode-level credit
only, `reward_motif.py:182–189`) — no per-turn contrast for MT-GRPO. Historical caveat: in
the July 2–3 runs the logged episode rewards were **strictly binary {0,1}** (see RUN_AUDIT) —
despite the graded design, tasks saturated into the 0.0 caps or the correct band; those runs
cannot validate the band design.

### `partial_gold_trace` (default of `nestful_mtgrpo_partial`, not used by v3 runs)

`nestful_mtgrpo_partial/partial_reward.py:172–235`. Graded gold-trace reward: per-turn
`0.4·name_ok + 0.3·keys_ok + 0.3·exec_ok` (gated cumulatively), episode
`0.7·mean(turn_scores) + 0.3·final_answer_pass`, clip [0,1]; extra calls ignored by default
(`length_penalty=0.0`). This is the fall-through policy of `_select_train_reward` when the v3
hook does not match — relevant because v3 runs launch through the partial `config.yaml`; the
hook (`_hook_select_train_reward`) intercepted correctly in all audited runs (verified via
`reward_policy_resolved` in every train_summary).

### `strict_gold_trace_reward` (minimal experiment default / fallback)

`reward.py:51-160`: binary. 1.0 iff call count == gold count AND every position matches tool
name + argument-key set + per-turn observation (vs `compute_gold_observations` replay) AND no
parse fail AND no clipping AND final answer matches. Everything else 0.0. `too_many_turns`,
`answer_correct_wrong_path` recorded as diagnostics only. Used for `strict_gold_trace_pass`
eval metric everywhere; as a *training* reward only outside the v3 wrapper.

### Dispatch / fallback matrix (verified in code)

| scenario | behavior |
|---|---|
| unknown `train_policy` | `ValueError` — no silent strict fallback (`vllm_dp_pool.py:146–153`) |
| unknown + `ALLOW_STRICT_REWARD_FALLBACK=1` | warning + strict fallback, `fallback_used=True` logged |
| graded policy resolves to strict module without the flag | `RuntimeError` abort (`grpo_train.py:88–94`) |
| graded reward emits only {0,1} over first 50 groups | early abort (`grpo_train.py:761–767`) |
| dead-group rate > 90 % over first 50 groups | early abort (`grpo_train.py:756–760`) |

Launcher default: `ALLOW_STRICT_REWARD_FALLBACK=0`. Note the binary-reward early-abort did
not exist yet for the July 2–3 runs (which is exactly the failure it now guards against).

## 2. Trainer coupling (`grpo_train.py`)

- Group-relative advantage per task group (`num_generations` = 4 old / 8 new completions).
- **Dead group = zero between-completion reward std (corrected definition, "audit Bug 3");
  skipped from updates and logged** (`update: skipped_dead_group`) — they contribute no
  gradient but still consume rollout compute.
- Per-position between-completion advantages with optional normalization
  (`normalize_advantage=True`), turn-level credit from `r_seq`.
- Position-artifact detection: groups whose flattened std is nonzero but between-completion
  std is zero would have trained on pure turn-position artifacts under the old logic; now
  detected (`position_artifact_detected`) and gated (`position_artifact_rate_lt_max`, max
  0.2 — Stage 3 FAILED this at 0.35–0.41).
- Early abort if first-50-group dead rate ≥ threshold; kl_beta 0.15 vs frozen adapter-off
  reference (logged `kl` was 0.0 throughout — with lr 5e-7 and few contributing steps the
  policy barely moves).
- Clipped completions masked from updates (`training.mask_clipped_from_update=true`).

## 3. Does the reward correlate with what we care about?

Evidence from the audited runs (RUN_AUDIT.csv, FAILURE_MODE_AUDIT):

| relation | observation | verdict |
|---|---|---|
| training mean reward ↔ dev official win | Stage-2 runs mean reward 0.47–0.54 while dev win stayed at/below baseline (0.515–0.545 vs 0.535–0.57); Stage-3 mean reward ~0.49 flat across 2 epochs while dev win drifted −1 pp | **no observable positive correlation** at run granularity |
| reward ↔ official NESTFUL win, per-band | reward pays 0.25–0.35 for under-calling; official win for those episodes is ~0 unless the direct answer luckily matches; the ≥0.9 band does correspond to official-winnable traces | monotone by design at the extremes, but the middle bands (0.3/0.6) dominate the distribution and carry no win signal |
| reward ↔ too_few_calls | by construction capped at 0.30; empirically too_few episodes cluster at 0.25–0.3, i.e. they still earn ~30% of max — combined with band flatness this under-penalizes stopping early | partially — the cap works but the gap to "wrong-args" (0.6) is only 0.3 |
| reward ↔ avg_predicted_calls | avg calls rose Stage 2 → Stage 3 (1.36→2.19) with gold 2→3; within a stage no upward drift across epochs | weak |
| reward variance ↔ dead-group rate | dead rate 0.65–0.88 despite 17–18 distinct reward values existing corpus-wide: variance exists across tasks, **not within groups**. Per-group unique episode rewards = 1.09–1.33 of 8 | the band structure quantizes within-group diversity away — this is the central reward-design failure |
| group reward variance ↔ learning | only 12–35% of groups produce gradient; optimizer steps 29–89/epoch on 800-task stages | signal starvation |

## 4. Findings

1. **The reward is well-instrumented and dispatch-safe** (no silent fallback, loud predicate
   errors, resolved-policy assertions). The July fixes (same reward for r_seq and episode,
   corrected dead-group definition, position-artifact detection) are working as designed.
2. **The band design defeats GRPO**: within a group, diverse completions collapse into the
   same cap band (0.3 too_few / 0.6 wrong_args), producing zero advantage. Group-relative
   methods need *within-group* spread; per-corpus "fractional rewards present" is the wrong
   health check.
3. **Under-calling is under-penalized relative to its frequency**: at 42–56% of episodes,
   too_few at 0.3 vs wrong-args at 0.6 gives a weak, often zero-variance gradient toward
   emitting the next call.
4. Recommendation (for IMPLEMENTATION_PLAN P3): densify within-band scoring — e.g. continuous
   per-call credit inside the too_few band (0.30·(n_correct_calls/gold_n)), argument-binding
   partial credit inside the 0.6 band, and a small distinct penalty step between "direct
   answer, no calls" and "call 1 then stop" — then re-check dead-group rate with the stage
   probe before any full run.
