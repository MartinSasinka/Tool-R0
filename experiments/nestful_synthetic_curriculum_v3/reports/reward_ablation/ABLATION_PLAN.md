# Reward-Only Ablation — Plan

Status: **Round 1 infrastructure implemented and CPU-tested. Round 1 GPU
training has NOT been launched from this environment** (no local GPU). This
document is the source of truth referenced by code comments throughout
`lib/reward_ablation_registry.py`, `lib/verifiable_process_reward.py`,
`scripts/ablation/*.py`, and `tests/test_reward_ablation*.py`.

## §1. Scientific question

Is the pure Stage-3 run's weakness caused by the **reward** (R0 sometimes
prefers official loss over official win: 22/93 wrong orderings in the
C0-win/E2-loss cohort, all fixed by outcome-first variants), or by the
**data**? This ablation isolates reward as the sole causal variable by
holding model, data, rollouts, hyperparameters, executor, credit
assignment, and evaluation identical across every arm.

## §2. Design — three-round reward tournament

**Round 1 (screening).** 5 arms (A0–A4, see §4) x 160 train tasks x 8
rollouts x 1 epoch x seed 20260724, evaluated on a fixed 500-task NESTFUL
diagnostic subset (temperature=0, top_p=1). Single experimental variable:
`reward.train_policy`.

**Round 2 (seed confirmation, NOT auto-launched).** A0_R0_CURRENT + the top
2 non-control arms that PASSed Round 1's hard gates (§12), re-run with seed
20260725 on the identical 160/500 subsets. `scripts/ablation/select_reward_arms.py`
prepares `ROUND2_PLAN.json` with exact commands; a human must launch it.

**Round 3 (final internal confirmation, prepare only).** A0_R0_CURRENT +
the single best arm from Round 2, on all 326 clean Stage-3 train tasks, 1
epoch, 2 training seeds, evaluated on the full available NESTFUL n=1661,
paired C0/R0/candidate comparison. Never launched automatically.

## §3/§4. Datasets (frozen, arm-independent)

- **Train**: `reports/reward_ablation/data/train_subset_160.jsonl` — 160
  tasks stratified from the 326-task clean Stage-3 dataset
  (`data/training_ready_v5/filtered/stage3_train_ready.jsonl`,
  SHA-256(LF-normalized)=`7df704bff35c8f8fd0ffb2b50e3c7c4c1e8d7f9a0f3e0c02a43327ef820dd596`)
  by motif, quality tier, call-count, and dependency structure; seed 20260724;
  every task replayed through the real synthetic executor; no NESTFUL leakage;
  no arm-specific filtering; easy tier capped at ≤10%. See
  `TRAIN_SUBSET_REPORT.md` for full stratification counts.
- **Eval**: `reports/reward_ablation/data/nestful_diagnostic_500_ids.json` —
  500 NESTFUL tasks stratified by gold call-count bucket (2/3/4/5/6+), motif,
  tool/domain family, and C0 outcome; seed 20260724; selection uses only
  dataset metadata + the (already-computed) C0 baseline, never any reward
  arm's results. This is an **internal development/ablation subset**, not an
  untouched final holdout — see `NESTFUL_DIAGNOSTIC_SUBSET.md`.

Both subsets are frozen (immutable once `*_manifest.json` is written) and
identical across every round/arm/seed.

## §5. Reward specifications and invariants

Centralized in `lib/reward_ablation_registry.py`. Unified 5-class terminal
taxonomy (best → worst, tested in `test_terminal_ordering_is_explicit_and_strict`):

```
official_success > executable_wrong_result > executable_partial
    > execution_failure > parse_or_no_call
```

- `official_success` = NESTFUL official `official_win` (eval) OR the
  synthetic path-invariant `final_pass` check (training) — whichever
  ground truth exists for that task.
- Every arm returns `{reward_id, terminal_class, terminal_score,
  process_score, epsilon, total_reward, components}`.
- `process_score` is always normalized to `[0, 1]`.
- Hard invariant (tested exhaustively in
  `test_no_process_component_can_flip_terminal_order`, for ALL possible
  `process_score` values, not just sampled ones):
  `epsilon * (P_max - P_min) < min adjacent gap between terminal scalars`.
- No arm penalizes `predicted_call_count < gold_call_count` directly — a
  shorter path is only worse if the actual outcome is worse (tested in
  `test_unified_terminal_class_no_gold_call_count_penalty`).

### Arm formulas

| Arm | Terminal bands | Process tie-break | Epsilon |
|---|---|---|---|
| A0_R0_CURRENT | own frozen v3.2 11-way bands (unchanged) | own dense per-turn shaping (unchanged) | n/a |
| A1_OUTCOME_ONLY | `OUTCOME_BANDS_R1` (imported) | none (`process_score := 0.0`) | `0.0` |
| A2_R3_OUTCOME_FIRST | `OUTCOME_BANDS_R3` (imported) | `_process_score_no_length` (gold-aware, = audited R3) | `0.02` |
| A3_VERIFIABLE_PROCESS | `OUTCOME_BANDS_R3` | `verifiable_process_components` (gold-FREE) | `0.02` |
| A4_GATED_VERIFIABLE | `OUTCOME_BANDS_R3` | same as A3, gated to `0.0` unless `gate_open(pred)` | `0.02` |

Epsilon for A2/A3/A4 is `min(DEFAULT_EPS_R3, 0.02)` rather than R3's own
`DEFAULT_EPS_R3=0.04`: extending R3's 4-class bands with the unified
taxonomy's 5th class (`executable_partial`) shrinks the tightest adjacent
gap to ≈0.0425, which would leave `DEFAULT_EPS_R3` almost no safety margin.
`0.02` gives ≈0.0225 margin. The bands themselves are reused verbatim.

A3/A4's verifiable process components (`lib/verifiable_process_reward.py`,
all gold-free, all `[0,1]`, checkable from trajectory + tool catalog +
execution outcome alone):

```
format_valid              0.10   parsed cleanly, not clipped
tool_exists_frac          0.15   named tool is in the task's own catalog
schema_keys_valid_frac    0.15   arg keys ⊆ declared params, required present
type_range_valid_frac     0.15   declared JSON type + min/max/minItems
reference_resolvable_frac 0.15   $varN[.field]$ references the executor resolved
execution_success_frac    0.15   real executor executed the call successfully
execution_integrity_frac  0.15   no executor-side failure on emitted calls
```

## §6. Freeze procedure (`scripts/ablation/freeze_reward_specs.py`)

1. run `tests/test_reward_ablation.py` (unit tests);
2. run the 16-task x 8-rollout probe (`scripts/ablation/reward_probe_16x8.py`,
   reusing `scripts/probe/probe_stage.py` — same rollout/reward-dispatch code,
   no optimizer step, no adapter write) for all 5 arms;
3. re-verify terminal-ordering + epsilon-band-safety invariants;
4. write `FROZEN_REWARD_SPECS.json` (formulas, scalars, epsilons, component
   weights, registry file hashes, git commit).

After freeze, terminal bands / epsilon / process weights / gates **must
not** change based on NESTFUL diagnostic-subset results. Any change
requires a new `reward_id` and a new `ablation_version`.

**Current freeze status**: `FROZEN_REWARD_SPECS.json` exists but was
produced with `--backend stub` (CPU pipeline self-test — deterministic fake
completions, proves wiring/registration/dispatch/invariants only). It is
**structurally valid but not a real GPU calibration**. Before spending real
Round 1 GPU time, re-run on RunPod:

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/freeze_reward_specs.py --backend vllm
```

`run_reward_ablation_round1.sh` refuses to launch Round 1 training unless
`FROZEN_REWARD_SPECS.json.probe.is_real_calibration == true` (override with
`ALLOW_STUB_FREEZE=1`, not recommended).

## §7-§8. Training infrastructure and parity

See `ABLATION_PARITY.md` (generated from the same YAMLs
`run_reward_ablation.py` loads) for the exhaustive list of everything held
identical. `scripts/ablation/run_reward_ablation.py` drives
`TwoPhaseTrainSession` directly (single `train_phase()` call = exactly one
epoch) rather than shelling out to `run_pure_stage3_two_epoch.py`, because
that script hardcodes 2 epochs / 326 rows / full dev-test sets. This keeps
100% of the actual trainer/rollout-pool/executor/checkpoint code identical
without editing or forking that production script.

## §9. Smoke test

`--smoke` caps each arm to 8 train tasks / 8 rollouts / ~2 optimizer steps
(`per_device_train_batch_size=1, gradient_accumulation_steps=4`) and 20
NESTFUL eval tasks. Gate: no crash, no NaN/Inf, optimizer step happened,
reward components logged, terminal inversions=0 for A1–A4, checkpoint
saves/loads, real synthetic executor (no gold replay), eval parity held,
W&B run created. Round 1 full training should only start after every arm's
smoke test passes on the target RunPod.

## §12. Hard gates (Round 2 elimination)

An arm is auto-excluded from Round 2 if any of: terminal inversion
count > 0; NaN/Inf present; an official loss outranks an official success
within the same rollout group; dead-group rate worse than A0 by > 5pp;
parse rate worse by > 2pp; executability worse by > 2pp; synthetic
path-invariant success drops sharply; executable_wrong positive-advantage
rate rises sharply; training reward rises while synthetic terminal success
falls; obvious reward hacking/degenerate strategy. A1_OUTCOME_ONLY may be
kept as a scientific control despite high dead-group rate (marked
CONDITIONAL, not eligible for Round 2 selection on its own).
Implementation: `scripts/ablation/select_reward_arms.py::evaluate_gates`.

## §13. Selection procedure

Lexicographic, NOT a single weighted score, NOT training-reward-mean:
(1) hard invariants/safety, (2) synthetic path-invariant terminal success,
(3) NESTFUL diagnostic win + paired gained/lost, (4) executable_wrong_result
rate, (5) executability/parse stability, (6) turn-2/3 conditional tool
accuracy, (7) dead/mixed group signal, (8) cross-seed stability, (9)
simplicity/interpretability. Implementation:
`scripts/ablation/select_reward_arms.py::rank_arms` /
`_lexicographic_key`. Verdicts: PASS / CONDITIONAL / FAIL.

## §17. Tests

- `tests/test_reward_ablation.py` — registry invariants (terminal ordering,
  epsilon-band safety, no gold-call-count penalty, alternative valid path,
  gold-free verifiable components, A4 gating, A0 parity, dataset-selection
  determinism + manifest hashes, group dead/mixed metric integration).
- `tests/test_reward_ablation_pipeline.py` — CLI/orchestration layer
  (config parity across arms, experiment-ID/hash determinism, eval-subset
  materialization, resume-safety guards, smoke-mode capping, summarize
  deliverables, McNemar test, hard-gate evaluation, lexicographic ranking,
  Round 2 plan construction).

Run: `python -m pytest experiments/nestful_synthetic_curriculum_v3/tests -q`
(both ablation test files pass together; 9 pre-existing failures in
`test_motif_extraction.py` / `test_prefix_decomposition.py` are unrelated —
caused by a missing `BOOLEAN_TOOL_NAMES` export in
`nestful_mtgrpo_minimal/synthetic_tool_registry.py`, predating this work).

## Known limitations

- Round 1 has not been executed on real GPU hardware from this session;
  `FROZEN_REWARD_SPECS.json` currently reflects a CPU stub probe only.
- `run_reward_ablation.py`'s heavy GPU code paths (`run_training`,
  `run_eval`) are structurally verified (dry-run mode, config/hash/resume
  logic all unit-tested) but not integration-tested end-to-end against a
  live vLLM DP pool from this environment.
- The A2 epsilon (0.02) is a deliberate deviation from R3's own
  `DEFAULT_EPS_R3` (0.04) to keep the unified 5-class taxonomy safe; A2's
  *bands* are still exactly R3's audited bands.
- `select_reward_arms.py`'s hard-gate checks for training-time signals
  (dead-group rate, terminal inversions, executable_wrong positive
  advantage, reward-hacking suspicion) expect a `training_diagnostics.json`
  produced from real training logs — this artifact does not exist yet
  because no real Round 1 run has been executed.
