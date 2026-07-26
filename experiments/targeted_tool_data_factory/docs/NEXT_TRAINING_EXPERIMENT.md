# NEXT TRAINING EXPERIMENT — data-only D0 vs D1 (PLAN ONLY, do not run here)

## Question

Does target-conditioned data (this factory) improve Qwen3-4B on NESTFUL-like
nested tool use relative to the old Stage-3 data, holding everything else
fixed?

## Arms

- **D0**: 160 old Stage-3 tasks
  (`nestful_synthetic_curriculum_v3/data/training_ready_v5/filtered/stage3_train_ready.jsonl`,
  first 160 by the existing subset manifest).
- **D1**: 160 new target-conditioned tasks
  (`targeted_tool_data_factory/outputs/splits/train_pilot1.jsonl`, exported
  GRPO format `outputs/selected/export_pilot1/train_grpo_pilot1.jsonl`).
- **D2 (optional ablation, only if D1 > D0)**: 160 new tasks re-selected
  *without* target matching (random from the validated pool) to isolate the
  contribution of profile-conditioned selection.

## Held constant across all arms (verified by config diff before launch)

C0 checkpoint (`Qwen/Qwen3-4B-Instruct-2507`), reward
(`reward_ablation_A4_GATED_VERIFIABLE` — pending dispatch-canary PASS),
reward dispatch path (post-fix run.py), seed 20260726, 8 rollouts/task,
identical optimizer-step budget (2 epochs over 160 tasks), optimizer/LR/KL,
LoRA config, credit assignment, decoding params, eval pipeline.

## Integration prerequisite (one-time, before D1)

The rollout executor must resolve this factory's tool names. Add a registry
adapter that maps exported tool names → deterministic primitives
(`targeted_tool_data.registry`), mirroring how `synthetic_tools.py` is
consumed today. Gold-replay preflight on `train_grpo_pilot1.jsonl` must pass
100 % before training starts; abort otherwise.

## Evaluation (frozen before launch)

1. structural held-out 80 (`heldout_pilot1.jsonl`) — jointly and per G/A track;
2. fixed NESTFUL diagnostic-500 (same subset as reward Round-1);
3. metrics: path-invariant synthetic success, official NESTFUL Win Rate,
   Function F1, Parameter F1, executability, wrong-first-tool,
   continuation-after-correct-prefix, argument key/type/value errors,
   observation grounding, actual premature stop (undercalling with correct
   prefix), gained/lost task lists;
4. statistics: paired bootstrap 95 % CI on Win-Rate delta; McNemar on
   paired win/loss flips; report both even if n.s.

## Decision rule (pre-registered)

- D1 − D0 Win-Rate delta > 0 with McNemar p < 0.05 → adopt factory data,
  scale up generation (next version, larger pool).
- Delta ≈ 0 but held-out (G+A) improves → capability learned, transfer
  bottleneck elsewhere (surface realization/naturalness first suspect:
  enable local-LLM paraphrase pass, re-run).
- Held-out flat too → training-side problem (update strength/credit
  assignment per forensic audit), not data; stop dataset iteration.

## Cost estimate

2 arms × ~2 GPU-h (canary-calibrated) + eval 2 × 500 tasks ≈ 5–6 GPU-h on
the RunPod 4×GPU pod. Stop conditions: NaN/Inf in loss, dispatch-guard
assertion failure, replay failure in preflight.
