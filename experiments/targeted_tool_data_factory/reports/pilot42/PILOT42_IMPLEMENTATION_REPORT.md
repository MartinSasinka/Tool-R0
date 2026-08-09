# Pilot4.2 implementation report

run_id: `pilot4_2_workflow_grounded` (artifacts `outputs/pilot4_2_workflow_grounded_v2/` — `--new-run-suffix v2` because the base directory already contained a prior incomplete run)

## 1. Executive summary

| Claim class | Status |
|---|---|
| IMPLEMENTED | Workflow-first generator, registries v2, hard validators, V4 pre-selection gate, nested subsets, freeze |
| GENERATED | 20000 candidates → 14843 hard-gated → 4000 selected (3000/500/500) |
| AUTOMATED_GATES_PASSED | **true** (this freeze) |
| LLM_VALIDATED | **false** (OpenRouter smoke not yet at 100% critic coverage on selected) |
| HUMAN_REVIEW_PENDING | **true** |
| TRAINING_READY | **false** (requires human audit import) |
| NOT_TESTED_BY_MODEL_PROBE / TRAINING / NESTFUL | **true** |

No claim is made that Pilot4.2 improves NESTFUL official win.

## 2. Scope and non-goals

No GRPO/SFT/Qwen/vLLM/NESTFUL eval. Pilot3/4/4.1 artifacts untouched.

## 3. Pilot4.1 root-cause findings

See `reports/pilot41_root_cause/PILOT41_ROOT_CAUSE_AUDIT.md`.

Primary defect: random typed DAG first, workflow label second; V4 after export.

## 4–10. Architecture

Generation order is enforced:

`WorkflowBlueprint → Instance → SemanticPlan → PrimitiveBinding → Typed DAG → Oracle → QueryContract → deterministic query → hard semantic validation → V4 gate → selection`.

Invariant: `was_generated_from_workflow=True` and `workflow_id` matches blueprint.

## 21–23. Dataset composition

- candidates: 20000
- hard_validated: 20000 (by construction + replay/necessity/alignment)
- V4+query gated: 14843
- selected: 4000
- train_master / heldout / reserve: 3000 / 500 / 500
- V4 shortcuts selected: 0
- V4 unresolved selected: 0
- nested: `train_core_{500,1000,2000}.jsonl` ⊂ `train_master_3000.jsonl`

## 25. OpenRouter

Config: `configs/pilot4_2_openrouter.yaml` (writer `openai/gpt-4o-mini-2024-07-18`, critic `google/gemini-2.5-flash-lite`). Budget cap $20. Replay mode supported via shared OpenRouter session.

## 30. Training readiness

`TRAINING_READY=false` until human-review thresholds are imported.
