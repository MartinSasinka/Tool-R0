# Pilot4.1 root-cause audit (for Pilot4.2)

- git commit: `d174486ff105fc5b3daed71bdfb59ff572177fe5`
- branch: `main`
- dirty working tree: yes (~40 paths)
- Python: 3.11.3
- PyYAML: 6.0.3 · httpx: 0.27.2
- Pilot4.1 run: `outputs/pilot4_1_profile_safe/` (read-only)

## Executive conclusion

Pilot4.1 is **not** workflow-first. It generates an executable typed DAG first, then attaches a workflow label and renders a query from that label’s goal/roles. Validators check textual fact presence and graph-leak phrases, not whether the gold program solves the asked question. V4 runs **after** selection/export and never filters the train set.

## Failure modes

### 1. Capabilities outside workflow

**Mechanism:** `pilot41/generate.py::build_semantic_candidate` calls `pilot4.patterns.generate_program` with no `capability_mix`, then `pick_workflow(...)`, then **overwrites** the workflow with `cell.workflow_ids[0]`.

**Unused field:** `WorkflowFamily.allowed_capability_sequence` is serialized but never enforced in generation, validation, or selection.

**Frozen evidence (selected):** capability outside workflow ≈ **1477/1500**; exact sequence mismatch ≈ **1490/1500**.

### 2. Structural pattern outside workflow

**Mechanism:** Cells rotate over all 15 patterns (`pilot41/cells.py`). `pick_workflow` filters by pattern with soft fallback `or pool` (`workflows.py`). Cell workflow override ignores pattern fit.

**Frozen evidence:** pattern outside workflow ≈ **483/1500** selected, **332/1000** train.

### 3. Call count outside workflow

**Mechanism:** All default workflows use `max_calls=6`, but cell bucket `6+` samples calls in `{6,7,8}`. Soft fallback + override allow 7–8 call programs under 6-call workflows.

**Frozen evidence:** **181/1500** selected (7+8 call) exceed every workflow max.

### 4. Query–program semantic mismatch

**Mechanism:** `build_semantic_contract` assigns program constants **positionally** to workflow roles (`query_render.py`). Renderers speak the workflow goal, not the program ops. Validators V9–V11 do not check program↔query equivalence. V12 ignores `program_sufficient_for_query` / node necessity; critic is denied `semantic_program_summary`. Deterministic path skips V12 (`run_v12=False`).

**Frozen example:** quality-gate query + arithmetic/round program returning `502.0` with no comparison — still `query_validation.passed` and in train.

### 5. Shortcuts in final train

**Mechanism:** `finalize_dataset` selects → splits → writes train → **then** runs V4. `select_records` has no V4 filter. `safe_for_core_train` is never consumed. Freeze sets `frozen: true` even when `selection_all_hard_constraints_met: false`.

**Frozen evidence:** V4 shortcut rate **16.8%** (252/1500); train has **45** strictly shorter paths; **77** unresolved selected.

## Architectural classification

| Question | Answer |
|---|---|
| Is workflow the generative source? | **No** — post-hoc label |
| Is V4 a hard pre-selection gate? | **No** — post-export soft report |
| Primary defect | Retain Pilot4 random DAG + layer workflow semantics afterward |

## Pilot4.2 fix direction (implementation plan)

1. **WorkflowBlueprint → Instance → SemanticPlan → PrimitiveBinding → Typed DAG → Oracle → QueryContract → Validate → V4 → Select.**
2. Fail closed: no `or pool` fallback; capability/pattern/call-range/sink/role must match.
3. Facts = typed role bindings shared by executor, renderer, validators.
4. Hard `V_WORKFLOW_PROGRAM_QUERY_ALIGNMENT` before any LLM render.
5. V4 + node necessity + replay **before** selection; shortcuts/unresolved never enter any split.
6. Freeze refuses `AUTOMATED_GATES_PASSED` / `TRAINING_READY` unless hard constraints are true.
7. Prefer quality over count; emit deficit report instead of weakening gates.

## Non-goals (unchanged)

No GRPO/SFT/Qwen/vLLM/NESTFUL eval; do not mutate Pilot3/4/4.1 outputs; no NESTFUL win claim.
