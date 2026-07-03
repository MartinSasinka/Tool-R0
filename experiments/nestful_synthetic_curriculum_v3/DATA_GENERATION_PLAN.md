# Data Generation Plan

**Training started: NO**

## v3.1 (prefix/motif-aware call-count)

1. `generate_full_motif_trajectories_v3_1.py` → full trajectories from failure motifs
2. `build_prefix_curriculum_from_trajectories.py` → stage1–4 prefix samples (dedup-aware)
3. `polish_non_scalar_samples_v3_1.py` → optional stage3 non-scalar boost
4. `process_filter_prefix_samples.py` → filtered/
5. `final_dataset_audit_v3_1.py` → pre-pilot hard-gate audit
6. `analyze_dataset_uniqueness_v3_1.py` → uniqueness gates + reports
7. Integrity + gold replay + question-trace alignment + realism + preflight

Orchestrator: `build_curriculum_v3_1_pipeline.py`

**Rule:** long-chain motifs decomposed to prefixes — never 7+ calls in stage1/2.

## v3 legacy

- `total_tasks: 1030`
- `stage_minimums`: stage1=250, stage2=200, stage3=80, stage4=120
- `nestful_topup_tasks: 130` + `nestful_motif_boost` for coverage
- Mixed tool registry in `scripts/synthetic_tool_registry.py`

## Pipeline

generate → build → validate → gold replay → audit → tool realism → preflight

## Outputs

1030 tasks in `outputs/synthetic_motif_tasks.jsonl`  
Curriculum in `outputs/curriculum_v3/`
