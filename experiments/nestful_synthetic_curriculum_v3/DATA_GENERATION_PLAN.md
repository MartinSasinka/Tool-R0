# Data Generation Plan

**Training started: NO**

## Config (pilot)

- `total_tasks: 1030`
- `stage_minimums`: stage1=250, stage2=200, stage3=80, stage4=120
- `nestful_topup_tasks: 130` + `nestful_motif_boost` for coverage
- Mixed tool registry in `scripts/synthetic_tool_registry.py`

## Pipeline

generate → build → validate → gold replay → audit → tool realism → preflight

## Outputs

1030 tasks in `outputs/synthetic_motif_tasks.jsonl`  
Curriculum in `outputs/curriculum_v3/`
