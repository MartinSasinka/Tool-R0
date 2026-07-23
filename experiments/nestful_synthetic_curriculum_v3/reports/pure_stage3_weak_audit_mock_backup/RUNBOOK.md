# Weak-model audit — RUNBOOK

Run all commands from the **repository root** (`Tool-R0/`).

Default outputs: `experiments/nestful_synthetic_curriculum_v3/reports/pure_stage3_weak_audit/`

Default run: `outputs/runs/pure_stage3_2ep_20260719_221918`

## Prerequisites

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_WEAK_MODEL="${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-v3.2}"
```

Optional offline test without API:

```bash
# append --mock to canary/run
```

## 1. Discovery

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py discover
```

Writes: `DISCOVERY.md`, `discovery.json`

## 2. Prepare packets + pass inputs

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py prepare
```

Writes: `selected_task_ids.json`, `selection_manifest.json`, `SELECTION_SUMMARY.md`, `case_packets.jsonl`, `pass_a_inputs.jsonl`, `pass_b_inputs.jsonl`, `pass_b_mapping.json`, `compression_report.json`

Validate packets:

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate-packets
```

## 3. Canary (required before full run)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py canary \
  --model "${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-v3.2}"
```

Check `CANARY_REPORT.md` gate == PASS.

## 4. Full Pass A

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run \
  --pass-label A \
  --model "${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-v3.2}" \
  --concurrency 4 \
  --temperature 0 \
  --max-output-tokens 250
```

## 5. Full Pass B

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run \
  --pass-label B \
  --model "${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-v3.2}" \
  --concurrency 4 \
  --temperature 0 \
  --max-output-tokens 250
```

Raw outputs: `pass_a_annotations_raw.jsonl`, `pass_b_annotations_raw.jsonl`

## 6. Validate + repair JSON

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate
```

Writes: `pass_a_annotations.jsonl`, `pass_b_annotations.jsonl`, `invalid_annotations.jsonl`

Skip LLM repair (offline):

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate --no-repair
```

## 7. Aggregate agreement + clusters

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py summarize
```

Writes: `annotation_agreement.csv`, `ANNOTATION_AGREEMENT.md`, `cluster_counts.csv`, `cluster_examples.json`, `WEAK_MODEL_SUMMARY.md`, `HIGH_PRIORITY_CASES.jsonl`, `HIGH_PRIORITY_CASES.md`

## 8. Resume after interruption

Both `run` and raw JSONL append are resume-safe by `task_id:pass`.

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run \
  --pass-label A \
  --model "${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-v3.2}" \
  --resume
```

Use `--no-resume` to restart a pass from scratch (delete raw file first if needed).

## 9. Unit tests

```bash
python -m pytest experiments/nestful_synthetic_curriculum_v3/scripts/analysis/tests/test_weak_audit.py -q
```

## CLI flags (run / canary)

| Flag | Default |
|------|---------|
| `--model` | `OPENROUTER_WEAK_MODEL` or `deepseek/deepseek-v3.2` |
| `--base-url` | OpenRouter client default |
| `--api-key-env` | `OPENROUTER_API_KEY` |
| `--concurrency` | 4 |
| `--max-retries` | 3 |
| `--temperature` | 0 |
| `--max-output-tokens` | 250 |
| `--limit` | all (canary default 10) |
| `--seed` | 20260723 |
| `--mock` | offline deterministic JSON |

## Notes

- Executor + official scorer in case packets are authoritative; weak model is annotator only.
- Cases over 6000 input tokens are listed in `manual_oversize_cases.jsonl` and skipped by `run`.
- Do **not** interpret weak-model root causes as verified facts; see `WEAK_MODEL_SUMMARY.md` sections.
