# GENERATION RUNBOOK

## Prerequisites

```bash
cd experiments/targeted_tool_data_factory
pip install -e ".[analysis,dev]"     # pydantic, PyYAML, numpy, jsonschema
                                     # + scipy, scikit-learn, rapidfuzz, pytest
```

CPU only. No GPU, no remote API, no LLM required for the core path.

## Tests

```bash
python -m pytest tests -q            # 48 tests
```

## Phase A — smoke (must pass before any pilot)

```bash
targeted-data generate --target nestful --version phaseA --candidates 50 --seed 20260726 --overwrite
targeted-data validate --target nestful --version phaseA --overwrite
targeted-data select   --target nestful --version phaseA --overwrite
targeted-data probe    --target nestful --version phaseA --overwrite
targeted-data split    --target nestful --version phaseA --overwrite
targeted-data export   --target nestful --version phaseA --overwrite
targeted-data report   --target nestful --version phaseA --overwrite
```

## Phase B — pilot (B1 1500 candidates; B2 deficit expansion is automatic)

```bash
targeted-data all \
  --config configs/pilot_local.yaml \
  --target nestful \
  --tracks adaptation,generalization \
  --adaptation-ratio 0.60 \
  --candidates 1500 \
  --seed 20260726 \
  --version pilot1 \
  --no-remote-api \
  --overwrite
```

Resume an interrupted run with `--resume` instead of `--overwrite`
(finished steps are skipped via `_<step>_<version>.DONE.json` markers).

## Phase C — optional Qwen3-4B probe (needs a local model server)

Start a local OpenAI-compatible server with the exact checkpoint
`Qwen/Qwen3-4B-Instruct-2507` (LM Studio / Ollama / local vLLM), then:

```bash
targeted-data probe --config configs/pilot_local.yaml --target nestful \
  --version pilot1 \
  --provider openai_compatible_local \
  --base-url http://127.0.0.1:1234/v1 \
  --model qwen3-4b-instruct-2507 \
  --overwrite
targeted-data report --target nestful --version pilot1 --overwrite
```

Without a reachable server the probe records `NOT_RUN_LOCAL` + P0 structural
difficulty only; nothing else in the pipeline depends on it.

## Outputs

```
outputs/profiles/nestful_profile.json, NESTFUL_PROFILE_REPORT.md
outputs/candidates/candidates_<v>.jsonl, cells_<v>.json, gen_stats_<v>.json
outputs/validated/validated_<v>.jsonl, rejected_<v>.jsonl, validation_summary_<v>.json
outputs/selected/selected_<v>.jsonl, selection_trace_<v>.jsonl, profile_match_<v>.json
outputs/selected/export_<v>/  (canonical/nestful/grpo/csv + manifest with SHA256)
outputs/splits/train|heldout|reserve_<v>.jsonl, leakage_audit_<v>.json
outputs/reports/PILOT_REPORT_<v>.md, run_state_<v>.json, verdict_<v>.json
docs/PILOT_REPORT.md, docs/COST_REPORT.md (latest run)
```

## Overwrite protection

Every step refuses to run over existing outputs without `--resume` or
`--overwrite`. New generator/config ⇒ use a new `--version`.
