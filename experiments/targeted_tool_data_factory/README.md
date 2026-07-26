# Targeted Tool Data Factory

Target-conditioned + program-first + executor-verified + failure-driven +
student-in-the-loop + transfer-validated data generation for tool-use models.

- First student: `Qwen/Qwen3-4B-Instruct-2507`
- First target: NESTFUL (dev split for profiling; test only as contamination blocklist)
- Core runs on CPU, no LLM, no paid API, deterministic and resume-safe.

## Install

```bash
pip install -e .
pip install -e ".[analysis,dev]"
```

## Local pilot

```bash
targeted-data all \
  --config configs/pilot_local.yaml \
  --target nestful \
  --tracks adaptation,generalization \
  --adaptation-ratio 0.60 \
  --seed 20260726 \
  --no-remote-api \
  --strict
```

Individual steps: `profile`, `generate`, `validate`, `select`, `probe`,
`split`, `export`, `report`. Every step supports `--resume`, `--seed`,
`--dry-run`, `--max-candidates`, `--no-llm`, `--strict`, `--version`,
`--overwrite`.

Docs: `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/PILOT_REPORT.md` (generated).
Outputs: `outputs/<version>/{profiles,candidates,validated,selected,splits,reports,cache}`.

NESTFUL-specific code lives only in `targets/nestful/` and
`configs/targets/nestful.yaml`.
