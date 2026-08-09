# Pilot3 forensics

Offline forensic analysis of the Pilot3 Targeted Tool Data Factory experiment (C0-vLLM vs D1-vLLM on diagnostic-500).

## Non-goals

- No training, model inference, vLLM/HF generation, OpenRouter, or new synthetic data.
- Does not mutate checkpoints, scorer, reward code, factory generators, or source eval artefacts.

## Run

From repo root:

```bash
python -m experiments.targeted_tool_data_factory.analysis.pilot3_forensics.cli \
  --repo-root . \
  --output-dir experiments/targeted_tool_data_factory/reports/pilot3_forensics \
  --seed 42
```

Optional explicit paths: `--c0-trajectories`, `--d1-trajectories`, `--train-data`, `--full-train-data`, `--diagnostic-data`, `--train-log`, etc.

## Outputs

All reports land under `--output-dir`, including `PILOT3_FORENSIC_ANALYSIS.md`, integrity manifests, paired outcomes, topology/surface/coverage audits, reward observability notes, and recommended generation cells.

## Tests

```bash
python -m pytest experiments/targeted_tool_data_factory/analysis/pilot3_forensics/tests -q
```
