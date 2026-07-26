# DATA CARD — pilot1 (frozen 2026-07-25)

## What

320 target-conditioned, program-first, executor-verified nested tool-use
tasks for Qwen3-4B ↔ NESTFUL. Train 160 / structural held-out 80 /
reserve 80 (leakage-audited, 0 collisions).

## Provenance & hashes

- generator: `ttdf-0.1.0`, seed 20260726, config_hash `f41669a07d5ebff9`
- profile_hash `54524a0fd967bcc4…` (NESTFUL dev n=200, aggregates only)
- registry_hash `cb9eba8930f7c951…`, executor_hash `6faebc050c898324…`
- full file SHA256 list: `outputs/selected/export_pilot1/manifest_pilot1.json`
- key files:
  - `outputs/selected/export_pilot1/train_grpo_pilot1.jsonl` (sha256 2e92dcb71708f65e…)
  - `outputs/selected/export_pilot1/heldout_grpo_pilot1.jsonl` (1589e15546f72bc0…)
  - `outputs/selected/export_pilot1/nestful_compat_pilot1.jsonl` (8483d0945afbf2ff…)
  - `outputs/selected/export_pilot1/canonical_pilot1.jsonl` (b97856f227d0b37b…)

## Composition (selected 320)

| dimension | value | target (NESTFUL dev) |
|---|---|---|
| track mix | A 191 (59.7 %) / G 129 (40.3 %) | plan 60/40 |
| call counts | 2: 37.8 %, 3: 17.2 %, 4: 13.4 %, 5: 9.4 %, 6–8: 22.2 % | 33/22/13.5/9.5/22 (2-call +4.5 pp failure-driven; donor bucket was 3-call) |
| motifs | linear 70.3 %, fan_in 21.9 %, branch_aggregate 7.8 % | linear 55 %, fan_in 43 % (fan-in partially rejected by shortcut audit) |
| reference arg share | 0.381 | 0.397 |
| hard-distractor tasks | 100 % | floor 50 % |
| offered tools/task | mean 10.8 | mean 11.0 |
| numeric-string tasks | 12 (3.8 %) | ns-args 0.1 %, string answers 7 % |
| answer types | float 97.5 %, numeric-string 2.5 % | float 77 %, str/list/int/bool 23 % (known gap, LIMITATIONS #6) |
| question length | mean 190 chars | mean 167 |
| generation cells | 41 used, max cell share 5.9 % | cap 10 % |
| templates | 21 surface templates, max share 5.3 % | cap 5 % (2 marginal overshoots) |

## Quality gates (all measured, PILOT_REPORT.md)

- deterministic replay: 100 % (2× per task, hard gate)
- oracle: execution-only, no LLM anywhere in the oracle path
- contamination vs NESTFUL dev+test: 16 candidate hits (A-track skeleton
  collisions + 1 near-dup) — all rejected; **selected pool: 0**
- dedup: exact/normalized/program — 2 drops in pool, 0 in selected
- minimal-path/shortcut audit: 117 candidates rejected (V4), selected pool
  declared = minimal path everywhere
- split leakage: 0 collisions across 6 group keys
- profile match: closer than Stage-3 on 6/8 metrics; two-sample AUC
  0.550 (Stage-3: 0.728); call-count JSD 0.0033 (Stage-3: 0.585)

## Student probe

`NOT_RUN_LOCAL` (no local GPU/server). P0 structural difficulty recorded per
task. Exact command to complete Phase C: GENERATION_RUNBOOK.md §Phase C.

## Intended use

GRPO data-only experiment D1 (NEXT_TRAINING_EXPERIMENT.md). NOT a benchmark;
never evaluate NESTFUL improvements on this data itself — use the frozen
NESTFUL diagnostic-500.
