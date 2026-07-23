# Stage 3 NESTFUL Syntax Audit

Generated: 2026-07-23T13:20:41.063218+00:00
Verdict: **NO_MISMATCH**

## Canonical syntax (from code)

Tool-R0 training stack: labels `$varN`, references `$varN.<output_key>$` (prompt.py + executor.py). Field is required for object outputs; optional for scalars.

### Supported alternatives

- $var_N.field$ — NESTFUL gold / IBM scorer (accepted by executor)
- $varN$ without field — accepted by executor for scalar obs

## Counts

- Rows: 326
- References checked: 684
- Hard-fail rows: 0
- Incompatible refs: 0
- Rows changed by normalization: 0
- Input SHA-256: `0d3a2c6cce18ea14ead14e182a59b4b97ad3e76c65cff034b9196ccdea689e00`
- Output SHA-256: `(same as input / no-op)`

## Ref class distribution

- `tool_r0_canonical`: 684

## Problem sample IDs

(none)

## Before/after examples

(no-op — no changes)

## Verdict rationale

Trainer registry v5.0.2 hash=f945b18ccdc2… Tool-R0 ReAct prompt teaches `$varN.field$` (prompt.py). Executor accepts `$varN` and `$var_N` via var_?(\d+). Official NESTFUL gold predominantly uses `$var_N.result$` (stylistic; scorer accepts both). Stage 3 refs: {'tool_r0_canonical': 684}. All Stage 3 rows replay through synthetic executor and match Tool-R0 canonical syntax. No derived dataset written.
