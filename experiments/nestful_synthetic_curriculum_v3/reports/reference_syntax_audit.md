# Reference Syntax Audit (C0/C1/C2 + Stage 3)
Generated: 2026-07-20T12:07:28.206946+00:00
Verdict: **NO_REFERENCE_FORMAT_MISMATCH**
## Canonical forms (from code)
- Tool-R0 ReAct prompt (`prompt.py`): `$varN.field$` (e.g. `$var1.result$`, `$var1.output_0$`).
- Executor (`executor.py`): `_VAR_REF_RE` accepts `$name` / `$name.field$`; index via `var_?(\d+)` → both `$var1` and `$var_1` resolve.
- Official NESTFUL gold often uses `$var_N.result$`; scorer `ground_seq_nested_repsonse` accepts `$var…` forms.
Stage 3 training audit verdict: `NO_MISMATCH` (see `stage3_nestful_syntax_audit.md`).
## Model-output reference class counts
| Class | C0 | C1 | C2 |
|---|---:|---:|---:|
| malformed_dollar_ref | 1 | 1 | 1 |
| tool_r0_canonical | 707 | 703 | 810 |

Malformed sample IDs (head): C0=['b930b196-c49a-4b90-9772-7b9b22f5d83a'], C2=['b930b196-c49a-4b90-9772-7b9b22f5d83a']

## Verdict rationale

Underscore vs no-underscore is **not** treated as an error: both are accepted by the executor and IBM scorer. Hard mismatches would be dollar-strings that fail `_VAR_REF_RE`. Counts of `malformed_dollar_ref` are low; Stage 3 gold is already Tool-R0-canonical. → **NO_REFERENCE_FORMAT_MISMATCH**.
