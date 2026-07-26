# TARGET PROFILE — NESTFUL (dev, n=200)

Machine-extracted by `targeted-data profile --target nestful`; full JSON at
`outputs/profiles/nestful_profile.json`, generated report at
`outputs/profiles/NESTFUL_PROFILE_REPORT.md`. Aggregates only — no raw
queries or gold programs are stored (hygiene D03/D04).

## Key measured facts (2026-07-25)

| feature | value |
|---|---|
| call counts | 2: 33.0 %, 3: 22.0 %, 4: 13.5 %, 5: 9.5 %, 6+: 22.0 % (max 18) |
| motifs (ref-graph classified) | linear 55 %, fan_in 43 %, mixed 2 % |
| dependency depth | 2: 40 %, 3: 28 %, 4: 16.5 %, 5: 10 %, 6+: 5.5 % |
| reference task rate | **100 %** (every task chains via `$var` refs) |
| reference arg share | 39.7 % of all arguments (direct 60.3 %) |
| argument types | int 56.8 %, reference 39.7 %, string 2.1 %, list 1.2 %, numeric-string 0.1 % |
| answer types | float 77 %, list 7 %, string 7 %, int 5 %, bool 2 %, numeric-string 2 % |
| offered tools per task | mean 11.0, p25 9, p50 10, p75 13, range 7–19 |
| relevant/offered ratio | 25.6 % (≈ 3 gold vs ~11 offered) |
| tool names | 49 % single-word (math core), 28 % two-token, long snake_case tail |
| description length | mean ≈ 87 chars |
| question length | mean 167 chars (p25 118, p75 207) |

## Correction of a prior audit claim

The a06 forensic report claimed NESTFUL args are "15–24 % references + many
str-encoded numbers". Re-measurement with a reference detector that accepts
the test split's `$var_1` (underscore) form shows **100 % of dev and test
rows contain references** (test per-arg: int 7802, reference 5812, list 160).
The "str-encoded numbers" were mostly miscounted references. Quotas in this
factory follow the corrected numbers.

## Student failure profile (Qwen3-4B C0, forensic a05, n=500)

- official_success 285, executable_wrong_result 129, parse_or_no_call 45,
  executable_partial 32, execution_failure 9;
- win rate by bucket: 2-call **45 %** (weakest), 3-call 62 %, 4+ 59 %;
- undercalling (fewer calls than gold) ≈ 67 % of failures.

Consequence for cells: +4.5 pp oversample of 2-call (33 % → ≈ 37.5 %),
continuation-after-observation and catalog-search skills prioritized,
premature-stop pressure in 5/6+ cells.
