# GENERATION CELLS

Cells are the unit of failure-driven quota control (DESIGN.md §7). They are
derived at `generate` time from the TargetProfile × StudentFailureProfile —
never hardcoded. The concrete cell list of a run is written to
`outputs/candidates/cells_<version>.json`; per-cell
requested/generated/validated/rejected/selected counts appear in
`outputs/candidates/gen_stats_<version>.json` and in PILOT_REPORT.md.

## Derivation rules

1. **Track share**: adaptation 60 % / generalization 40 % (CLI
   `--adaptation-ratio`).
2. **Call buckets** (2/3/4/5/6+): target dev shares, with a +4.5 pp 2-call
   oversample applied **only because** the measured Qwen3-4B failure profile
   has 2-call as its weakest bucket (D07). The 6+ bucket samples 6–8 calls.
3. **Motifs per bucket**: 2-call → linear; 3–4 → linear/fan_in/
   branch_aggregate; 5/6+ adds selection. Weighted by the profiled motif
   distribution (linear 55 %, fan_in 43 %).
4. **Skill/failure mapping**:
   - 2-call → `continuation_after_observation` /
     `wrong_second_tool_after_correct_prefix`;
   - 3–4-call → `variable_planning` / `too_few_calls`;
   - 5/6+ → `long_horizon_planning` / `premature_stop`;
   - every third sub-cell → `tool_catalog_search` / `distractor_confusion`
     (may add controlled irrelevant facts to the query);
   - one numeric-string cell per (track × bucket) →
     `argument_typing` / `numeric_string_confusion` (parse-number input or
     string-formatted answer).
5. **Hard-distractor rotation** across sub-cells:
   `same_signature_different_semantics`, `near_semantics`, `similar_name`.
6. **Offered-tool bucket rotation**: small 8–9 / medium 10–12 / large 13–18
   (histogram from the dev profile).
7. **Anti-dominance**: any cell whose derived weight exceeds ~5.5 % is
   subdivided; a > 10 % share of the final dataset would need explicit
   justification in PILOT_REPORT (none expected).

## Cell ID format

`{track}_{bucket}call_{motif}_{skill12}_{nn}` — e.g.
`A_2call_linear_continuation_00`, `G_6pcall_fan_in_tool_catalog__01`,
`A_3call_ns_numstring_00`. Every candidate and every exported row carries its
`generation_cell_id`; the selection step fills cell quotas by greedy deficit
matching, and Phase B2 regenerates only cells below 60 % coverage.
