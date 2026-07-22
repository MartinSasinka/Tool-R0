# Format Status — C0 / C1 / C2

## Verdikt pro supervizora (≤5 vět)

**Kategorie: `FORMAT_LARGELY_RESOLVED_SEMANTIC_ERRORS_DOMINATE`**

1. Základní ReAct formát se mírně zlepšil: ostrý `parse_fail` 74 → 63 (4.46% → 3.79%); širší syntax+no-call flag 10.96% → 10.54%.
2. Formát **není** hlavní bottleneck — u neúspěšných úloh dominuje sémantika (wrong tool / wrong values / executable-wrong-result); podíl semantic_dominant mezi non-wins ≈ 71.90%.
3. Zbývající formát: hlavně `parse:invalid_json` (často finální číslo v tagu místo `[]`) a `parse:no_tag` / no-tool-call; truncace je vzácná (8 (0.48%)).
4. Chyby jsou převážně **sémantické**; schema/reference jsou menšinové; „official parser_errors=0“ **nedokazuje** vyřešený formát — měří jinou věc.
5. Další krok: cílit credit assignment / Stage-3 reasoning (wrong values & wrong tool na 4–5 call), ne další format reward.

Generated: 2026-07-20T12:07:28.207949+00:00
Run: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\two_phase_20260718_192902\two_phase_20260718_192902`
n = 1661 paired nestful_test tasks

## 1. Definice vrstev

| Layer | Co znamená |
|---|---|
| A Raw output | prázdný text, neuzavřený tag, invalid JSON, truncace, no-call |
| B Parser | interní `parse_tool_call` gate → `parse_fail` |
| C Schema | unknown tool, missing/extra keys, type |
| D Reference | malformed `$…$`, unresolved var/field |
| E Semantic | wrong tool/values, executable wrong result, call-count |

## 2. Kanonické artefakty

- **C0**: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\two_phase_20260718_192902\two_phase_20260718_192902\eval\eval\final_test\C0_baseline`
  - `final_eval_trajectories.jsonl`, `metrics_official.json`, `final_eval_predictions.partial.jsonl`, `eval_manifest.json`
- **C1**: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\two_phase_20260718_192902\two_phase_20260718_192902\eval\eval\final_test\C1_phase1`
  - `final_eval_trajectories.jsonl`, `metrics_official.json`, `final_eval_predictions.partial.jsonl`, `eval_manifest.json`
- **C2**: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\two_phase_20260718_192902\two_phase_20260718_192902\eval\eval\C2_nestful_test`
  - `final_eval_trajectories.jsonl`, `metrics_official.json`, `final_eval_predictions.partial.jsonl`, `eval_manifest.json`
- Interní parser: `nestful_mtgrpo_minimal/parser.py` (`<tool_call_answer>` + exactly one call)
- Official parser path: `nestful_official_score.build_item` → `parse_llama_3_output` on **pre-extracted** JSON calls
- Prompt: `nestful_mtgrpo_minimal/prompt.py`
- Resolver: `executor.py` `_VAR_REF_RE` / `resolve_variables`

## 3. Rozpor: official parser_errors=0 vs interní parse/format

| Arm | official `num_pred_parsing_errors` | internal `parse_fail` |
|---|---:|---:|
| C0 | 0 | 74 |
| C1 | 0 | 68 |
| C2 | 0 | 63 |

Official metrics_official.json reports num_pred_parsing_errors=0 because final_eval feeds the official Llama-3.1 parser a JSON-serialized list of ALREADY extracted predicted_calls (build_item → generated_text=json.dumps(calls)). That list always parses; empty/partial lists are scored as 0 calls, not parse errors. Internal taxonomy 'parse/format error' counts ReAct rollout stop_reason=parse_fail (strict single-call <tool_call_answer> gate in nestful_mtgrpo_minimal/parser.py).

**Konkrétní sample IDs (C0 parse_fail head):** `056e78aa-e183-4205-95b9-e527e36055a3`, `07341e6a-f838-45df-b83e-1ed142ffa487`, `07d3dbd1-5b23-499b-abf7-de415450280f`, `0acb0dee-55be-47d6-8bc6-8cceba632ecc`, `0af2696a-9970-4e55-ad31-57e14ca65f20`, `0ba7094f-2d50-457b-9db1-af4dcb4d7bc8`, `0cea3214-8bbb-443a-943e-de9038f8b5f9`, `1067e243-98e4-448a-a38b-2d916c16908a`

### Příklad raw output (C0 parse_fail)

- sample_id: `056e78aa-e183-4205-95b9-e527e36055a3`
- stop_reason: `parse_fail`
- failing turn fail_reason=`parse:no_tag`
```
I need to compute the SHA-256 hash of the string 'Hello, World!' and then calculate the CRC-32 hash of that resulting hash. I will first compute the SHA-256 hash.
```
Official scorer never sees this raw ReAct text — it receives `predicted_calls` with [] already-parsed calls (often partial), so `parse_valid=True` / parsing_errors=0.

## 4. Kvantitativní srovnání

| Metric | C0 | C1 | C2 | C2−C0 |
|---|---:|---:|---:|---:|
| raw_output_syntax_failure | 182 (10.96%) | 181 (10.90%) | 175 (10.54%) | -0.42% |
| parser_extraction_failure | 74 (4.46%) | 68 (4.09%) | 63 (3.79%) | -0.66% |
| malformed_tool_call | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0.00% |
| unknown_tool | 8 (0.48%) | 9 (0.54%) | 11 (0.66%) | 0.18% |
| missing_argument_key | 41 (2.47%) | 42 (2.53%) | 42 (2.53%) | 0.06% |
| extra_argument_key | 41 (2.47%) | 42 (2.53%) | 42 (2.53%) | 0.06% |
| wrong_type_serialization | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0.00% |
| malformed_reference | 1 (0.06%) | 1 (0.06%) | 1 (0.06%) | 0.00% |
| unresolvable_reference | 2 (0.12%) | 2 (0.12%) | 1 (0.06%) | -0.06% |
| missing_output_field | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0.00% |
| final_answer_extraction_failure | 15 (0.90%) | 12 (0.72%) | 6 (0.36%) | -0.54% |
| output_truncation | 10 (0.60%) | 11 (0.66%) | 8 (0.48%) | -0.12% |
| unsupported_trace | 95 (5.72%) | 89 (5.36%) | 100 (6.02%) | 0.30% |
| executable | 1331 (80.13%) | 1316 (79.23%) | 1314 (79.11%) | -1.02% |
| wrong_tool | 163 (9.81%) | 160 (9.63%) | 172 (10.36%) | 0.54% |
| wrong_argument_value | 157 (9.45%) | 152 (9.15%) | 141 (8.49%) | -0.96% |
| executable_wrong_result | 172 (10.36%) | 179 (10.78%) | 187 (11.26%) | 0.90% |
| no_tool_call | 108 (6.50%) | 113 (6.80%) | 112 (6.74%) | 0.24% |
| syntax_format | 182 (10.96%) | 183 (11.02%) | 175 (10.54%) | -0.42% |
| schema_or_reference | 52 (3.13%) | 54 (3.25%) | 55 (3.31%) | 0.18% |
| semantic_dominant | 553 (33.29%) | 549 (33.05%) | 550 (33.11%) | -0.18% |

Bootstrap 95% CI (C2 syntax_format rate): [0.09151113786875377, 0.11980734497290789]
Bootstrap 95% CI (C2 semantic_dominant rate): [0.30885009030704397, 0.35460565924142085]

### Párově (syntax_format flag)

- **C1_vs_C0**: new_errors=45, resolved=44, net_rate_delta=0.0006, CI95=[-0.011438892233594221, 0.011438892233594221]
- **C2_vs_C1**: new_errors=45, resolved=53, net_rate_delta=-0.0048, CI95=[-0.016857314870559904, 0.006622516556291391]
- **C2_vs_C0**: new_errors=40, resolved=47, net_rate_delta=-0.0042, CI95=[-0.015653220951234198, 0.006622516556291391]

## 5. Per-turn / per-bucket

### First syntax-format failure turn (C2)

| Turn | count | % of all tasks |
|---|---:|---:|
| final_answer_segment | 6 | 0.36% |
| turn_1 | 52 | 3.13% |
| turn_2 | 12 | 0.72% |
| turn_3 | 3 | 0.18% |
| turn_4+ | 4 | 0.24% |
| unknown_or_no_call | 104 | 6.26% |

### By gold call count (C2)

| Bucket | n | syntax | parser_fail | schema/ref | semantic |
|---|---:|---:|---:|---:|---:|
| 2 | 543 | 7.92% | 4.05% | 7.18% | 40.33% |
| 3 | 363 | 13.22% | 3.86% | 2.75% | 25.07% |
| 4 | 223 | 9.87% | 3.14% | 1.35% | 31.84% |
| 5 | 154 | 9.74% | 3.25% | 1.95% | 32.47% |
| 6+ | 378 | 12.43% | 3.97% | 0.00% | 31.48% |

Parse reason dist C0→C2: {'no_tag': 19, 'invalid_json': 55} → {'no_tag': 24, 'invalid_json': 39}

## 6. Reference syntax

See `reference_syntax_audit.md`. Verdict: **NO_REFERENCE_FORMAT_MISMATCH**.

## 7. Kvalitativní příklady

Uloženo v `format_error_examples.json` (10 fixed, 10 regressed, 10 nuance, 10 semantic).

Ukázka fixed (C0→C2):
- `07341e6a-f838-45df-b83e-1ed142ffa487`: C0 stop=`parse_fail` → C2 stop=`terminal` win=False
- `0af2696a-9970-4e55-ad31-57e14ca65f20`: C0 stop=`parse_fail` → C2 stop=`executor_error` win=False
- `0cea3214-8bbb-443a-943e-de9038f8b5f9`: C0 stop=`parse_fail` → C2 stop=`executor_error` win=False

Ukázka semantic wrong-result (C2):
- `00830acf-eda7-4310-8396-aff5f243860e`: executable trajectory ending wrong result
- `008d5d59-d7f6-450a-8b28-9fffb47a5ca4`: executable trajectory ending wrong result
- `013c1149-7617-42df-ad8e-122f87f55cb7`: correct keys, wrong argument values

## 8. Layer mix among non-wins

| Arm | A syntax/parser | C/D schema/ref | E semantic | ok? |
|---|---:|---:|---:|---:|
| C0 | 178 | 51 | 543 | 0 |
| C1 | 180 | 53 | 536 | 0 |
| C2 | 173 | 54 | 538 | 0 |

## Message for supervisor

### Krátká verze

Formát tool callů se oproti C0 mírně zlepšil (parse_fail 74→63), ale už teď není hlavní problém — official „0 parser errors“ je navíc matoucí, protože oficiální scorer dostává už vytěžené JSON call listy, ne raw ReAct text. Většina proher je sémantická (špatný tool / hodnoty / výsledek).

### Delší verze

Po dvou fázích GRPO vypadá ostrý ReAct parse_fail spíš jako okrajový jev (cca 3.79% úloh; širší syntax+no-call 10.54%) než jako bottleneck: typické zbývající format chyby jsou „finální číslo v `<tool_call_answer>`“ nebo chybějící tag, ne rozbitá JSON syntax napříč trajektorií. Rozpor mezi official parser_errors=0 a interními desítkami parse/format chyb je definiční — jiný parser, jiný vstup. Mezi nevyhranými úlohami dominují wrong-tool / wrong-value / executable-wrong-result; schema a reference mismatch nejsou hlavní příběh (NO_REFERENCE_FORMAT_MISMATCH). Další investice by měla jít do sémantiky a credit assignment na delších řetězcích, ne do dalšího format rewardu.
