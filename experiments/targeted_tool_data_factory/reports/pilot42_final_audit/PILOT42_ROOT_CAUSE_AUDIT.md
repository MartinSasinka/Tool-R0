# Pilot4.2 independent root-cause audit

- export dir (frozen, read-only): `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_2_workflow_grounded_v2`
- report dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot42_final_audit`
- auditor: `analysis/pilot43_independent_audit` - standard library only, no producer module imported; every number below is recomputed from exported JSONL content or read from an exported report file
- audited files: `train_master_3000.jsonl` (3000 records), `heldout_500.jsonl` (500 records), `reserve_500.jsonl` (500 records), `selected.jsonl` (4000 records)
- rows audited: 8000; unique tasks: 4000 (train_master_3000 + heldout_500 + reserve_500 are the same 4000 tasks as selected.jsonl, so all aggregate shares below count each task exactly once, while declared-vs-recomputed checks run over every row)
- independent audit verdict: **FAIL** with 15 deficits

## Scope and undecidable invariants

Pilot4.2 training records are thin (`call_count`, `cell_tier`, `gold_answer`, `gold_calls`, `question`, `requested_query_mode`, `task_id`, `tools`, `was_generated_from_workflow`, `workflow_id`), so no intermediate node values exist. Node output kinds are passed as `unknown` for every node except the sink, and `TYPE_TRANSITION_CHAIN` is therefore reported as **undecidable**, not as false.

Supplementary: `selected.jsonl` does export `oracle_observations`, which makes the invariant measurable for 4000 records; 0 of them satisfy TYPE_TRANSITION_CHAIN (0.000%). This is a supplement, not the verdict for the training files.

## Defect table

### 1. zero true 6+ call tasks

- **confirmed: yes**
- measured evidence:
  - recomputed call-count histogram from `gold_calls` over all 4 audited files: 2 calls: 1712, 3 calls: 659, 4 calls: 976, 5 calls: 653; maximum observed call count = 5
  - share of tasks with >= 6 calls = 0.000% (0 of 4000 records); share with >= 5 calls = 16.325%
  - provenance: len(record['gold_calls']) recomputed per record in train_master_3000.jsonl, heldout_500.jsonl, reserve_500.jsonl, selected.jsonl
- required fix for Pilot4.3: Pilot4.3 must generate programs with 6-10 calls as a first-class tier and gate on a recomputed histogram (not a declared call_count), with a hard minimum share of 6+ call tasks per split; the generator's program templates must be extended beyond the 2-5 node blueprints.

### 2. declared pattern label vs actual dependency DAG mismatch

- **confirmed: yes**
- measured evidence:
  - selected.jsonl `pattern_family` vs independently reconstructed DAG: 2692/4000 records declare a pattern their graph does not satisfy (disagreement rate 67.300%); 900 agree and 408 declare TYPE_TRANSITION_CHAIN, which the thin export cannot decide and which is therefore scored as undecidable rather than wrong
  - per declared label (agree / disagree / undecidable): DIAMOND: 0 / 245 / 0; FAN_IN_SINGLE: 0 / 574 / 0; LATE_REFERENCE: 0 / 244 / 0; LINEAR_CHAIN: 819 / 82 / 0; MULTI_JOIN: 0 / 161 / 0; PARALLEL_THEN_MERGE: 0 / 163 / 0; REPEATED_PRIMITIVE: 0 / 571 / 0; REUSE_EARLY_OUTPUT: 81 / 489 / 0; TWO_STAGE_AGGREGATION: 0 / 163 / 0; TYPE_TRANSITION_CHAIN: 0 / 0 / 408
  - the thin training files declare NO structural label at all: `pattern_family` is absent in 4000 of 8000 audited rows - i.e. in all of train_master_3000.jsonl / heldout_500.jsonl / reserve_500.jsonl, which carry only call_count, cell_tier, gold_answer, gold_calls, question, requested_query_mode, task_id, tools, was_generated_from_workflow, workflow_id
  - recomputed primary pattern distribution over the 4000 unique tasks: LINEAR_CHAIN: 3184, REPEATED_PRIMITIVE: 572, DIAMOND: 244
  - the declared labels are near-disjoint from the structures actually present - number of records that genuinely satisfy each invariant vs number of records declaring it: LINEAR_CHAIN: satisfied by 3756, declared by 901, both 819; FAN_IN_SINGLE: satisfied by 244, declared by 574, both 0; REPEATED_PRIMITIVE: satisfied by 653, declared by 571, both 0; REUSE_EARLY_OUTPUT: satisfied by 244, declared by 570, both 81; TYPE_TRANSITION_CHAIN: satisfied by 0, declared by 408, both 0; DIAMOND: satisfied by 244, declared by 245, both 0; LATE_REFERENCE: satisfied by 0, declared by 244, both 0; TWO_STAGE_AGGREGATION: satisfied by 0, declared by 163, both 0; PARALLEL_THEN_MERGE: satisfied by 0, declared by 163, both 0; MULTI_JOIN: satisfied by 0, declared by 161, both 0
  - worked examples (declared label, then the reconstructed graph): p42_6aabb059b6ce declares FAN_IN_SINGLE but has 3 nodes, 2 edges and 0 join nodes, satisfying ['LINEAR_CHAIN']; p42_b8fe6f19b5c7 declares REUSE_EARLY_OUTPUT but has 3 nodes, 2 edges and 0 join nodes, satisfying ['LINEAR_CHAIN']; p42_c4cf79664f33 declares LATE_REFERENCE but has 5 nodes, 5 edges and 1 join nodes, satisfying ['DIAMOND', 'FAN_IN_SINGLE', 'FAN_OUT', 'REPEATED_PRIMITIVE', 'REUSE_EARLY_OUTPUT']; p42_d299fc7147f5 declares FAN_IN_SINGLE but has 2 nodes, 1 edges and 0 join nodes, satisfying ['LINEAR_CHAIN']
- required fix for Pilot4.3: The structural label must be computed from the emitted gold_calls DAG by the same code path that the audit uses, exported per record, and hard-gated: any record whose declared pattern is not satisfied by its own reconstructed graph must be rejected before selection.

### 3. low real capability and primitive diversity

- **confirmed: yes**
- measured evidence:
  - distinct gold tool surface names actually called = 9; distinct primitives actually used = 9 (mapping source: record.tools[].semantic_id + primitive_registry.json)
  - distinct capability families actually used = 3: arithmetic.binary: 9468, comparison: 3021, statistics: 81
  - distinct exact primitive sequences = 15, top-1 sequence share = 24.400%, top-10 share = 91.900%
  - primitive counts: add: 3593, is_greater: 3021, increase_by_percent: 2778, decrease_by_percent: 1386, multiply: 976, subtract: 328, percent_of: 327, average_two: 81, divide: 80
- required fix for Pilot4.3: Pilot4.3 must impose a per-primitive and per-capability-family coverage floor measured from the exported gold_calls (for example >= 40 distinct primitives and >= 8 capability families, with no single primitive above 15% of calls and no single call sequence above 5% of tasks), and select against that recomputed coverage rather than against registry size.

### 4. only cosmetic generic/coding workflow labels over arithmetic programs

- **confirmed: yes**
- measured evidence:
  - share of gold calls whose capability family is arithmetic/comparison/logic = 99.356% (capability families overall: arithmetic.binary: 18936, comparison: 6042, statistics: 162)
  - 9 workflow ids carry a non-arithmetic domain label while their programs use only arithmetic/comparison primitives, e.g. boolean_logic.scale_adjust_compare -> ['arithmetic.binary', 'comparison']; date_time.scale_adjust_compare -> ['arithmetic.binary', 'comparison']; dictionary_processing.scale_adjust_compare -> ['arithmetic.binary', 'comparison']; file_path.scale_adjust_compare -> ['arithmetic.binary', 'comparison']; geometry.scale_adjust_compare -> ['arithmetic.binary', 'comparison']; list_processing.scale_adjust_compare -> ['arithmetic.binary', 'comparison']
  - domains present in the export (25): boolean_logic, classification, commerce, date_time, dictionary_processing, energy, file_path, geometry, inventory, list_processing, measurement, operations, personal_finance, quality, quality_control, rates, rates_and_ratios, resource_allocation, statistics, text_processing, threshold_decision, time_duration, travel, travel_distance, url_processing
  - workflow ids present in the export = 56 of 66 declared in workflow_registry.json
- required fix for Pilot4.3: A workflow label must be backed by primitives from its own domain: Pilot4.3 must require that every list/text/path/url/dictionary/date workflow uses at least one primitive from the matching capability family (sequence.*, text.*, path.*, url.*, datetime.*), verified from the exported gold_calls, and must drop domain labels that cannot be so backed.

### 5. non-functional / never-executed OpenRouter rendering and critic phases

- **confirmed: yes**
- measured evidence:
  - llm_rendered.jsonl = 0 bytes and llm_rendered_smoke.jsonl = 0 bytes: the rendering phase produced no output at all
  - openrouter_smoke_summary.json: n_input=30, n_passed=0, n_reject=30, pass_rate=0.0, llm_status='failed_or_empty', critic_coverage=0.0
  - openrouter_failures.jsonl contains 109 failure records (HTTP codes: 429: 83, other: 26)
  - no exported record carries a critic verdict: critic coverage = 0/4000 unique tasks; no `llm_critic` field exists on any record, in either the thin training files or the richer selected.jsonl
  - freeze_manifest.json: LLM_VALIDATED=False, llm_status='partial_smoke_attempted'
- required fix for Pilot4.3: Pilot4.3 must treat the LLM rendering and critic phases as hard dependencies: a non-empty llm_rendered.jsonl with one record per selected task, per-task critic verdicts stored on the record, and a build that fails when rendering pass rate or critic coverage is below the configured floor instead of silently exporting template text.

### 6. overly explicit, synthetic, repetitive questions

- **confirmed: yes**
- measured evidence:
  - exact `question` duplicate rate = 0.000% (4000 distinct texts over 4000 records)
  - normalized lexical skeleton concentration: 108 distinct skeletons, top-1 skeleton share = 1.300%, top-10 skeleton share = 12.300%
  - intent-template concentration: 25 distinct intents, top-1 intent share = 15.375%, top-10 intent template share = 72.550%
  - explicitness measured from the exported text: 100.000% of questions spell inputs out as a semicolon-separated fact list (mean 3.06 semicolons), 69.375% end with a 'Report x.' imperative, 97.950% contain 3 or more numeric literals (mean 4.55 numbers, mean length 33.8 words)
- required fix for Pilot4.3: Queries must be LLM-rendered and gated on recomputed diversity: cap the top-1 lexical skeleton share and the top-10 intent-template share (for example <= 5% and <= 25%), forbid the 'name is value; ... Report x.' scaffold, and require the target to be implied by the goal rather than named imperatively.

### 7. V4 shortcut check skipped for boolean and other non-numeric answers

- **confirmed: yes**
- measured evidence:
  - selected.jsonl `v4_gate.search.searched`: 3021/4000 records (75.525%) never ran the shortcut search; skip reasons: non-numeric answer: 3021
  - skipped records by recomputed answer kind: boolean: 3021 (recomputed from `gold_answer` with bool checked before int)
  - v4_report.json nevertheless reports shortcut_rate=0.0 and n_shortcuts=0 over n=4000, i.e. a clean rate derived from a search that was skipped for the majority of records
- required fix for Pilot4.3: The V4 shortcut search must cover every answer kind: enumerate boolean and categorical answers explicitly (a boolean answer has only two candidate values, so a shortcut predictor is trivially testable), record `searched: true` per task, and fail the build when shortcut-search coverage is below 100% of selected tasks.

### 8. insufficient per-node necessity evidence

- **confirmed: yes**
- measured evidence:
  - selected.jsonl `semantic_validation.layers.V_NODE_NECESSITY` present on 4000 records, of which 4000 carry a verdict only and 0 carry any per-node detail; observed keys = passed (4000), reasons (4000)
  - in the actual training artefacts the evidence is absent entirely: node-necessity coverage over the 4000 unique tasks as exported in train_master_3000.jsonl / heldout_500.jsonl / reserve_500.jsonl = 0 records with a `semantic_validation.layers.V_NODE_NECESSITY` block
- required fix for Pilot4.3: Necessity must be evidenced per node: for each node store the ablated-program answer and the observed answer delta, export that table on the record, and gate on 'every node changes the answer when removed' computed from the exported evidence rather than on a single boolean verdict.

### 9. workflow / program / query-template leakage between splits

- **confirmed: yes**
- measured evidence:
  - recomputed from record content: `workflow_id` shared between train_master_3000.jsonl and heldout_500.jsonl = 49 values, i.e. 100.000% of the 49 distinct heldout workflows were already seen in train
  - joined via task_id onto selected.jsonl: `query_template_fingerprint` shared train/heldout = 98 (100.000% of heldout), `program_family_id` shared train/heldout = 49 (100.000% of heldout)
  - key-independent proxy: 98 lexical query skeletons are shared between train and heldout, i.e. 100.000% of the 98 distinct heldout skeletons
  - split_manifest.json declares leak_free=True while reporting soft_key_overlap = {"query_template_fingerprint": 1000, "workflow_id": 1000, "program_family_id": 1000}
  - hard keys are indeed disjoint: `semantic_program_id` shared = 0, `workflow_instance_id` shared = 0
- required fix for Pilot4.3: Split on the generalisation-relevant keys, not on instance ids: workflow_id, program_family_id and query_template_fingerprint must be disjoint across train and heldout, `leak_free` must be false whenever any soft-key overlap is non-zero, and the split gate must be recomputed from the exported records instead of trusting the manifest.

### 10. strongly unbalanced boolean answers

- **confirmed: yes**
- measured evidence:
  - recomputed from `gold_answer`: 3021 boolean answers (75.525% of the 4000 selected records), overall True share = 78.120% (2360 True vs 661 False)
  - a majority-class baseline that always answers True therefore scores 78.120% on boolean tasks
  - 26 workflow ids with >= 20 boolean tasks are skewed beyond 70/30, e.g. commerce.four_step_adjust_total: 82/82 True (100.000%); energy.four_step_adjust_total: 81/81 True (100.000%); inventory.four_step_adjust_total: 82/82 True (100.000%); measurement.four_step_adjust_total: 81/81 True (100.000%); operations.four_step_adjust_total: 82/82 True (100.000%); personal_finance.four_step_adjust_total: 82/82 True (100.000%)
  - answer kind mix recomputed from `gold_answer`: boolean: 3021, float: 979
- required fix for Pilot4.3: Boolean answers must be balanced by construction and gated: sample threshold literals so that the True share sits in [0.45, 0.55] overall AND per workflow / per cell, and cap the overall share of boolean answers so the dataset is not dominated by a two-way guess.

### 11. unrealistic, trivially predictable values

- **confirmed: yes**
- measured evidence:
  - literal numeric arguments recomputed from `gold_calls` (32652 values): min=3.0, max=1800.0, mean=577.69, median=452.0, 1792 distinct values
  - share integer = 100.000%, share round multiple of 10 = 9.800%, share that looks like a generic random integer in 1..2000 = 100.000%
  - trivial predictability of the answer: the majority-class boolean baseline already reaches 78.120% (see defect 10), and no non-integer literal appears at all (share integer = 100.000%), so no realistic money/measure precision is present
- required fix for Pilot4.3: Values must be drawn from role- and unit-aware realistic distributions (prices with cents, rates in plausible bands, counts with realistic magnitudes, occasional outliers) and gated on recomputed realism statistics: bounded share of round integers, non-zero share of fractional values, and no single uniform range covering the majority of literals.

### 12. unmet or merely reported selection tier quotas

- **confirmed: yes**
- measured evidence:
  - selection_report.json declares selected=4000 of requested=4000 from eligible_pool=14843 with selection_all_hard_constraints_met=True and deficit=0, but contains no per-tier quota field (tier/quota-related keys present: none), so the quota claim cannot be verified from the export
  - recomputed `cell_tier` counts over selected.jsonl: STRUCTURAL_ENRICHMENT: 1952, CORE_PROFILE: 1716, CAPABILITY_ENRICHMENT: 169, CHALLENGE: 163
  - workflow support declared in selection_report.json is extremely uneven: min=1, max=82, 7 of 56 workflows have <= 2 supporting tasks while the modal workflow has 82
  - call-count support per tier recomputed from `gold_calls`: CAPABILITY_ENRICHMENT: 3 calls x169; CHALLENGE: 5 calls x163; CORE_PROFILE: 2 calls x1226, 3 calls x490; STRUCTURAL_ENRICHMENT: 2 calls x486, 4 calls x976, 5 calls x490
- required fix for Pilot4.3: Tier and cell quotas must be declared as explicit numeric targets in the selection config, exported per tier next to the achieved count, and enforced: a shortfall in any tier (including per-workflow minimum support, for example >= 20 tasks per workflow) must make the selection gate fail rather than be reported as met.

### 13. registry size vs primitives actually used in the dataset

- **confirmed: yes**
- measured evidence:
  - primitive_registry.json declares 89 primitives; the dataset actually uses 9 of them, i.e. registry coverage = 10.112%
  - workflow_registry.json declares 66 workflows; 56 appear in the exported records
  - distinct gold tool surfaces = 9, and the surface -> primitive map derived from `record.tools[].semantic_id` has 0 collisions
- required fix for Pilot4.3: Report and gate on used-primitive coverage, not registry size: export the used/declared ratio, require a configured minimum coverage of the registry, and remove or exercise primitives that no generated program ever calls.

### 14. metrics computed from metadata labels instead of exported content

- **confirmed: yes**
- measured evidence:
  - the declared structural label disagrees with the reconstructed DAG on 2692/4000 selected records (67.300%), so any metric aggregated over `pattern_family` describes labels rather than programs
  - v4_report.json reports shortcut_rate=0.0 over n=4000 while 3021 of 4000 records never ran the shortcut search
  - validation_report.json is a four-number summary ({"hard_gated": 14843, "rejected": 5157, "reject_reasons": {"v4_shortcut": 926, "v4_unresolved": 604, "query_fail": 3930}}) with no per-task or per-layer detail, and PILOT42_DATA_QUALITY_REPORT.json reports only counts plus label-derived booleans
  - the thin training files contain no structural, primitive or validation metadata at all (`pattern_family` absent in 4000 of 8000 audited rows), so their quality cannot be reconstructed from declared fields - only from content, as done here
- required fix for Pilot4.3: Every reported metric must be recomputed from the exported JSONL content by an auditor that cannot import producer code (this package), the audit must run as part of the build, and the freeze must carry the auditor's verdict rather than producer-side self-reports.

### 15. mixed OpenRouter logs from different runs

- **confirmed: yes**
- measured evidence:
  - openrouter_requests.jsonl holds 147 requests spanning 2026-07-31T14:39:28Z .. 2026-07-31T15:14:52Z with models openai/gpt-4o-mini-2024-07-18: 75, google/gemini-2.5-flash-lite: 67, mistralai/mistral-small-24b-instruct-2501: 5
  - models observed in the logs but not declared in openrouter_model_snapshot.json / openrouter_usage_summary.json (declared: google/gemini-2.5-flash-lite, openai/gpt-4o-mini-2024-07-18): mistralai/mistral-small-24b-instruct-2501
  - prompt_template_version values present: pilot41.critic.v1: 67, pilot41.writer.v1: 80 - Pilot4.1 template versions appear in a Pilot4.2 export
  - 74 of 147 request records carry no resolvable task id (task_ids == ['unknown']), so no log line can be attributed to an exported task; openrouter_failures.jsonl adds 109 records from google/gemini-2.5-flash-lite (89), mistralai/mistral-small-24b-instruct-2501 (3) plus 17 records with no model field
- required fix for Pilot4.3: OpenRouter logs must be per-run and attributable: write them under a run id directory, stamp every record with the run id, the prompt template version of THIS pilot and the concrete task ids, refuse to append to a log written by a different run id or template version, and fail the freeze when the declared model set does not equal the observed model set.

### 16. incomplete reproducibility and empty input hashes

- **confirmed: yes**
- measured evidence:
  - freeze_manifest.json provenance.input_hashes = {} (empty object: no input artefact is hash-pinned)
  - git state at freeze: commit=d174486ff105fc5b3daed71bdfb59ff572177fe5, dirty=True, n_dirty_files=46 - the export was produced from an uncommitted tree
  - freeze_manifest.json records cli_args=["select-pilot42", "--selected-target", "4000", "--resume", "--new-run-suffix", "v2"] and seeds={"seed": 20260731} but LLM_VALIDATED=False and llm_status='partial_smoke_attempted', so the LLM-dependent part of the pipeline is not reproducible from the manifest
  - MANIFEST.sha256.json is present (6964 bytes) and hashes outputs, but no input hashes exist to tie those outputs to their sources
- required fix for Pilot4.3: The freeze must pin inputs: hash every config, registry, workflow blueprint, prompt template and upstream JSONL into `input_hashes`, refuse to freeze from a dirty git tree (or record the full diff), and store the resolved model snapshot plus prompt hashes so an LLM-dependent run can be replayed.

### 17. dataset markable as complete despite unmet hard gates

- **confirmed: yes**
- measured evidence:
  - freeze_manifest.json: frozen=True, AUTOMATED_GATES_PASSED=True, selection_all_hard_constraints_met=True, while LLM_VALIDATED=False, TRAINING_READY=False, HUMAN_REVIEW_PENDING=True
  - PILOT42_DATA_QUALITY_REPORT.json reports AUTOMATED_GATES_PASSED=True and leak_free=True on the same export that shows soft_key_overlap={"query_template_fingerprint": 1000, "workflow_id": 1000, "program_family_id": 1000}
  - this independent audit records 15 deficits and returns verdict FAIL / INDEPENDENT_AUDIT_PASSED=False
  - concretely unmet at freeze time: zero 6+ call tasks; LLM_VALIDATED=false; non-zero soft-key split overlap; declared pattern labels contradicted by the DAG; V4 shortcut search skipped
- required fix for Pilot4.3: Completion must be a conjunction of independently recomputed gates: the build may set TRAINING_READY / COMPLETE only when the independent auditor returns PASS with zero deficits, and AUTOMATED_GATES_PASSED must not be settable while any hard gate (call-count tier, pattern agreement, leakage, LLM validation, shortcut coverage) is unmet.

## Independent audit deficits (verbatim)

- missing_field:pattern_family (absent in 4000/8000 records)
- missing_field:structural_skill (absent in 4000/8000 records)
- declared_pattern_disagreement measured 2692/4000 tasks vs required exactly 0
- actual_pattern_disagreement measured 2692/4000 tasks vs required exactly 0
- share_call_count_ge_6 measured 0.000000 vs required >= 0.050000
- top10_intent_share measured 0.725500 vs required <= 0.250000
- boolean_true_share measured 0.781198 vs required within [0.450, 0.550]
- n_distinct_primitives measured 9.000000 vs required >= 30.000000
- n_distinct_capability_families measured 3.000000 vs required >= 6.000000
- top1_primitive_sequence_share measured 0.244000 vs required <= 0.100000
- split_overlap[workflow_id][heldout_500] measured 49 vs required <= 0
- split_overlap[workflow_id][reserve_500] measured 49 vs required <= 0
- v4_coverage measured 0.000000 vs required >= 1.000000
- critic_coverage measured 0.000000 vs required >= 1.000000
- node_necessity_coverage measured 0.000000 vs required >= 1.000000

## Recomputed reference tables

### Call count (recomputed from gold_calls)

| calls | records | share |
| --- | --- | --- |
| 2 | 1712 | 0.4280 |
| 3 | 659 | 0.1648 |
| 4 | 976 | 0.2440 |
| 5 | 653 | 0.1633 |

### Call count per split

| split | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- |
| train_master_3000 | 1269 | 502 | 743 | 486 |
| heldout_500 | 217 | 84 | 111 | 88 |
| reserve_500 | 226 | 73 | 122 | 79 |
| selected | 1712 | 659 | 976 | 653 |

### Recomputed primary structural pattern

| pattern | records |
| --- | --- |
| LINEAR_CHAIN | 3184 |
| REPEATED_PRIMITIVE | 572 |
| DIAMOND | 244 |

### Primitives actually used

| primitive | capability family | calls |
| --- | --- | --- |
| add | arithmetic.binary | 3593 |
| is_greater | comparison | 3021 |
| increase_by_percent | arithmetic.binary | 2778 |
| decrease_by_percent | arithmetic.binary | 1386 |
| multiply | arithmetic.binary | 976 |
| subtract | arithmetic.binary | 328 |
| percent_of | arithmetic.binary | 327 |
| average_two | statistics | 81 |
| divide | arithmetic.binary | 80 |
