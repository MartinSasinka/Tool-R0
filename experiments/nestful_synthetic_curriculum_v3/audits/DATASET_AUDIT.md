# DATASET AUDIT — NESTFUL synthetic curriculum experiments

Date: 2026-07-09 · Read-only audit · Machine-readable companion: `DATASET_AUDIT.json`
(produced by `audits/tools/dataset_audit.py`; row identity checks by
`audits/tools/verify_data_base_identity.py`, leak inspection by `audits/tools/check_leaks.py`).

## 0. Executive answer: A vs B

**Datasets A and B are two entirely different corpora. Neither is derived from the other.**

| | A: `curriculum_v3_1/filtered/` | B: `filtered_toolr0_synthetic/` |
|---|---|---|
| Generator | v3.1 motif/failure-cluster pipeline (`build_curriculum_v3_1_pipeline.py`), July 2026 | older "Tool-R0 synthetic" pipeline (LLM-generated tasks, `synthetic-epochN-*` ids) |
| Schema | 21 fields incl. `sample_id`, `question`, `gold_calls`, `observations`, `gold_answer`, `motif_type`, `stage`, `source_failure_cluster`, `dependency_graph`, `process_labels` | 5 fields: `sample_id`, `input`, `output`, `tools`, `gold_answer` (NESTFUL-shaped) |
| Rows | 4 × 800 = 3,200 | 2,668 (400×6 shards, epoch_6 has 268; `curriculum_toolr0_all.jsonl` = copy of epoch_6 only, see §2.3) |
| Call counts | stage1=1, stage2=2, stage3=3, stage4=4–6 | epoch_N = exactly N calls |
| ID prefix | `prefix_v3_1_traj_v3_1_…` | `synthetic-epochN-NNNNNN` |
| Overlap with each other | **0 question hashes, 0 gold-trace hashes, 0 sample ids across all 28 A×B pairs** | — |
| Overlap with NESTFUL dev/test/full | 0 / 0 / 0 | 0 / 0 / 0 |
| Precomputed gold observations | yes (`observations`) | no |
| Used by which GRPO runs | **all v3/v3.1 curriculum runs** (via `run_curriculum_v3.sh` symlinks into `OUTPUT_ROOT/data_base/epoch_N_Ncall.jsonl`) | default `DATA_BASE` of `nestful_mtgrpo_minimal/run_curriculum.sh` and default `paths.train/eval/all_stages` in both `config.yaml`s — i.e. any run launched *without* the v3 wrapper |
| Canonical? | **Yes — canonical for the v3.1 experiments** | Legacy; keep only because `config.yaml` defaults still point at it |

Verified provenance of run inputs: for runs `20260707_103035_v3_1` and `20260708_212347_v3_1`,
every `data_base/epoch_N_Ncall.jsonl` row is **byte-identical (canonical-JSON hash) to the
corresponding A stage file, 800/800 rows, same id set** (file-level SHA256 differs only because
the pod copies are materialized symlinks). The two July-2 runs (`20260702_*`) used a **third**
corpus — the older `curriculum_v3` (non-v3.1) generator output with `task_id =
synthetic_v3_*` ids and only 223 stage-2 rows — so July-2 results are not comparable to v3.1 runs
at the dataset level either.

One residual confusion source: `nestful_mtgrpo_partial/config.yaml` still sets
`paths.eval_jsonl = data/filtered_toolr0_synthetic/epoch_4_4call.jsonl` and
`all_stages_jsonl = curriculum_toolr0_all.jsonl` (dataset B). The v3 pilots override
`paths.eval_jsonl` to the full NESTFUL file at launch, so B was *not actually used* by the
v3.1 runs — but any run that forgets the override silently evaluates on B.

## 1. Per-dataset report

Full field-by-field statistics are in `DATASET_AUDIT.json`. Summary:

### A. `experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1/filtered/`

| file | rows | calls | uniq questions | dup questions | dup gold traces | null answers | leaks in prompt |
|---|---|---|---|---|---|---|---|
| stage1_1call_atomic.jsonl | 800 | 1×800 | 800 | 0 | 0 | 0 | none |
| stage2_2call_dependency.jsonl | 800 | 2×800 | 800 | 0 | **1** | 0 | none |
| stage3_3call_composition.jsonl | 800 | 3×800 | 800 | 0 | 0 | 0 | none |
| stage4_4to6call_persistence.jsonl | 800 | 4:504, 5:233, 6:63 | 800 | 0 | 0 | 0 | none |

- Tools: 20–24 distinct tools used per stage (math/string/list families from
  `tool_registry_v3_1.py`); ~6–7 tools offered per task (incl. distractors).
- Motif distribution present (`target_full_motif`): linear_dependency, long_chain,
  reference_reuse, distractor_tools, boolean_output, string_output, argument_transformation, …
  (exact counts in JSON).
- Failure-cluster provenance present (`source_failure_cluster`) — internal metadata, **not**
  leaked into `question` (0 hits for motif/cluster/stage/trajectory patterns).
- `gold_answer` never null; no unresolved `$var…$` references in answers.
- **No overlap** with NESTFUL dev (200), test (1,661) or full (1,861) — by question hash,
  gold-trace hash, and sample id.
- Cross-stage duplicates inside A: none (stages are disjoint prefixes families but distinct
  questions/traces).

### B. `experiments/nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic/`

| file | rows | calls | dup traces | null answers | **gold_answer with unresolved `$var…$`** | prompt leak-pattern hits |
|---|---|---|---|---|---|---|
| epoch_1_1call.jsonl | 400 | 1 | 0 | 0 | 0 | 2 (benign: "star cluster", "server cluster") |
| epoch_2_2call.jsonl | 400 | 2 | 0 | 0 | **19** | 0 |
| epoch_3_3call.jsonl | 400 | 3 | 0 | 0 | **19** | 0 |
| epoch_4_4call.jsonl | 400 | 4 | 0 | **1** (`synthetic-epoch4-000170`) | 13 | 1 (benign) |
| epoch_5_5call.jsonl | 400 | 5 | 0 | 0 | **25** | 6 (benign: "stage1..4" are domain words) |
| epoch_6_6call.jsonl | 268 | 6 | 0 | 0 | **30** | 6 (benign) |
| curriculum_toolr0_all.jsonl | 268 | 6 | 0 | 0 | 30 | 6 (benign) |

- All "leak" hits were manually inspected: they are natural uses of the words
  cluster/stage in question text, **not** metadata leakage.
- **Real data-quality problems in B:** 106 rows (~4%) have a `gold_answer` that still contains
  unresolved variable-reference placeholders such as `"$var_1.output_0$"` instead of the
  executed value (e.g. epoch_2 row 0: `[[["$var_1.output_0$", …]]]`). Any final-answer-based
  metric or reward on those rows is unscoreable. One null gold answer in epoch_4.
- `curriculum_toolr0_all.jsonl` is **not** "all stages": it contains only the 268 6-call rows
  (question/trace/id overlap with `epoch_6_6call.jsonl` = 268/268) and stores `tools` as a JSON
  **string** rather than a list. The name is misleading; the `all_stages_jsonl` config default
  therefore never did what the name implies.

### NESTFUL reference data

| file | rows | notes |
|---|---|---|
| `data/splits/nestful_dev.jsonl` | 200 | dev slice of official data; 2–18 calls |
| `data/splits/nestful_test.jsonl` | 1,661 | 1 duplicate question, 13 duplicate gold traces (inherited from upstream) |
| `data/NESTFUL-main/data_v2/nestful_data.jsonl` | 1,861 | dev+test = full; final evals ran on the FULL file, i.e. **dev is contained in the "full NESTFUL" eval set** |

Note dev ⊂ full: full-NESTFUL evaluation (1,861) includes the 200 dev tasks used for
checkpoint selection. Contamination is mild (model selection saw dev only through win-rate
scalar), but a paper-grade protocol should report test-only (1,661) alongside full.

## 2. Which dataset should be canonical

1. **Canonical training data for the v3.1 experiments: dataset A**, specifically
   `outputs/curriculum_v3_1/filtered/stage*.jsonl` together with
   `curriculum_v3_1_manifest.json` (row counts match the manifest, 800/stage).
2. **Dataset B is legacy** (predecessor experiment). It should be archived; before archiving,
   nothing in the active v3.1 path may depend on it — today `config.yaml` defaults do
   (train/eval/all_stages), which is a footgun rather than a dependency.
3. The `curriculum_v3` (non-v3.1) corpus used by the July-2 runs is a third, superseded
   generation; treat as archive-only.

## 3. Risks found

| # | Risk | Severity | Where |
|---|---|---|---|
| D1 | Two unrelated synthetic corpora share the same `epoch_N_Ncall.jsonl` naming convention; run provenance is only recoverable via sample-id prefixes | high (confusion) | A vs B |
| D2 | `config.yaml` defaults still point to legacy B; forgetting one override silently trains/evals on the wrong corpus | high | `nestful_mtgrpo_{minimal,partial}/config.yaml` |
| D3 | ~4% of B rows have unresolved `$var…$` gold answers + 1 null answer | medium (only if B is reused) | B epoch_2..6 |
| D4 | `curriculum_toolr0_all.jsonl` contains only epoch-6 rows and string-encoded `tools` | medium | B |
| D5 | Full-NESTFUL final evals (1,861) include the 200-dev used for model selection | low-medium | eval protocol |
| D6 | 1 duplicate gold trace in A stage2 | negligible | A |
| D7 | July-2 runs used a third corpus (curriculum_v3, 223 stage-2 rows) — cross-run comparisons with v3.1 runs are not dataset-controlled | medium | runs `20260702_*` |

No gold-answer or internal-metadata leakage into user-visible prompts was found in either
corpus (the prompt builder consumes only `question` + `tools`; metadata fields are dropped by
`data.normalize_task`, verified in `nestful_mtgrpo_minimal/data.py` `_METADATA_FIELDS`).
