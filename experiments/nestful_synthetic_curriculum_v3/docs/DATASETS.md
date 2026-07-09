# DATASETS

Full provenance analysis: `audits/DATASET_AUDIT.md` / `.json` (SHAs, row counts, overlap
checks, leak checks). This page is the operational summary.

## Canonical: dataset A — curriculum v3.1

`outputs/curriculum_v3_1/filtered/` (+ `curriculum_v3_1_manifest.json`):

| file | rows | gold calls |
|---|---|---|
| `stage1_1call_atomic.jsonl` | 800 | exactly 1 |
| `stage2_2call_dependency.jsonl` | 800 | exactly 2 |
| `stage3_3call_composition.jsonl` | 800 | exactly 3 |
| `stage4_4to6call_persistence.jsonl` | 800 | 4–6 |

Built by prefix-decomposing 888 NESTFUL-style failure trajectories; 100 % gold-replayable;
0 duplicate questions; no metadata leaks into prompts. This is what all v3.1 GRPO runs
trained on (the run-side `data_base/epoch_N_*.jsonl` files are symlinked/copied from these —
verified row-identical in the audit). The Stage-2 SFT view
(`outputs/sft/stage2_continuation/`) is a derived serialization of the same stage-2 file,
not a new dataset.

Known limitation: the corpus is **toy-like relative to NESTFUL** (~20 math/string tools,
`arg_0/arg_1` schemas). That is a transfer-gap concern (RESEARCH_FIX_PLAN E5), not a
validity concern.

## Legacy: dataset B — filtered_toolr0_synthetic

`experiments/nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic/` (`epoch_N_Ncall.jsonl`,
`curriculum_toolr0_all.jsonl`). Different provenance and generation era; quality issues
(~4 % unresolved `$var…$` placeholders in gold answers, a null answer, `tools` serialized as
a string, and `curriculum_toolr0_all.jsonl` actually contains only the 6-call slice).

**Do not train or evaluate on it.** Traps:

- `nestful_mtgrpo_{minimal,partial}/config.yaml` **still default to dataset-B paths** —
  never rely on their defaults; always pass explicit overrides (the v3 launcher does).
  Repointing these shared configs is P1 (needs approval).
- The eval batch runner and `check_env.sh` detect `filtered_toolr0_synthetic` in resolved
  paths and refuse (`--allow-legacy-dataset` is the explicit escape hatch).
- Dataset B uses the same `epoch_N_Ncall.jsonl` naming as run-side `data_base/` copies of
  dataset A — the filename alone does not identify the corpus; check the directory.

Archival to `archive/legacy_datasets/` is planned (P3).

## NESTFUL evaluation data

| file | n | role |
|---|---|---|
| `nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl` | 200 | checkpoint selection ONLY |
| `nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl` | 1,661 | headline evaluation |
| `nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl` | 1,861 | dev+test; used by historical "full" evals (mild dev contamination) |

Rules: never train on any of these; never copy their questions or gold traces into synthetic
data; new synthetic corpora must ship a zero-overlap proof (question hash, trace hash,
sample id) via `audits/tools/dataset_audit.py`. The audit verified dataset A has zero
overlap with NESTFUL eval data.

## Provenance conventions (new artifacts)

Every new dataset/eval result records: file SHA256 + row count (in the run manifest),
generator script + seed (for synthetic corpora), and git commit of the producing code.
`scripts/lib/paths.py` is the single source for canonical locations — do not hardcode
dataset paths in new scripts.
