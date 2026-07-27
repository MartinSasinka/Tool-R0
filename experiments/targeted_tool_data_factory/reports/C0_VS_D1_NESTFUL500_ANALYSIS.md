# C0 vs D1 — NESTFUL-500 paired analysis

## Main result (one sentence)

On paired NESTFUL-500, D1 (Pilot3 GRPO n=300) reaches 57.6% official win vs C0 48.8% (Δ +8.8 pp, McNemar p=8.41e-05, gained 83 / lost 39); largest lift on 2-call tasks, but gain is not exclusive to them. Backend confound: C0 HF vs D1 vLLM.

## Artefact paths

- **c0_traj**: `experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C0_nestful500\final_eval_trajectories.jsonl`
- **d1_traj**: `experiments\targeted_tool_data_factory\outputs\runpod_pilot3\train_nestful500_from_zip\train_nestful500\eval\D1_nestful500\final_eval_trajectories.jsonl`
- **c0_manifest**: `experiments\targeted_tool_data_factory\outputs\runpod_pilot2\phase1_canary_from_zip\eval\C0_nestful500\eval_manifest.json`
- **d1_manifest**: `experiments\targeted_tool_data_factory\outputs\runpod_pilot3\train_nestful500_from_zip\train_nestful500\eval\D1_nestful500\eval_manifest.json`
- **diagnostic**: `experiments\targeted_tool_data_factory\runpod_bundle_pilot2\data\nestful_diagnostic_500.jsonl`
- **train_full**: `experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\train_grpo_pilot3.jsonl`
- **train_n300_definition**: `first 300 rows of experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\train_grpo_pilot3.jsonl (matches run_train_nestful500_4gpu.sh)`
- **config**: `experiments\nestful_mtgrpo_partial\config.yaml`
- **d1_eval_script**: `experiments\targeted_tool_data_factory\runpod_bundle_pilot3\eval_nestful500_sharded.py`

## 1. Fairness of comparison

| Check | Result |
|---|---|
| Paired sample_ids | **500/500** (only_c0=0, only_d1=0) |
| Diagnostic set | same `nestful_diagnostic_500.jsonl` |
| Temperature / top_p / rollouts / paradigm | 0.0 / 1.0 / 1 / react (both) |
| Scorer | official_win (NESTFUL) |
| Inference backend | C0: **HF (no use_vllm override in eval_manifest)**; D1: **vLLM (eval_nestful500_sharded.py DECODING)** |
| Backend identical? | **NO** |

**Confound:** C0 phase-1 canary eval did not set `hardware.use_vllm=true`; D1 used `eval_nestful500_sharded.py` which forces vLLM. Headline Δ is still directionally informative but not backend-clean. No training was run; C0 matched re-eval command is in §Commands.

## 2. Headline paired metrics

| Metric | C0 | D1 | Δ |
|---|---:|---:|---:|
| Official Win Rate | 48.8% (244/500) | 57.6% (288/500) | +8.8 pp |
| Function F1 (mean) | 0.432 | 0.598 | +0.165 |
| Parameter F1 (mean) | 0.179 | 0.263 | +0.083 |
| Executable | 71.4% | 83.8% | +12.4 pp |
| Final-answer pass | 46.6% | 59.6% | +13.0 pp |
| Solution-equivalent | 43.8% | 53.2% | +9.4 pp |
| Strict gold-trace | 8.4% | 16.2% | +7.8 pp |

- Gained / Lost / Both✓ / Both✗: **83 / 39 / 205 / 173**
- Bootstrap 95% CI for Δ win (pp): **[4.60, 13.20]**
- Exact McNemar p-value: **8.407e-05**
- Relative error reduction: **17.2%** (err 51.2% → 42.4%)

## 3. Where D1 gained / regressed

### By gold call count

| gold calls | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 100 | 23.0% | 47.0% | +24.0 | 28 | 4 |
| 3 | 100 | 59.0% | 62.0% | +3.0 | 14 | 11 |
| 4 | 100 | 57.0% | 65.0% | +8.0 | 17 | 9 |
| 5 | 100 | 55.0% | 60.0% | +5.0 | 11 | 6 |
| 6 | 29 | 48.3% | 58.6% | +10.3 | 6 | 3 |
| 7 | 16 | 50.0% | 50.0% | +0.0 | 2 | 2 |
| 8 | 19 | 42.1% | 36.8% | -5.3 | 2 | 3 |
| 9 | 7 | 71.4% | 85.7% | +14.3 | 1 | 0 |
| 10 | 5 | 40.0% | 60.0% | +20.0 | 1 | 0 |
| 11 | 4 | 75.0% | 50.0% | -25.0 | 0 | 1 |
| 12 | 2 | 50.0% | 50.0% | +0.0 | 0 | 0 |
| 13 | 7 | 42.9% | 42.9% | +0.0 | 0 | 0 |
| 14 | 2 | 50.0% | 50.0% | +0.0 | 0 | 0 |
| 15 | 1 | 100.0% | 100.0% | +0.0 | 0 | 0 |
| 18 | 2 | 50.0% | 100.0% | +50.0 | 1 | 0 |
| 19 | 1 | 100.0% | 100.0% | +0.0 | 0 | 0 |
| 20 | 2 | 50.0% | 50.0% | +0.0 | 0 | 0 |
| 25 | 1 | 100.0% | 100.0% | +0.0 | 0 | 0 |
| 36 | 1 | 0.0% | 0.0% | +0.0 | 0 | 0 |
| 53 | 1 | 0.0% | 0.0% | +0.0 | 0 | 0 |

### Focus 2–8 calls

| gold calls | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 100 | 23.0% | 47.0% | +24.0 | 28 | 4 |
| 3 | 100 | 59.0% | 62.0% | +3.0 | 14 | 11 |
| 4 | 100 | 57.0% | 65.0% | +8.0 | 17 | 9 |
| 5 | 100 | 55.0% | 60.0% | +5.0 | 11 | 6 |
| 6 | 29 | 48.3% | 58.6% | +10.3 | 6 | 3 |
| 7 | 16 | 50.0% | 50.0% | +0.0 | 2 | 2 |
| 8 | 19 | 42.1% | 36.8% | -5.3 | 2 | 3 |

Share of gained tasks that are 2-call: **33.7%** (28/83). Net win change is **not** only from 2-call (see contribution table in JSON).

### By answer type / motif / offered tools / ref density / prompt length

**Answer type**

| answer | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| bool | 7 | 28.6% | 100.0% | +71.4 | 5 | 0 |
| dict | 1 | 0.0% | 100.0% | +100.0 | 1 | 0 |
| float | 432 | 53.7% | 59.0% | +5.3 | 58 | 35 |
| int | 14 | 28.6% | 42.9% | +14.3 | 2 | 0 |
| list | 22 | 13.6% | 40.9% | +27.3 | 8 | 2 |
| numeric_string | 2 | 0.0% | 0.0% | +0.0 | 0 | 0 |
| string | 22 | 13.6% | 45.5% | +31.8 | 9 | 2 |

**Motif (from NESTFUL gold refs)**

| motif | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| fan_in | 283 | 53.4% | 60.1% | +6.7 | 40 | 21 |
| linear | 211 | 42.2% | 55.0% | +12.8 | 43 | 16 |
| mixed | 6 | 66.7% | 33.3% | -33.3 | 0 | 2 |

**Offered tools**

| offered | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| 10-12 | 236 | 55.9% | 58.1% | +2.1 | 27 | 22 |
| 13+ | 75 | 20.0% | 48.0% | +28.0 | 27 | 6 |
| 8-9 | 189 | 51.3% | 60.8% | +9.5 | 29 | 11 |

**Reference density**

| ref dens | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| ref0.25-0.45 | 423 | 51.1% | 57.7% | +6.6 | 62 | 34 |
| ref<0.25 | 3 | 0.0% | 0.0% | +0.0 | 0 | 0 |
| ref>=0.45 | 74 | 37.8% | 59.5% | +21.6 | 21 | 5 |

**Prompt length**

| prompt | n | C0 | D1 | Δ pp | gained | lost |
| --- | --- | --- | --- | --- | --- | --- |
| q120-199 | 207 | 49.8% | 59.4% | +9.7 | 40 | 20 |
| q<120 | 136 | 56.6% | 66.9% | +10.3 | 24 | 10 |
| q>=200 | 157 | 40.8% | 47.1% | +6.4 | 19 | 9 |

### Failure-class distribution

| class | C0 | D1 |
|---|---:|---:|
| success | 244 | 288 |
| too_few_calls | 215 | 121 |
| parse | 14 | 15 |
| too_many_calls | 11 | 19 |
| final_answer | 7 | 20 |
| wrong_args | 6 | 32 |
| wrong_tool | 2 | 4 |
| unresolved_reference | 1 | 1 |

Gained/lost task listing: `GAINED_LOST_TASKS.csv`.

## 4. Train-300 coverage vs transfer

Train definition: `first 300 rows of experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\train_grpo_pilot3.jsonl (matches run_train_nestful500_4gpu.sh)`.

| gold calls | train300 n | train share | eval Δ pp | gained | lost |
|---|---:|---:|---:|---:|---:|
| 2 | 89 | 29.7% | +24.0 | 28 | 4 |
| 3 | 58 | 19.3% | +3.0 | 14 | 11 |
| 4 | 43 | 14.3% | +8.0 | 17 | 9 |
| 5 | 34 | 11.3% | +5.0 | 11 | 6 |
| 6 | 44 | 14.7% | +10.3 | 6 | 3 |
| 7 | 18 | 6.0% | +0.0 | 2 | 2 |
| 8 | 14 | 4.7% | -5.3 | 2 | 3 |

Train-300 motif shares: {'branch_aggregate': 0.07333333333333333, 'fan_in': 0.43666666666666665, 'linear': 0.49}; answer shares: {'float': 0.677, 'string': 0.067, 'bool': 0.053, 'int': 0.077, 'numeric_string': 0.03, 'list': 0.097}; track shares: {'G': 0.4633333333333333, 'A': 0.5366666666666666}; distinct generation cells: 38.

**Correlation ≠ causation:** 2-call is both oversampled in train (failure-driven) and the largest eval lift — consistent with coverage, but does not prove those cells caused the win. Broader diagnostic lifts (executable, F1, sol_eq) suggest improved tool-use competence, not only memorizing 2-call templates.

See also `TRAIN_COVERAGE_VS_TRANSFER.md`.

## 5. DAG / program diversity (train-300)

- **topology_id**: unique=40, top1=29.7%, top10=74.0%, H=4.01 bits, mean tasks/family=7.50
- **primitive_program_id**: unique=295, top1=0.7%, top10=5.0%, H=8.20 bits, mean tasks/family=1.02
- **surface_program_id**: unique=298, top1=0.7%, top10=4.0%, H=8.22 bits, mean tasks/family=1.01
- **semantic_program_family**: unique=294, top1=0.7%, top10=5.3%, H=8.19 bits, mean tasks/family=1.02

Details: `DAG_DIVERSITY_AUDIT.md` / `.json`.

## 6. What we can claim

### Certain
- Same 500 sample_ids paired 500/500; same diagnostic JSONL path.
- Official win C0=0.488 D1=0.576; gained=83 lost=39.
- Exact McNemar p=8.407e-05; bootstrap 95% CI for Δpp=[4.60,13.20].
- D1 eval used vLLM; C0 phase1 manifest has no use_vllm=true override.
- Train subset for D1 was first 300 rows of train_grpo_pilot3.jsonl.

### Interpretation
- Largest absolute win lift on 2-call bucket aligns with train oversample of short tasks + student failure profile.
- Executable / final-answer / sol_eq / F1_func all move with official win → not only answer-flip noise.
- Positive deltas on 3–5 call buckets suggest broader transfer than 2-call-only, but smaller.

### Open
- How much of Δ is vLLM vs HF backend (need matched C0 vLLM re-eval).
- Causal role of specific train cells vs generic GRPO on tool-calling.
- Whether 8-call regression is noise (n=19) or systematic long-horizon regression.

## 7. Recommended next analyses (no training)

1. Re-eval **C0 with vLLM** into a new directory; recompute paired Δ.
2. Qualitative review of lost 8-call / long-horizon tasks in `GAINED_LOST_TASKS.csv`.
3. Stratify gained tasks by whether gold tools are A-track-like math names.
4. Compare internal F1 trajectories on gained vs lost (tool-choice vs args).
5. Contaminate-check: nearest train-300 neighbor (embedding/skeleton) for gained IDs.

## Commands for missing C0 matched inference

```bash
# Backend-matched C0 re-eval (NEW dir — do NOT overwrite phase1 C0)
cd /workspace/Tool-R0
mkdir -p experiments/targeted_tool_data_factory/outputs/runpod_pilot3/c0_vllm_placeholder
# no adapter_config.json => eval_nestful500_sharded.py sets model.lora_adapter=null
python experiments/targeted_tool_data_factory/runpod_bundle_pilot3/eval_nestful500_sharded.py \
  --run-dir experiments/targeted_tool_data_factory/outputs/runpod_pilot3/c0_vllm_placeholder \
  --diagnostic experiments/targeted_tool_data_factory/runpod_bundle_pilot2/data/nestful_diagnostic_500.jsonl \
  --out-dir experiments/targeted_tool_data_factory/outputs/runpod_pilot3/eval_C0_nestful500_vllm_matched \
  --run-py experiments/nestful_synthetic_curriculum_v3/run.py \
  --config experiments/nestful_mtgrpo_partial/config.yaml \
  --gpus 0,1,2,3
# DECODING inside the script already forces: use_vllm=true, T=0, top_p=1, react, 1 rollout
```
