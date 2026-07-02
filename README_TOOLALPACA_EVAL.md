# ToolAlpaca Evaluation (Tool-R0 Style)

This repo includes a standalone ToolAlpaca benchmark path that reuses the existing Tool-R0 canonical tool-call parsing and solver matching logic.

## Goal

- **Primary headline metric:** `eval_simulated.json`
- **Secondary supplementary metric:** `eval_real.json` (optional)
- **Main metric style:** canonical/AST-style structured matching (not BLEU/ROUGE/text exact-match of full response)

## What Is Reused From Existing Repo Logic

Evaluation utilities reuse solver-side parsing and scoring from `rewards_solver.py`:

- canonical schema: `{"name": "...", "arguments": {...}}`
- robust parsing helpers and normalization
- structured scoring components:
  - tool/function name match
  - argument key overlap (F1)
  - argument value match (robust value compare)
- extra predicted call penalty (`EXTRA_CALL_PENALTY_ALPHA`)
- greedy matching of each gold call to best unused predicted call

This is the closest in-repo implementation to the Tool-R0 paper's AST-style matching behavior.

## Files

- `scripts/toolalpaca_eval_utils.py`
  - ToolAlpaca loader/adaptation
  - robust output parsing
  - canonical conversion and evaluation wrapper
- `scripts/eval_toolalpaca.py`
  - single-model evaluation CLI
- `scripts/compare_eval_results.py`
  - base vs trained comparison summary
- `run_eval_toolalpaca.sh`
  - configurable runner for single/base/trained; optional real split
- `run_eval_toolalpaca_base_vs_trained.sh`
  - dedicated compare runner with simulated headline output

## Dataset Rules

- `train_data.json` is **not** allowed for benchmark eval.
- Use:
  - `./data/eval_simulated.json` for headline score
  - `./data/eval_real.json` for optional supplementary score

## Single-Model CLI

Base model on simulated split:

```bash
python scripts/eval_toolalpaca.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path ./data/eval_simulated.json \
  --output_path ./outputs/toolalpaca_eval/base_eval_simulated.json \
  --batch_size 8 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

Trained checkpoint on simulated split:

```bash
python scripts/eval_toolalpaca.py \
  --model_path ./qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50 \
  --dataset_path ./data/eval_simulated.json \
  --output_path ./outputs/toolalpaca_eval/trained_eval_simulated.json \
  --batch_size 8 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

Optional supplementary real split:

```bash
python scripts/eval_toolalpaca.py \
  --model_path ./qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50 \
  --dataset_path ./data/eval_real.json \
  --output_path ./outputs/toolalpaca_eval/trained_eval_real.json \
  --batch_size 8 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Runner Commands

### A) Base + trained headline comparison (simulated)

```bash
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
TRAINED_MODEL="./qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50" \
SIM_DATASET_PATH="./data/eval_simulated.json" \
OUTPUT_DIR="./outputs/toolalpaca_eval" \
bash run_eval_toolalpaca_base_vs_trained.sh
```

### B) Same as above + optional real supplementary run

```bash
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
TRAINED_MODEL="./qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50" \
SIM_DATASET_PATH="./data/eval_simulated.json" \
REAL_DATASET_PATH="./data/eval_real.json" \
RUN_REAL_EVAL=1 \
OUTPUT_DIR="./outputs/toolalpaca_eval" \
bash run_eval_toolalpaca_base_vs_trained.sh
```

### C) Generic runner

```bash
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
TRAINED_MODEL="./qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50" \
SIM_DATASET_PATH="./data/eval_simulated.json" \
RUN_REAL_EVAL=0 \
OUTPUT_DIR="./outputs/toolalpaca_eval" \
bash run_eval_toolalpaca.sh
```

## Output Artifacts

Typical files in `OUTPUT_DIR`:

- `base_simulated_toolalpaca_eval.json`
- `trained_simulated_toolalpaca_eval.json`
- `toolalpaca_comparison_simulated.json`
- `toolalpaca_comparison_simulated.md` (**headline table**)
- optional real split:
  - `base_real_toolalpaca_eval.json`
  - `trained_real_toolalpaca_eval.json`
  - `toolalpaca_comparison_real.json`
  - `toolalpaca_comparison_real.md`

Each summary JSON includes:

- `benchmark`, `dataset_path`, `split_name`, `split_role`
- `total_examples`
- `parseable_predictions`
- `exact_canonical_matches`
- `final_accuracy`, `final_accuracy_percent`
- `mean_name_match_rate`, `mean_key_match_rate`, `mean_value_match_rate`
- `mean_soft_score`
- `parse_reason_counts`
- `elapsed_seconds`

## Faithfulness Notes vs Paper

This implementation is faithful to the paper intent by using structured tool-call matching with canonical forms and component-wise name/key/value scoring reused from solver evaluation logic.

Potential deviations (explicitly acknowledged):

- public paper text does not fully expose every implementation constant/weight; repo constants are used (`LAMBDA_*`, extra-call penalty).
- malformed ToolAlpaca gold entries with non-JSON placeholders are normalized with best-effort fallback to keep eval reproducible.
- `eval_real` here is still tool-call prediction matching, not full live API execution success.
