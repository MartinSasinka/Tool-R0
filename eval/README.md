# Tool-R0 Evaluation

Evaluation scaffold for comparing **baseline vs fine-tuned Tool-R0 models** on function-calling benchmarks.

## Benchmarks

| Benchmark | Tasks | What it tests | Ground Truth | Scorer |
|-----------|-------|---------------|-------------|--------|
| **BFCL AST** | 1040 | Function call accuracy (name, params, values) | AST matching | Standalone / official `bfcl-eval` |
| **BFCL Exec** | 240 | Real function execution (Python functions) | Execution result | Standalone / official `bfcl-eval` |
| **ToolAlpaca** | ~200 | Same metric as training reward | Exact tool calls | `compute_accuracy_score` |
| **API-Bank** | 753 | Real API call generation (73 APIs, diverse domains) | Exact match | Name + parameter accuracy |
| **ToolTalk** | 178 turns | Multi-turn real API conversations (28 tools, 7 suites) | Ground-truth oracle | Turn accuracy + conv. success |
| **NESTFUL** | 1861 | Nested sequences of API calls (math + coding) | Ordered sequence match | Full/partial match + Name F1 |

### BFCL (Berkeley Function Calling Leaderboard)

Industry gold standard for function calling evaluation (ICML 2025).

**AST categories** (ground-truth matching, 1040 tasks):

| Category | Tasks | Description |
|----------|-------|-------------|
| `simple` | 400 | Single function, single call |
| `multiple` | 200 | Select one function from 2-4 candidates |
| `parallel` | 200 | Multiple concurrent calls from one query |
| `irrelevance` | 240 | No function should be called |

**Executable categories** (real Python function execution, 240 tasks):

| Category | Tasks | Description |
|----------|-------|-------------|
| `exec_simple` | 100 | Execute single function call |
| `exec_multiple` | 50 | Execute selected function |
| `exec_parallel` | 50 | Execute parallel calls |
| `exec_parallel_multiple` | 40 | Execute parallel + multiple |

### ToolAlpaca

Real API schemas (OpenAPI) with ground-truth tool calls. Uses the **same scoring** as Tool-R0 training:
- Name match (weight=0.2), Key F1 (weight=0.3), Value match (weight=0.5)

### API-Bank (Real API Benchmark)

Comprehensive benchmark with 73 real-world executable APIs across diverse domains (healthcare, smart home, calendar, search, finance, etc.). Published at EMNLP 2023.

**Level 1 (given-desc)**: API descriptions are provided; the model must generate the correct API call with proper parameters from a conversation context.

- API Name Accuracy: correct API selected from available tools
- Parameter Accuracy: correct name AND all parameters match exactly
- Parameter F1: fine-grained F1 over parameter key-value pairs

Reference: Li et al., "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs", EMNLP 2023.

### ToolTalk (Multi-Turn Real API)

Microsoft's multi-turn conversational benchmark with 28 real tools across 7 suites (Account, Alarm, Calendar, Email, Messages, Reminder, Weather). Contains 78 conversations (28 easy, 50 hard) with ground-truth tool call annotations.

- Turn Accuracy: correct tool call(s) predicted at each conversation turn
- Conversation Success Rate: all turns in a conversation correct
- Easy vs Hard breakdown: easy conversations use 1 tool suite, hard use 2-3
- Precision/Recall: over individual tool calls

Reference: Farn & Shin, "ToolTalk: Evaluating Tool-Usage in a Conversation Setting", 2023.

### NESTFUL (Nested API Sequences)

IBM benchmark with 1861 tasks requiring nested/chained function call sequences (2-8 calls). The output of one call feeds as input to a subsequent call via variable references (`$var_N.result$`). Domains: mathematical reasoning and coding tools.

- Full Match Accuracy: entire predicted sequence matches gold (strictest)
- Partial Match Accuracy: per-call accuracy averaged across examples
- Name F1: F1 over predicted vs gold function name sequences
- Arg Match Ratio: argument correctness (handles variable references)

Reference: Basu et al., "NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls", 2024.

## Quick Start

### One-command DGX evaluation

```bash
# Run EVERYTHING: BFCL AST + BFCL Exec + ToolAlpaca, baseline + finetuned
bash eval/scripts/run_all.sh

# With custom model paths
BASELINE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
FINETUNED_MODEL=./qwen2.5-1.5b-instruct-tool-r0 \
bash eval/scripts/run_all.sh

# Quick test (limit tasks)
MAX_TASKS=10 bash eval/scripts/run_all.sh
```

### Individual benchmarks

```bash
# BFCL AST (all categories)
python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml --profile-name baseline

# BFCL Exec only
python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml \
    --profile-name baseline --category exec_simple,exec_multiple,exec_parallel,exec_parallel_multiple

# ToolAlpaca
python -m eval.run_eval --benchmark toolalpaca --config eval/configs/baseline.yaml --profile-name baseline

# API-Bank (real API benchmark)
python -m eval.run_eval --benchmark apibank --config eval/configs/baseline.yaml --profile-name baseline

# ToolTalk (multi-turn real API)
python -m eval.run_eval --benchmark tooltalk --config eval/configs/baseline.yaml --profile-name baseline

# NESTFUL (nested API sequences)
python -m eval.run_eval --benchmark nestful --config eval/configs/baseline.yaml --profile-name baseline

# Single BFCL category
python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml --category simple
```

### Compare results

```bash
python -m eval.scripts.compare \
    --baseline eval/results/bfcl/baseline_summary.json \
    --finetuned eval/results/bfcl/finetuned_summary.json \
    --output eval/results/bfcl/comparison.json
```

## Structure

```
eval/
├── run_eval.py                    # Unified CLI
├── model_adapter.py               # vLLM / OpenAI / dummy backends
├── metrics.py                     # Metrics & output helpers
├── parse_utils.py                 # Robust tool-call parser
├── benchmarks/
│   ├── bfcl/                      # BFCL (AST + Exec)
│   │   ├── loader.py              # Dataset loader from HuggingFace
│   │   ├── checker.py             # AST-style function call checker
│   │   └── runner.py              # Evaluation runner
│   ├── toolalpaca/
│   │   └── runner.py              # ToolAlpaca runner
│   ├── apibank/
│   │   ├── loader.py              # HuggingFace downloader + format converter
│   │   └── runner.py              # API-Bank Level 1 evaluation runner
│   ├── tooltalk/
│   │   ├── loader.py              # GitHub repo cloner + conversation parser
│   │   └── runner.py              # Multi-turn evaluation with GT oracle
│   └── nestful/
│       ├── loader.py              # HuggingFace downloader + format converter
│       └── runner.py              # Nested sequence evaluation runner
├── configs/
│   ├── baseline.yaml
│   ├── finetuned.yaml
│   └── smoke_test.yaml
├── scripts/
│   ├── run_all.sh                 # One-command full evaluation
│   ├── run_benchmark.sh           # Single benchmark script
│   ├── compare.py                 # Result comparison
│   └── smoke_test.sh
└── results/                       # Output directory
```

## DGX Deployment

```bash
pip install -r requirements.txt
pip install -r eval/requirements-eval.txt
export CUDA_VISIBLE_DEVICES=0,1,2,4   # exclude GPU 3 (DGX display)
bash eval/scripts/run_all.sh
```

## Output

Each run produces `{profile}_predictions.jsonl` (per-task) and `{profile}_summary.json` (aggregate).

BFCL summary includes per-category accuracy breakdown. ToolAlpaca includes component scores (name/key/value match). API-Bank includes API name accuracy, parameter accuracy, and parameter F1. NESTFUL includes full/partial match accuracy and name F1.
