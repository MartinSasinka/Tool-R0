# NESTFUL multi-turn rollouts (standalone)

Runs the IBM/NESTFUL benchmark in multi-turn mode against any HuggingFace causal LM via vLLM. Writes one JSONL row per `(task, rollout)` plus an aggregate summary, matching the schema in `eval/results/nestful`.

Tool calls are dispatched **only** through NESTFUL's own helpers in `data_v2/executable_functions/` (preloaded `basic_functions.py` + lazy-imported `func_file_map.json`). No primitives, no judge — pass / fail is decided purely by numeric match against the gold answer.

## Install

Linux + CUDA (or ROCm). vLLM does not run on Windows.

```bash
pip install -r requirements.txt
```

## Run

```bash
# full benchmark on a single GPU
python run.py --output-dir results

# tensor parallelism over 4 GPUs
python run.py --tensor-parallel-size 4 --gpu-memory-utilization 0.92 --output-dir results

# pilot
python run.py --max-tasks 50 --num-rollouts 2 --output-dir results
```

The first run downloads `ibm-research/nestful` from HuggingFace and `git clone`s `https://github.com/IBM/NESTFUL` into `--nestful-repo-dir`. Both are reused on subsequent runs.

## CLI defaults

| flag                       | default                       |
|----------------------------|-------------------------------|
| `--model`                  | `Qwen/Qwen3-4B-Instruct-2507` |
| `--num-rollouts`           | `8`                           |
| `--max-tasks`              | None (all 1 861 tasks)        |
| `--max-steps`              | `10`                          |
| `--temperature`            | `0.7`                         |
| `--top-p`                  | `0.95`                        |
| `--max-new-tokens`         | `1024`                        |
| `--max-model-len`          | `10240`                       |
| `--tensor-parallel-size`   | `1`                           |
| `--gpu-memory-utilization` | `0.90`                        |
| `--output-dir`             | `nestful_results`             |
| `--seed`                   | `0`                           |
| `--nestful-repo-dir`       | `./nestful_repo`              |

`--model-profile NAME` overrides the auto-derived filename slug.

## Output

Two files in `--output-dir`:

* `<profile>_multiturn_predictions.jsonl` — one row per rollout
* `<profile>_multiturn_summary.json` — aggregate metrics

### Predictions row

```json
{
  "task_id": "3a4b7b77-...",
  "question": "If 20 liters of chemical X are added to 80 liters...",
  "status": "completed",
  "score": 1.0,
  "verdict": "pass",
  "verdict_reason": "executor_match",
  "stopped": "no_more_calls",
  "num_steps": 4,
  "predicted_final": 40.0,
  "gold_answer": 40.0,
  "predicted_calls": [{"name": "multiply", "arguments": {"arg_0": 0.25, "arg_1": 80}, "label": "var_1"}],
  "execution_trace": [{"index": 0, "name": "multiply", "label": "var_1",
                       "arguments_resolved": {"arg_0": 0.25, "arg_1": 80},
                       "result": 20.0, "error": null, "source": "ibm"}],
  "raw_completions": ["...", "..."],
  "execution_error": null,
  "trace_source_counts": {"ibm": 4},
  "num_tool_calls": 4,
  "error_category": "no_more_calls",
  "rollout_idx": 0,
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "tools": [...],
  "gold_calls": [...],
  "messages": [...]
}
```

`stopped`: `no_more_calls`, `execution_error`, `step_limit`, `context_limit`, `explicit_final`, `advance_error`.
`verdict_reason`: `executor_match`, `executor_mismatch`, `no_final_value`. Pass requires `abs(predicted - gold) < 1e-3` (or string equality if non-numeric).

### Summary

```json
{
  "benchmark": "nestful",
  "model_profile": "...",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "mode": "multiturn",
  "total_tasks": 14888,
  "completed": 7421,
  "failed": 7467,
  "errors": 0,
  "mean_score": 0.4985,
  "final_answer_accuracy": 0.4985,
  "passed": 7421,
  "avg_tool_calls": 2.3,
  "avg_steps": 3.1,
  "step_limit_hit_rate_percent": 6.07,
  "context_limit_hit_rate_percent": 2.26,
  "error_categories": {...},
  "stop_reason_breakdown": {...},
  "execution_class_breakdown": {"executed_ok_ibm": 11110, "unknown_function": 200},
  "ibm_registry_stats": {"available": true, "cached_imports": 44, "func_map_entries": 4416},
  "elapsed_seconds": 8222.13,
  "max_steps_setting": 10,
  "num_unique_tasks": 1861,
  "num_rollouts_per_task": 8,
  "total_rollouts": 14888
}
```

`total_tasks` counts rows (one per rollout); use `num_unique_tasks` for the raw NESTFUL task count.

## Analysis dashboard

After a run, analyze predictions and open the interactive diagnostic dashboard:

```bash
python analyze_results.py \
  --predictions ../nestful_results/qwen__qwen3-4b-instruct-2507_multiturn_predictions.jsonl \
  --output analysis_outputs
```

**Open `analysis_outputs/dashboard.html` directly in a browser** (double-click) — single self-contained file (~17 MB) with all data embedded. No HTTP server needed.

Also generated under `analysis_outputs/`:

* `dashboard.html` — **recommended** — one-file interactive dashboard
* `analysis_report.md` — markdown summary
* CSV exports (task/rollout level, problematic/unstable subsets)
* `plots/*.png` — static charts
* `dashboard_data.json`, `index.html` — multi-file variant (optional)
