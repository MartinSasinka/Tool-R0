# Pilot4.1 implementation report

- commit: `d174486ff105fc5b3daed71bdfb59ff572177fe5` dirty=True
- generated: 2026-07-31T12:55:42.909270+00:00

## 1. Executive summary

### IMPLEMENTED

- semantic types + validate_semantic_edge
- workflow grammar (~40 families)
- non-leak deterministic query modes
- V9–V13 validators
- dense CORE cells
- OpenRouter writer/critic with budget + replay

### GENERATED

```json
{
  "candidates": 8068,
  "validated": 8068,
  "shortlist": 2000,
  "selected": 1500,
  "selected_llm_queries": 1032,
  "train": 1000,
  "heldout": 250,
  "reserve": 250
}
```

### DETERMINISTICALLY VERIFIED

- graph leak metrics vs Pilot4
- fact/unit preservation
- family-safe split
- executable oracle-before-query

### LLM-VALIDATED: True

### HUMAN-REVIEW REQUIRED / NOT TESTED BY TRAINING / NOT TESTED BY NESTFUL

No claim is made that Pilot4.1 improves NESTFUL official win.

## 2. Existing implementation audit

Pilot4 stages_related_rate (train): 0.845
Pilot4 high_or_complete graph leak: 0.8833

## 9. Cost and request statistics

```json
{
  "schema_version": "ttdf.openrouter.v41",
  "budget": {
    "requests": 4003,
    "prompt_tokens": 1630180,
    "completion_tokens": 710399,
    "usd": 0.245749,
    "max_requests": 100000,
    "max_usd": 20.0
  },
  "mode": "GENERATE_NEW_LLM_OUTPUTS",
  "writer_model": "mistralai/mistral-small-24b-instruct-2501",
  "critic_model": "google/gemini-2.5-flash-lite"
}
```

## 11. Dataset composition

```json
{
  "candidates": 8068,
  "validated": 8068,
  "shortlist": 2000,
  "selected": 1500,
  "selected_llm_queries": 1032,
  "train": 1000,
  "heldout": 250,
  "reserve": 250
}
```

## 17. Reproduction commands

```bash
python -m targeted_tool_data.cli audit-pilot4-language
python -m targeted_tool_data.cli build-workflow-registry
python -m targeted_tool_data.cli generate-semantic-pilot41 --candidates 10000
python -m targeted_tool_data.cli select-render-shortlist --target 2000
python -m targeted_tool_data.cli render-queries-openrouter --stage smoke
python -m targeted_tool_data.cli render-queries-openrouter --stage pilot
python -m targeted_tool_data.cli render-queries-openrouter --stage full
python -m targeted_tool_data.cli select-pilot41 --train 1000 --heldout 250 --reserve 250
python -m targeted_tool_data.cli audit-pilot41
python -m targeted_tool_data.cli implementation-report-pilot41
# replay freeze without API:
python -m targeted_tool_data.cli render-queries-openrouter --stage full --replay
```

## 18. Recommended training experiment, not executed

- Train MT-GRPO on pilot4_1 train-1000 with history-adaptive sampler and new logs.
- Matched-engine paired eval vs Pilot3/4 adapters before claiming NESTFUL gains.
