# NESTFUL / MT-GRPO — briefing pro schůzku

## TL;DR

- **Fine-tuning nepřekonal baseline** na hlavních metrikách (Full Acc, Win Rate).
- **Strict/simple reward** na ReAct konci curriculum **kolabuje** (Win 0.544 → 0.325, macro-F1 0.89 → 0.15).
- **Partial reward** drží **partial·s1e4 ReAct** prakticky na baseline (Win **0.543** vs 0.544); delší trénink (s3/s4) degraduje pomaleji než strict, ale stále pod baseline.
- **Direct** je stabilní (~Full 0.16, Win ~0.27–0.29); mírná degradace Win u finetuned checkpointů.
- Problém: **reward mismatch** — trénujeme proxy (strict/partial gold trace), metrika je **execution Win Rate**.
- **Doporučení:** jeden cílený běh s **execution-dominant reward** + early stopping podle val Win; ITAT = reward-design analýza (trace fidelity vs execution), ne benchmark improvement.

## Co víme jistě

| běh | paradigma | Full | Win | závěr |
|-----|-----------|------|-----|-------|
| baseline (no LoRA) | direct | 0.169 | 0.292 | baseline |
| baseline (no LoRA) | react | 0.000 | 0.544 | baseline |
| minimal strict s4e2 | direct | — | — | missing metrics |
| minimal strict s4e2 | react | 0.001 | 0.325 | collapse |
| partial s1e4 | direct | 0.163 | 0.274 | degradation |
| partial s1e4 | react | 0.000 | 0.543 | stable-near-baseline |
| partial s4e1 | direct | 0.164 | 0.274 | degradation |
| partial s4e1 | react | 0.001 | 0.450 | degradation |

Paper reference (NESTFUL Table 1, arXiv:2409.03797 v3 / EMNLP 2025, one-shot Direct / zero-shot ReAct):

Direct (Table 1): GPT-4o **0.59**, DeepSeek **0.43**, Hammer **0.16**, Llama-8B **0.06**, Mixtral **0.07**.

ReAct (samostatná tabulka, **bez GPT-4o**): DeepSeek **0.46**, Mixtral **0.30**, Hammer **0.07**, AgentLM **0.00**.

Náš Qwen3-4B baseline Direct: Win **0.292**, Full **0.169** — pod frontier modely, nad Llama-8B Direct (0.06).
ReAct baseline **0.544** není přímo srovnatelný s paper ReAct (jiný setup); paper ReAct (DeepSeek max 46 %).

### ReAct Win Rate podle checkpointu (policy drift)

| checkpoint | Win (agg) | per-sample | macro-F1 |
|------------|-----------|------------|----------|
| baseline | 0.544 | 0.000 | 0.894 |
| minimal s1e4 | 0.532 | 0.549 | 0.895 |
| minimal s2e4 | 0.513 | 0.518 | 0.734 |
| minimal s4e2 | 0.325 | 0.000 | 0.153 |
| partial s1e4 | 0.543 | 0.564 | 0.926 |
| partial s2e2 | 0.533 | 0.548 | 0.808 |
| partial s3e2 | 0.479 | 0.472 | 0.472 |
| partial s4e1 | 0.450 | 0.393 | 0.350 |

## Win/loss overlap (per-task official Win)

- **baseline_react vs partial_s1e4_react** (n=1861): both win 0.0%, baseline win / ft fail 0.0%, baseline fail / ft win 56.4%, both fail 43.6%
- **baseline_react vs minimal_s4e2_react** (n=1861): both win 0.0%, baseline win / ft fail 0.0%, baseline fail / ft win 0.0%, both fail 100.0%
- **baseline_direct vs partial_s1e4_direct** (n=1861): both win 23.8%, baseline win / ft fail 5.4%, baseline fail / ft win 3.6%, both fail 67.2%

## Co ještě nevíme

- Per-example overlap pro všechny požadované páry je k dispozici.

Heuristická failure taxonomy — viz `failure_taxonomy.csv`. Taxonomie je heuristická (parse/call-count/stop_reason/official_win vs strict); nereflektuje fine-grained wrong_function bez gold parse per call.

## Doporučený další krok

1. **Execution-dominant reward** (viz níže) — jeden běh, early stop na val Win Rate.
2. **ITAT framing:** mixed/negative reward design — „trace fidelity vs execution success“, ne claim o SOTA.
3. **Nepoužívat** macro-F1 Func jako success metric (formátová compliance, ~900 tříd).

### Návrh execution-dominant rewardu

```
R = 0.50 * tool_final_answer_pass
  + 0.20 * executable
  + 0.15 * grounded_step_similarity
  + 0.10 * valid_references
  + 0.05 * call_count_score

capy:
  if not parse_valid: R = 0.0
  if not executable: R = min(R, 0.30)
  if no_tool_calls: R = min(R, 0.25)
  if executable and tool_final_answer_pass: R = max(R, 0.80)
```

---
Generováno: `experiments/comparison/meeting_analysis.py`
