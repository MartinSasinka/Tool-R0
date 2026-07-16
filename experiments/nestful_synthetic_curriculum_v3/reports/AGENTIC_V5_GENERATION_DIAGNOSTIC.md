# Agentic v5 — diagnostika generování

Souhrn ze čtyř worker běhů (`data/agentic_workers/`).

## Executive summary

| Run | Accepted | Accept rate | Kandidátů probed (log) | grpo_ok rate |
|-----|----------|-------------|------------------------|--------------|
| `agentic_v5_stage3_win1` | 6 | 2.7% | 219 | 3.2% |
| `agentic_v5_stage3_loose` | 13 | 13.0% | 99 | 31.3% |
| `agentic_v5_workers_stage3` | 0 | 0.0% | 0 | n/a |
| `agentic_v5_pilot_stage2` | 44 | 9.6% | 568 | 11.4% |

### Klíčové závěry

1. **Pipeline technicky funguje** — ve všech bězích `pool=5/5`, `cheap_rejects=0`; problém je až rollout gate.
2. **`win1` (ROLLOUT_REQUIRE_ACHIEVABLE_WIN=1) na Stage 3** → ~3.6% accept rate, 6/200 cíl; vyžaduje alespoň jeden full win v 8 rolloutech.
3. **`loose` (win=0) na Stage 3** → ~12% accept rate, 13/100 cíl; **3–4× vyšší throughput** při stejné obtížnosti.
4. **Stage 2 pilot** → 44 accepted, ~7–10% rate s win=1; model zvládá 2-call úlohy častěji než 3-call.
5. **Rejected JSONL s rollout vektory** existuje jen u staršího pilotu (gpu2); novější win1/loose běhy ukládají jen accepted + logy.

## agentic_v5_stage3_win1

**Agregovaný progress (RUN_PROGRESS.json):**
- accepted: **6** / target 200
- rejected: 214
- accept rate: **2.7%**
- local weak rollouts (Qwen): 1971
- OpenRouter spend: $0.0625
- top rejects: `{'low_grpo_signal_prediction': 212, 'unresolved_var': 1, 'not_nestful_like': 1}`

**Z GPU logů:**
- 219 kandidátů × 8 rolloutů ≈ **1752 Qwen episod**
- grpo_ok (prošlo gate): 7 (3.2%)

**Přijaté úlohy — rollout profil:**
- full_success_rate: min=0.125 max=0.625 avg=0.375 (0%=0, partial=6)
- unique_rewards: `{2: 4, 3: 2}`
- quality tiers: `{'frontier': 6}`

### Worker `gpu0`
- progress: 2/50 (iter 11, status=running)
- last batch: grpo_ok=0/5, rate=0.036, rejects=`low_grpo_signal_prediction:5`
- accepted: 2

### Worker `gpu1`
- progress: 3/50 (iter 11, status=running)
- last batch: grpo_ok=0/5, rate=0.055, rejects=`low_grpo_signal_prediction:5`
- accepted: 3

### Worker `gpu2`
- progress: 1/50 (iter 11, status=running)
- last batch: grpo_ok=0/4, rate=0.018, rejects=`low_grpo_signal_prediction:4, unresolved_var:1`
- accepted: 1

### Worker `gpu3`
- progress: 0/50 (iter 11, status=running)
- last batch: grpo_ok=0/5, rate=0.0, rejects=`low_grpo_signal_prediction:5`

## agentic_v5_stage3_loose

**Agregovaný progress (RUN_PROGRESS.json):**
- accepted: **13** / target 100
- rejected: 87
- accept rate: **13.0%**
- local weak rollouts (Qwen): 891
- OpenRouter spend: $0.0385
- top rejects: `{'low_grpo_signal_prediction': 68, 'not_nestful_like': 8, 'best_of_n_not_selected': 6, 'ambiguous_question': 4, 'unresolved_var': 1}`

**Z GPU logů:**
- 99 kandidátů × 8 rolloutů ≈ **792 Qwen episod**
- grpo_ok (prošlo gate): 31 (31.3%)

**Přijaté úlohy — rollout profil:**
- full_success_rate: min=0 max=0.875 avg=0.2308 (0%=9, partial=4)
- unique_rewards: `{2: 12, 3: 1}`
- quality tiers: `{'partial_frontier': 9, 'frontier': 4}`

### Worker `gpu0`
- progress: 3/25 (iter 5, status=running)
- last batch: grpo_ok=2/5, rate=0.12, rejects=`low_grpo_signal_prediction:3, not_nestful_like:1`
- accepted: 3

### Worker `gpu1`
- progress: 4/25 (iter 5, status=running)
- last batch: grpo_ok=4/5, rate=0.16, rejects=`not_nestful_like:2, low_grpo_signal_prediction:1, best_of_n_not_selected:1`
- accepted: 4

### Worker `gpu2`
- progress: 4/25 (iter 5, status=running)
- last batch: grpo_ok=2/5, rate=0.16, rejects=`low_grpo_signal_prediction:3, not_nestful_like:1, ambiguous_question:1`
- accepted: 4

### Worker `gpu3`
- progress: 2/25 (iter 5, status=running)
- last batch: grpo_ok=1/4, rate=0.08, rejects=`low_grpo_signal_prediction:3, unresolved_var:1`
- accepted: 2

## agentic_v5_workers_stage3

**Agregovaný progress (RUN_PROGRESS.json):**
- accepted: **0** / target 200
- rejected: 434
- accept rate: **0.0%**
- local weak rollouts (Qwen): 0
- OpenRouter spend: $0.1349
- top rejects: `{'invalid_schema': 257, 'semantic_incompatible_reference': 57, 'weak_solver_passed': 35, 'non_executable_gold_trace': 34, 'too_hard_both_solvers_fail': 22, 'invalid_trace_labels': 16, 'low_grpo_signal_prediction': 6, 'strong_solver_failed': 4, 'unresolved_var': 1, 'invalid_json': 1, 'metadata_leakage': 1}`

### Worker `gpu0`
- progress: 0/50 (iter 11, status=running)
- last batch: grpo_ok=None/None, rate=0.0, rejects=`invalid_schema:8`

### Worker `gpu1`
- progress: 0/50 (iter 17, status=running)
- last batch: grpo_ok=None/None, rate=0.0, rejects=`invalid_schema:6, semantic_incompatible_reference:2`

### Worker `gpu2`
- progress: 0/50 (iter 13, status=running)
- last batch: grpo_ok=None/None, rate=0.0, rejects=`invalid_schema:7, too_hard_both_solvers_fail:1`

### Worker `gpu3`
- progress: 0/50 (iter 15, status=running)
- last batch: grpo_ok=None/None, rate=0.0, rejects=`invalid_schema:1`

## agentic_v5_pilot_stage2

**Agregovaný progress (RUN_PROGRESS.json):**
- accepted: **44** / target 52
- rejected: 414
- accept rate: **9.6%**
- local weak rollouts (Qwen): 5112
- OpenRouter spend: $0.0718
- top rejects: `{'low_grpo_signal_prediction': 397, 'not_nestful_like': 8, 'best_of_n_not_selected': 6, 'ambiguous_question': 2, 'unresolved_var': 1}`

**Z GPU logů:**
- 568 kandidátů × 8 rolloutů ≈ **4544 Qwen episod**
- grpo_ok (prošlo gate): 65 (11.4%)

**Přijaté úlohy — rollout profil:**
- full_success_rate: min=0.125 max=0.875 avg=0.3949 (0%=0, partial=44)
- unique_rewards: `{2: 39, 3: 5}`
- quality tiers: `{'frontier': 44}`

**Odmítnuté s rollout_signal (JSONL):**
- n=106
- GRPO sub-důvody: `{'all_same_reward': 97, 'no_full_success': 9, 'skipped': 6}`
- full_success: 0%=105, partial=0, 100%=1
- **would pass if win=0:** 9 z 106 rejectů

### Worker `gpu0`
- progress: 11/13 (iter 30, status=running)
- last batch: grpo_ok=1/5, rate=0.073, rejects=`low_grpo_signal_prediction:4`
- accepted: 11

### Worker `gpu1`
- progress: 11/13 (iter 30, status=running)
- last batch: grpo_ok=1/5, rate=0.073, rejects=`low_grpo_signal_prediction:4`
- accepted: 11

### Worker `gpu2`
- progress: 13/13 (iter ?, status=complete)
- last batch: grpo_ok=1/5, rate=0.104, rejects=`low_grpo_signal_prediction:4`
- accepted: 13

### Worker `gpu3`
- progress: 9/13 (iter 29, status=running)
- last batch: grpo_ok=2/5, rate=0.062, rejects=`low_grpo_signal_prediction:3, best_of_n_not_selected:1`
- accepted: 9

## Doporučení (aligned s GPT analýzou)

| Akce | Priorita |
|------|----------|
| Zastavit dlouhé běhy na 50/GPU s `win1` | okamžitě |
| Stage 3 generovat s `ROLLOUT_REQUIRE_ACHIEVABLE_WIN=0` | vysoká |
| Ukládat rejected JSONL + rollout vektory u všech běhů | vysoká |
| Rozdělit `low_grpo_signal_prediction` na sub-důvody v kódu | střední |
| Diagnostický pilot 20–30 kandidátů offline | střední |
| vLLM batched screening (4 rollout cheap → 8 rollout top-k) | později |
| Nepřecházet na Stage 4 dokud Stage 3 win=0 nedá objem | vysoká |

### Odhad nákladů (win1, Stage 3)

Při ~3.6% accept a 8 rolloutech/kandidát:
- 50 accepted ≈ **~1400 kandidátů** ≈ **~11 000 Qwen rolloutů** na GPU
- 4 GPU × 50 = 200 accepted → **~44 000 rolloutů** celkem

S `win=0` (~12% accept): stejný cíl **~3× levněji**.
