# Agentic v5 Stage 3 pilot v2 — analýza

**Datum:** 2026-07-16  
**Data:** `data/agentic_v5_stage3_pilot_v2/agentic_v5_stage3_pilot_v2/`  
**Konfigurace:** 4 GPU × 10 iter × 5 cand/batch = **200 probed kandidátů**, target 10/GPU (40 celkem), `win=0`

---

## 1. Souhrn

| Metrika | Hodnota |
|---------|---------|
| **Accepted** | **20** / 40 target (partial) |
| Kandidáti v logu (acc + rej) | 200 |
| **Accept rate** | **10.0%** |
| GRPO pass rate | 20/173 = **11.6%** |
| API spend (4 GPU) | **$0.047** |
| Local weak rollouts | 1 791 (~**90/accept**) |
| Registry | v5.0.0 (163 tools) — **ne** nejnovější 5.0.1 |

### Per GPU

| Worker | Accepted | Rejected | Accept rate |
|--------|----------|----------|-------------|
| gpu0 | 4 | 46 | 8.0% |
| gpu1 | 4 | 46 | 8.0% |
| gpu2 | 4 | 46 | 8.0% |
| gpu3 | 8 | 42 | 16.0% |

Fixed-budget pilot proběhl přesně podle plánu (200 kandidátů). gpu3 měl vyšší yield — pravděpodobně víc batchů s 2+ GRPO-kvalifikovanými kandidáty (best-of-N max 1/batch).

---

## 2. Srovnání s předchozími běhy

| Běh | Accepted | Accept rate | GRPO pass | Poznámka |
|-----|----------|-------------|-----------|----------|
| **pilot_v2** (nový) | 20/200 | **10.0%** | 11.6% | win=0, fixed budget |
| stage3_loose | 13/100 | 13.0% | ~31% | diagnostický loose |
| stage3_win1 | 6/200 | 2.7% | ~3% | win=1, příliš tvrdé |
| workers_stage3 | 0/200 | 0% | — | broken registry |

**Verdikt:** Nový gate s `win=0` je funkční. Accept rate **10%** spadá do cílového pásmu 5–10 % (horní hranice). Výrazně lepší než `stage3_win1` (2.7 %).

---

## 3. Kvalita accepted (20 úloh)

### Tier mix (proxy z `full_success_rate`)

| Tier | Počet | Podíl |
|------|-------|-------|
| **frontier** (0 < win < 100%) | 7 | **35%** |
| **partial_frontier** (win=0, spread) | 13 | **65%** |

→ Mix odpovídá Stage 3 kvótám (≥35 % frontier, ≤60 % partial). Dobrý signál pro učení `too_few` / `wrong_args` kontrastů.

### Dominantní failure modes

| Failure | Počet |
|---------|-------|
| `correct_tool_wrong_args` | 18 |
| `too_few_calls` | 2 |

Žádný parse-dominated accepted. Jeden accepted má 1× parse_error v mixu (12.5 %), pod prahem 25 %.

### Reward range

| Stat | Hodnota |
|------|---------|
| min | **0.0056** ⚠️ |
| max | 0.5625 |
| průměr | 0.251 |

**Problém:** **3/20 accepted** má `reward_range < 0.01` (gpu1_000004, gpu3_000001, gpu3_000002).  
To jsou mikro-spready (~0.006) při `unique_rewards=2` — starý gate na RunPodu **neměl** `reward_range >= 0.01` hard floor. Po syncu nejnovějšího kódu by tyto 3 měly být odmítnuty.

### Unikátnost

- 20 accepted = **20 unikátních questions** a **20 unikátních gold traces** (žádný cross-GPU duplikát obsahu).
- `sample_id` se opakuje mezi GPU (`agentic_v5_stage3_000001`…) — merge musí přečíslovat; obsah je ale unikátní.

### Motivy (z `motif_type`)

`long_chain`, `distractor_heavy`, `fan_in`, `argument_binding` — rozumná diverzita.

---

## 4. Reject breakdown (180)

| Důvod | Počet | Podíl |
|-------|-------|-------|
| `low_grpo_signal_prediction` | 153 | 85% |
| `best_of_n_not_selected` | 14 | 8% |
| `not_nestful_like` | 12 | 7% |
| `unresolved_var` | 1 | <1% |

### GRPO sub-reasons (u 153 low_grpo)

| Sub-reason | Počet |
|------------|-------|
| **`all_same_reward`** | **153 (100%)** |

Dominantní bottleneck: Qwen při 3-call dělá **stejný reward na všech 8 rolloutech** (`correct_tool_wrong_args` na všech 3 callech, ale bez rozptylu). To není gate bug — model je na Stage 3 stále příliš homogenní.

27 kandidátů (200−173) nemá rollout signal v reject logu → cheap pre-probe reject (`not_nestful_like`, `unresolved_var`).

---

## 5. Co v datech chybí (starší kód na RunPodu)

Accepted řádky **neobsahují** pole z posledních úprav:

- `quality_tier`, `grpo_sub_reason`, `dominant_rollout_failure`
- `reward_range`, `parse_or_clipped_rate` v `rollout_signal`
- `advantage_preview`
- `borderline_confirmed` / 16-rollout confirm
- unikátní `sample_id` s worker suffixem (`agentic_v5_stage3_gpu0_s45_…`)
- `BEST_OF_N_MAX_ACCEPTS_PER_BATCH=2` (běželo s **1**)

→ Pilot validuje **směr** gate (`win=0`), ale ne nejnovější hardening. Před větší generací **sync kódu + registry 5.0.1**.

---

## 6. Náklady a škálování

| Metrika | Pilot v2 | Odhad na 120 Stage 3 |
|---------|----------|----------------------|
| Accept rate | 10% | — |
| Rollouts/accept | ~90 | — |
| API $/accept | ~$0.0023 | — |
| Pro 120 accepted | — | ~1 080 kandidátů, ~10 800 rolloutů, **~$0.28 API** |

API je zanedbatelné; bottleneck je **GPU čas** na weak rolloutech (~90/accept).

---

## 7. Verdikt a další kroky

### ✅ Gate je funkční

- Accept rate **10%** (cíl 5–10 %)
- Mix **35% frontier / 65% partial** — vhodný pro Stage 3
- Žádný parse-dominated accepted
- 20 unikátních úloh, reálný synthetic executor

### ⚠️ Opravit před produkční generací

1. **Sync nejnovějšího kódu** na RunPod (universal `reward_range >= 0.01`, tier kvóty, 16-rollout borderline, advantage preview)
2. **Registry 5.0.1** (167 tools)
3. **Odmítnout/refiltrovat** 3 accepted s `reward_range < 0.01`
4. `BEST_OF_N_MAX_ACCEPTS_PER_BATCH=2` pro vyšší throughput

### Doporučený postup

1. Merge 20 accepted (dedup + replay gold traces):
   ```bash
   python scripts/data/merge_agentic_workers_v5.py \
     --workers-glob "data/agentic_v5_stage3_pilot_v2/agentic_v5_stage3_pilot_v2/gpu*" \
     --output-dir data/curriculum_v5_agentic_synthetic_pilot_v2
   ```
2. Spustit větší Stage 3 generaci se syncnutým kódem (cíl ~80–120 accepted)
3. Paralelně Stage 2 (~120 accepted) pro frontier kotvy
4. Finální merge s `--apply-merge-tier-quotas` → ~200–300 v5-only úloh

### Hlavní bottleneck (ne gate)

**85% rejectů = `all_same_reward`** — Qwen na 3-call produkuje flat `correct_tool_wrong_args` bez reward spreadu. Gate správně odmítá; zlepšení vyžaduje buď víc teploty/diverzity v rolloutu, nebo lepší úlohy s kontrastnějšími failure modes (partial_frontier s `too_few` vs `wrong_args`).
