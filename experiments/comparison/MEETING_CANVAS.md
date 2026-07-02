# NESTFUL / MT-GRPO — meeting canvas

> **Vizuální verze:** otevři [nestful-meeting.canvas.tsx](file:///C:/Users/%C5%A0unka/.cursor/projects/c-Users-unka-Documents-GitHub-Tool-R0/canvases/nestful-meeting.canvas.tsx) vedle chatu (grafy, paper tabulky, W&B-style N+1 eval).

## 1.

- **Žádný fine-tuning zatím nepřekonal baseline** ve Full Acc ani Win Rate (Direct i ReAct).
- **Strict/minimal reward** v pozdějších stage **silně degraduje ReAct** (Win 54.4% → 32.5% u s4e2).
- **Partial reward** kolaps **výrazně zmírňuje** — nejlepší checkpoint **partial s1e4 ReAct Win 54.3%** ≈ baseline 54.4%.
- Full Acc zůstává u všech běhů **~0** (ReAct) resp. **pod baseline** (Direct); F1 Func sám o sobě úspěch neindikuje.

### Paper reference (NESTFUL Table 1, arXiv v3)

**Direct (one-shot ICL, Table 1):** GPT-4o Win 59%, DeepSeek 43%, Hammer 16%, Llama-8B 6%, Mixtral 7%.

**ReAct (zero-shot agent, samostatná tabulka — GPT-4o v ní není):**

| model | Direct Win | ReAct Win |
|---|---:|---:|
| DeepSeek-V3 | 43% | 46% |
| Mixtral-8x22B | 7% | 30% |
| Hammer2.0-7b | 16% | 7% |
| AgentLM-13B | 0% | 0% |

Náš Qwen3-4B: Direct Win **29.2%**, ReAct Win **54.4%**. Direct je srovnatelný s paperem; ReAct **není** stejný setup jako paper ReAct agent.

---

## 2. Direct výsledky

| checkpoint | Full Acc | Win Rate | závěr |
|---|---:|---:|---|
| baseline | 16.9% | 29.2% | referenční bod |
| minimal/strict s4e2 | 15.2% | 24.3% | mírný pokles obou metrik |
| partial s1e4 | 16.3% | 27.4% | blízko baseline, stále pod ním |
| partial s4e1 | 16.4% | 27.4% | stejně — bez zlepšení nad baseline |

---

## 3. ReAct výsledky

| checkpoint | Full Acc | Win Rate | závěr |
|---|---:|---:|---|
| baseline | 0.0% | **54.4%** | referenční bod |
| minimal/strict s4e2 | 0.1% | **32.5%** | kolaps (−21.9 pp Win) |
| partial s1e4 | 0.0% | **54.3%** | prakticky na baseline (−0.1 pp) |
| partial s4e1 | 0.1% | 45.0% | lepší než strict, ale −9.4 pp vs baseline |

---

## 4. N+1 curriculum eval vs full NESTFUL transfer

Checkpointy s lokálním N+1 eval i full eval na celém NESTFUL benchmarku (1861 úloh). **Win Rate** = oficiální metrika z final eval (ReAct — model volá nástroje po krocích). Zdroj: `nplus1_vs_full_transfer.csv`.

| experiment | checkpoint | stage | epoch | Win Rate |
|---|---|---:|---:|---:|
| **baseline** | — | — | — | **54.4%** |
| mtgrpo_partial | s1_e4 | 1 | 4 | **54.3%** |
| mtgrpo_partial | s2_e2 | 2 | 2 | 53.3% |
| mtgrpo_partial | s3_e2 | 3 | 2 | 47.9% |
| mtgrpo_partial | s4_e1 | 4 | 1 | 45.0% |
| mtgrpo_minimal | s1_e4 | 1 | 4 | 53.2% |
| mtgrpo_minimal | s2_e4 | 2 | 4 | 51.3% |
| mtgrpo_minimal | curriculum s4e2 | 4 | 2 | 32.5% |

**Interpretace**

- Lokální N+1 metriky během tréninku **neodráží spolehlivě** výsledek na full NESTFUL.
- Nejlepší checkpoint = **partial s1e4 (54.3%)** ≈ baseline 54.4%; pokračování curriculum Win snižuje (s4e1: 45.0%).
- Checkpoint vybírat podle **full eval Win Rate**, ne podle lokálního tréninkového skóre.

---

## 5. Tréninkový reward — strict vs partial

Oba rewardy vycházejí **jen z gold trace + gold answer** v tréninkových datech — ne z Win Rate. Evaluace na NESTFUL benchmarku je u obou experimentů stejná.

### Strict / minimal reward — `reward.py` · binární gold-trace reward

**R ∈ {0, 1}.** Reward je all-or-nothing: pozitivní reward dostane jen trajektorie, která projde strict gold-trace kontrolou. Prakticky to znamená shodu s gold trajektorií v počtu kroků, pořadí/názvech tool callů, argumentové struktuře, observations/execution výsledcích a finální odpovědi podle implementace v `reward.py`.

Správná odpověď dosažená jinou než gold-like cestou se v této variantě nebere jako plný úspěch; solution-equivalent success je pouze eval-only, ne training reward.

Clipped / neparsovatelný rollout → 0.

### Partial / graded reward — `partial_reward.py` · graded reward [0, 1]

**R ∈ [0, 1].** Reward dává dílčí kredit za jednotlivé pozice v gold trajektorii a bonus za finální odpověď. Chybějící nebo špatný call nezruší celou epizodu, jen sníží průměrné per-turn skóre.

Per-turn skóre zohledňuje:
- správný název toolu
- správnou argumentovou strukturu
- exekuční / observation shodu podle implementace

Epizodický reward kombinuje průměrné per-turn skóre a final_answer bonus:

```
R = clip(0.7 · mean(turn_score) + 0.3 · 1{final_answer_pass}, 0, 1)
```

Perfektní epizoda = 1.0.

---

## 6. Navržený směr — execution-based reward

**Cíl:** reward více zarovnat s official Win Rate, ne jen s podobností ke gold trace.

Navržený směr:
- rewardovat správnou odpověď odvozenou z vykonaných tool callů
- rewardovat exekutovatelnou trajektorii
- rewardovat validní reference mezi tool cally
- ponechat malý signál za podobnost s gold trace
- silně penalizovat `no_tool_call`
- silně penalizovat `too_few_calls`, pokud model nedosáhl správné odpovědi
- penalizovat předčasné ukončení ReAct trajektorie

**Cílový reward** *(návrh, ještě neimplementováno)*:

```
R = 0.45 · tool_final_answer_pass
  + 0.20 · executable_trajectory
  + 0.15 · tool_use_completeness
  + 0.10 · valid_references
  + 0.10 · small_gold_trace_progress
```

**Proč zrovna takto**

| složka | váha | význam |
|---|---:|---|
| `tool_final_answer_pass` | 0.45 | hlavní cíl: správná odpověď odvozená přes vykonané tool cally |
| `executable_trajectory` | 0.20 | model musí produkovat spustitelné tool cally |
| `tool_use_completeness` | 0.15 | brání `no_tool_call`, `too_few_calls` a předčasnému ukončení |
| `valid_references` | 0.10 | hlídá nested závislosti mezi cally |
| `small_gold_trace_progress` | 0.10 | jen slabý pomocný signál za podobnost s gold trace |