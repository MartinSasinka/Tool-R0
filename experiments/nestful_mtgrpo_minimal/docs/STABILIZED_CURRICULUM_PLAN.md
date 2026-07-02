# Stabilized Curriculum Plan

Stabilizace NESTFUL / MT-GRPO pipeline na základě poslední analýzy. Cílem je
**čistší data**, **stabilnější curriculum training** a **prompt hardening** proti
`no_tool_call` / `too_few_calls`.

> **Reward se NEMĚNÍ.** `reward.py`, `partial_reward.py` ani `reward.*` v configu
> nejsou touto změnou dotčeny. Tady připravujeme jen proces; reward experimenty
> přijdou až následně. Žádné nové prompt-ablation ani drahé eval běhy nejsou
> součástí těchto úprav.

---

## Co se změnilo (přehled)

| Oblast | Soubor | Stručně |
|--------|--------|---------|
| Čistá data | `experiments/data/prepare_clean_training_set.py` | validace + bezpečné opravy stage 1–6 → `data/clean_curriculum/` |
| Mixed replay | `data.py` (`load_tasks_mixed`), `run.py` (`mode_train`), `config.yaml` | stage N trénuje na váženém mixu stage 1..N |
| Init z checkpointu | `run_curriculum.sh` (`INIT_FROM`) | `baseline` (base model) nebo `checkpoint` (vyžaduje `CHECKPOINT_IN`) |
| Stabilní profil | `run_curriculum.sh` (`stabilized_curriculum`) | clean data + mixed replay + nižší LR + vyšší KL + per-epoch Win + early stop |
| Checkpoint sidecars | `grpo_train.py` | `config_used.{json,yaml}`, `trainer_state.json`, `wandb_run_id.txt` u každého adapteru |
| Per-epoch validace | `run.py` (`mode_val_eval`), `config.yaml` (`validation`) | oficiální ReAct Win na validačním setu → `metrics_epoch_<E>.json` |
| Early stopping | `run_curriculum.sh` | stop dle `react_win_rate` (patience / min_delta) + `best_react_win_adapter` |
| Prompt hardening | `prompt.py` | anti-no-tool / anti-early-finish / anti-mental / exact-arg pravidla |

Vše žije ve `nestful_mtgrpo_minimal/` a **dědí se do `nestful_mtgrpo_partial/`**
(tenký wrapper, který `exec`-uje sdílený `run_curriculum.sh` a deleguje na sdílený
`run.py`; do wrapperu byl jen doplněn `val_eval` mód).

---

## 1. Clean / repaired training set — a proč nevyhazujeme těžké příklady

`experiments/data/prepare_clean_training_set.py` projde `data/filtered_toolr0_synthetic/epoch_{1..6}_{N}call.jsonl`
a vytvoří čistou variantu v `data/clean_curriculum/`.

**Odstraní se POUZE objektivně vadné / neexekuovatelné příklady:**
neparsovatelný JSON; chybějící `input`/`tools`/`output`/`gold_answer`; gold trace
není seznam callů (nebo je prázdný); tool call volá funkci mimo `tools`; argument,
který schéma tooly nedeklaruje; chybějící required argumenty; `$var` reference na
neexistující/budoucí call nebo špatné output pole; selhání gold exekuce (když je
IBM executor dostupný); `gold_answer` s unresolved `$var…`, který nejde dopočítat.

**Bezpečné opravy:** `tools`/`output` uložené jako JSON-string → naparsovat na
nativní objekty; `gold_answer` s unresolved `$var…` → nahradit konkrétní hodnotou
z exekuce gold trace.

**Nikdy se neodstraňují** těžké / dlouhé / multi-call úlohy, úlohy, které model
neumí, ani úlohy s alternativní validní cestou. Filtrujeme *vady dat*, ne *obtížnost* —
jinak bychom uměle nadhodnotili metriky a zúžili distribuci.

Výstupy: `epoch_{N}_{N}call.jsonl`, `CLEANING_REPORT.md`, `removed_examples.csv`,
`repaired_examples.csv`, `validation_summary.json`.

```bash
# z repo rootu
python experiments/data/prepare_clean_training_set.py --stages 1,2,3,4,5,6
# bez IBM executoru (jen strukturální kontroly):
python experiments/data/prepare_clean_training_set.py --no_exec
```

## 2. Mixed curriculum replay — a proč

Klasické curriculum vystaví stage N **jen** N-call úlohám, takže model zapomíná
jednodušší dovednosti (catastrophic forgetting) a přechody mezi stage jsou tvrdé.
**Mixed replay** trénuje stage N na váženém mixu stage 1..N → plynulejší přechody,
zachování dříve naučeného a stabilnější gradienty.

- `data.py::load_tasks_mixed(stage_files, weights, max_tasks, seed)` — bez
  `num_calls` filtrace (soubory už jsou per-stage), váhové samplování (rovnoměrné,
  když váhy chybí), vrací i per-stage statistiky.
- `run.py::mode_train` — při `data.mixed_replay=true` použije `load_tasks_mixed`
  a loguje počty/stage, váhy a batchů/epochu.
- Config: `data.mixed_replay`, `data.mixed_stage_files`, `data.replay_weights`.
- `run_curriculum.sh` pro stage N sestaví seznam `epoch_1..N` z `DATA_BASE` a předá
  ho přes `--override` (env `CURRICULUM_MIXED_REPLAY=1`, váhy `CURRICULUM_REPLAY_WEIGHTS`).

## 3. Init: baseline vs. checkpoint

`INIT_FROM` v `run_curriculum.sh`:
- `baseline` (default) — stage 1 začíná z base modelu (případné `CHECKPOINT_IN`
  se ignoruje a zaloguje).
- `checkpoint` — vyžaduje `CHECKPOINT_IN` ukazující na validní adapter dir
  (kontrola `adapter_config.json`); jinak skript skončí s jasnou chybou. Zdroj
  inicializace se loguje a propisuje i do `trainer_state.json`.

## 4. Profil `stabilized_curriculum`

`PROFILE=stabilized_curriculum` nastaví (vše přepsatelné env proměnnou):
- `STAGES="1 2 3 4"`, `DATA_BASE=<minimal>/data/clean_curriculum`
- `CURRICULUM_MIXED_REPLAY=1`, `EVAL_EVERY_EPOCH=1`
- early stop: `EARLY_STOP_METRIC=react_win_rate`, `EARLY_STOP_PATIENCE=1`, `EARLY_STOP_MIN_DELTA=0.005`
- **LR 0.5×** (`training.learning_rate=0.5e-6`) a **KL 2×** (`training.kl_beta=0.04`)
  oproti defaultům v `config.yaml` (1.0e-6 / 0.02) — nižší LR a silnější KR-ke-referenci
  brání policy driftu, který v analýze degradoval pozdější stage.

Profily `pilot` a `curriculum` zůstávají **beze změny**.

## 5. Checkpoint po každé epoše + sidecars

`grpo_train.py` u každého `adapter_epoch_<E>` zapíše vedle adapteru:
`config_used.json` / `config_used.yaml` (přesný resolved config),
`trainer_state.json` (stage, epoch, lr, kl, num_generations, global_step,
`init_checkpoint`, `resumed_from_checkpoint`, `mixed_replay`) a `wandb_run_id.txt`
(když je W&B aktivní). To umožní audit i resume libovolného checkpointu.

## 6. Per-epoch validation ReAct Win + early stopping

- `run.py --mode val_eval` spustí ReAct rollout + **oficiální** NESTFUL Win na
  validačním setu (default **plný NESTFUL**; `validation.subset_size>0` →
  deterministický subset uložený do `validation_subset_ids.json`, identický pro
  baseline i všechny checkpointy) a zapíše `metrics_epoch_<E>.json` s `react_win_rate`.
- `run_curriculum.sh` po každé epoše spustí `val_eval`, drží globálně nejlepší
  adapter v `outputs/<run>/best_react_win_adapter` (+ `best_meta.json`) a aplikuje
  **early stopping** dle `react_win_rate`: patience se resetuje jen při zlepšení
  `>= min_delta`.

**Proč ReAct Win, ne `strict_gold_trace_pass`?** Analýza ukázala, že trénink
optimalizoval jinou veličinu než finální cíl (Win Rate). Gate na
`strict_gold_trace_pass` zůstává pro postup mezi stage, ale výběr nejlepšího
checkpointu a early stopping řídí **validační ReAct Win** = metrika, na které nám
reálně záleží.

## 7. Prompt hardening

`prompt.py::SYSTEM_PROMPT` nově explicitně zakazuje:
- řešit cokoli „v hlavě“ bez tool callu (anti-mental),
- prázdný `<tool_call_answer>[]</tool_call_answer>` na prvním turnu / před prvním
  reálným `<tool_response>` (anti-`no_tool_call`),
- předčasné ukončení u mezivýsledku — pokračuj dalším callem (anti-`too_few_calls`),
a vyžaduje přesně jeden neprázdný call per turn, exaktní názvy argumentů dle schématu,
label u každého callu a referenční syntaxi `$varN.result$` (basic) /
`$varN.output_0$` (complex). Tato pravidla jsou konzistentní s `_EVAL_HARDENING`,
který se jako dříve přidává jen při evaluaci (formát/tagy se nemění).

> Pozn.: oficiální NESTFUL scorer (Win) vyžaduje referenční pole, tj. `$varN.result$`
> nebo `$varN.output_0$`. Training executor pole ignoruje, ale prompt učí obě
> varianty, takže výstupy jsou skórovatelné.

---

## Jak to spustit

### Krok 0 — připravit clean data (jednorázově)
```bash
python experiments/data/prepare_clean_training_set.py --stages 1,2,3,4,5,6
```

### A) Stabilized curriculum z baseline (strict reward)
```bash
cd experiments/nestful_mtgrpo_minimal
CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 \
  PROFILE=stabilized_curriculum INIT_FROM=baseline STAGES="1 2 3 4" \
  bash run_curriculum.sh
```

### B) Stabilized curriculum z existujícího checkpointu
```bash
cd experiments/nestful_mtgrpo_minimal
CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 \
  PROFILE=stabilized_curriculum INIT_FROM=checkpoint \
  CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_1 \
  STAGES="3 4" \
  bash run_curriculum.sh
```

### C) Přes partial wrapper (partial reward na tréninku, eval zůstává strict+oficiální)
```bash
cd experiments/nestful_mtgrpo_partial
CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 \
  PROFILE=stabilized_curriculum INIT_FROM=baseline STAGES="1 2 3 4" \
  bash run_curriculum.sh
```
Wrapper přebírá `DATA_BASE` z profilu (sibling `clean_curriculum`) a deleguje na
sdílený `run.py`/`run_curriculum.sh`; reward zůstává nezměněn touto úpravou.

### Užitečné přepínače (env)
| Env | Default (stabilized) | Význam |
|-----|----------------------|--------|
| `INIT_FROM` | `baseline` | `baseline` \| `checkpoint` |
| `CHECKPOINT_IN` | — | adapter dir pro `INIT_FROM=checkpoint` |
| `CURRICULUM_MIXED_REPLAY` | `1` | zapnout mixed replay |
| `CURRICULUM_REPLAY_WEIGHTS` | uniform | např. `"2.0,1.0,1.0,1.0"` |
| `EVAL_EVERY_EPOCH` | `1` | per-epoch `val_eval` |
| `EARLY_STOP_PATIENCE` | `1` | počet evalů bez zlepšení před stopem |
| `EARLY_STOP_MIN_DELTA` | `0.005` | min. zlepšení ReAct Win pro reset patience |
| `VAL_SUBSET_SIZE` | `0` | `0` = plný NESTFUL; `>0` = deterministický subset |
| `STABILIZED_LR` / `STABILIZED_KL` | `0.5e-6` / `0.04` | override LR / KL (reward netýká) |

Výstup nejlepšího modelu (dle validační ReAct Win) je v
`outputs/<run>/best_react_win_adapter` — ten použij pro `final_eval` / nasazení.

---

## Sanity checks

`experiments/nestful_mtgrpo_minimal/tests/test_stabilized_pipeline.py` ověřuje
(bez GPU běhu): integritu clean dat, validní tools/args/refs, mixed replay loader,
prompt pravidla, konzistenci eval/train promptu a logiku výběru best checkpointu.

```bash
cd experiments/nestful_mtgrpo_minimal
python -m pytest tests/test_stabilized_pipeline.py -q
```
