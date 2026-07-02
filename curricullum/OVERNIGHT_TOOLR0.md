# Tool-R0 synthetic curriculum → NESTFUL eval

Cíl: trénovat ve **stejném formátu jako NESTFUL multiturn eval** (`<think>` + `<tool_call_answer>` + IBM exekuce), bez leaku z benchmarku.

## Pipeline (3 fáze)

| Fáze | Co dělá | Výstup |
|------|---------|--------|
| **Generace** (OpenRouter) | Nové úlohy jako JSON (`input`, `tools`, `output`, `gold_answer`) | `curricullum/data/filtered_toolr0_synthetic/` |
| **Příprava tréninku** | Multi-turn prompt + gold tagy + simulované `<tool_response>` | GRPO dataset v `prepare_dataset_toolr0.py` |
| **Eval** | Stejný loop jako `nestful_evaluation/run.py` | held-out test set |

**Důležité:** DeepSeek generuje jen JSON. Tagy `<think>` a `<tool_call_answer>` se přidávají až při přípravě tréninku (`prepare_dataset_toolr0.py`) — stejně jako eval očekává od modelu.

## 1. Build dataset (Windows / lokální PC)

Předpoklady:
- `OPENROUTER_API_KEY` v `.env` v kořeni repa
- NESTFUL data pro tool schémata: `eval/data/NESTFUL-main/data_v2/nestful_data.jsonl`
- IBM verify (doporučeno): `nestful_repo` v kořeni repa

```powershell
cd C:\Users\Šunka\Documents\GitHub\Tool-R0

# IBM helpers (jednorázově)
git clone --depth 1 https://github.com/IBM/NESTFUL.git nestful_repo

# volitelné přepsání defaultů
$env:MODEL = "deepseek/deepseek-v4-flash"   # or "deepseek-ai/DeepSeek-V4-Flash" (auto-mapped)
$env:N_GENERATE = "1500"      # kandidátů na epochu
$env:N_FINAL = "500"          # finálních řádků na stage
$env:MAX_STAGES = "3"         # 1 / 2 / 3 call
$env:PARALLEL_WORKERS = "16"
$env:USE_EXECUTOR = "1"       # IBM verify ve step2

python curricullum/run_generate_toolr0_curriculum.py
```

Výstup:
- `curricullum/data/filtered_toolr0_synthetic/epoch_1_1call.jsonl` (500)
- `curricullum/data/filtered_toolr0_synthetic/epoch_2_2call.jsonl` (500)
- `curricullum/data/filtered_toolr0_synthetic/epoch_3_3call.jsonl` (500)
- `curricullum/data/filtered_toolr0_synthetic/curriculum_toolr0_all.jsonl`

Rychlý test (méně API volání):

```powershell
$env:N_GENERATE = "80"
$env:N_FINAL = "10"
$env:MAX_STAGES = "1"
python curricullum/run_generate_toolr0_curriculum.py
```

## 2. Train + eval na DGX (jeden příkaz)

```bash
cd /path/to/Tool-R0

# full: 3 stage train (4 GPU) + NESTFUL eval (baseline + finální stage3)
bash curricullum/run_train_and_eval_toolr0_dgx.sh

# pilot eval po tréninku
MAX_TASKS=50 NUM_ROLLOUTS=2 bash curricullum/run_train_and_eval_toolr0_dgx.sh

# jen eval (trénink už hotový)
SKIP_TRAIN=1 bash curricullum/run_train_and_eval_toolr0_dgx.sh
```

Výstupy eval: `curricullum/evaluation/results_toolr0/`  
Profily: `curriculum_baseline`, `curriculum_stage3_3call` (stejný driver jako `nestful_evaluation/run.py`).

Volitelně: `CUDA_DEVICES=0,1,2,4` pokud GPU 3 drží display.

---

## 2b. Train only (GPU, typicky DGX)

```bash
export CUDA_VISIBLE_DEVICES=0
export OVERWRITE=1
export MAX_STAGES=3
export TRAINING_FORMAT=tool_r0

bash curricullum/run_train_toolr0.sh
# checkpoints: curricullum/checkpoints/qwen3_4b_lora_grpo_toolr0/
```

## 3. Eval

Pouze held-out NESTFUL test — ne trénovací syntetika.

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_ROLLOUTS=8

python curricullum/evaluation/run_eval.py --only stage3_3call_toolr0
```

## Klíčové soubory

| Soubor | Účel |
|--------|------|
| `curricullum/run_generate_toolr0_curriculum.py` | Orchestrátor generace (Windows OK) |
| `curricullum/data/step1_gen_candidates.py` | OpenRouter generace |
| `curricullum/data/step2_verify_candidates.py` | Verify + `--use_executor` |
| `curricullum/data/step3_select_curriculum.py` | Výběr 500/epoch |
| `curricullum/train/prepare_dataset_toolr0.py` | Multi-turn + think + tool_response |
| `curricullum/train/rewards_toolr0_exec.py` | Reward: tagy + IBM exec |
| `curricullum/train/configs/qwen3_4b_lora_grpo_toolr0.yaml` | 3 stage config |
| `curricullum/run_train_toolr0.sh` | GRPO trénink |

## Co jsme odstranili

- `build_toolr0_dataset.py` — slice reálného NESTFUL benchmarku (data leakage)
- Starý JSON-plán curriculum (`run_generate_curriculum.*`, `filtered/`, 4-call epoch)
