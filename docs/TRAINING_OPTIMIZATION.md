# Optimalizace tréninku Tool-R0 pro Qwen3.5-4B

*Technická studie a roadmap*

---

## Abstrakt

Tato studie analyzuje paměťová a propustnostní omezení současné konfigurace tréninku Tool-R0 nad modelem `Qwen/Qwen3.5-4B` na hardwaru DGX (3× NVIDIA A100 40 GB využitelné, GPU 3 vyhrazena pro display). Identifikuje tři hlavní problémy: (i) předčasné osekávání generovaných odpovědí kvůli `max_completion_length = 2048`, (ii) trvající nestabilitu DeepSpeed ZeRO-3 s rodinou Qwen3.5, a (iii) nevyužitý 3. GPU během GRPO tréninku. Navrhuje minimálně-invazivní přechod z plného fine-tuningu na LoRA, kvantitativně podložený VRAM rozpočtem, který umožní rozšíření kontextového okna z 2048 na ~8192 tokenů bez OOM. Implementační dopad je odhadnut na řádově deset řádků kódu, neboť TRL `GRPOTrainer` PEFT integraci nativně podporuje a relevantní importy (`get_peft_config`, `get_quantization_config`) jsou v pipeline již přítomny, ale dosud nezapojeny.

---

## 1. Stav DeepSpeed ZeRO-3 pro rodinu Qwen3.5 (duben 2026)

### 1.1 Pozorovaný stav

Současné repozitorní configy ([`configs/deepseed_zero2.yaml`](../configs/deepseed_zero2.yaml), [`configs/deepseed_zero3.yaml`](../configs/deepseed_zero3.yaml)) obsahují explicitní varování proti použití ZeRO-3 s Qwen3.5 s odkazem na [transformers issue #45313](https://github.com/huggingface/transformers/issues/45313). Default v [`run_main.sh`](../run_main.sh) je proto nastaven na `deepseed_zero2_offload.yaml`.

### 1.2 Aktuální problémy v upstream ekosystému

K dnešnímu datu jsou v ekosystému (DeepSpeed, transformers, TRL) stále dokumentovány tři nezávislé třídy problémů ZeRO-3 + Qwen3.5/Qwen-VL:

| Problém | Reference | Dopad |
|---------|-----------|-------|
| Hang při mixed-modality datasetu (`video_grid_thw=None` na podmnožině ranků) | [DeepSpeed #7170](https://github.com/deepspeedai/DeepSpeed/issues/7170) | Trénink nelze dokončit |
| Nesoulad `language_model.*` ↔ `model.language_model.*` při ukládání | [transformers #45313](https://github.com/huggingface/transformers/issues/45313), částečně řešeno v [`grpo_processing.py::fix_checkpoint_for_vllm`](../grpo_processing.py) | Checkpoint nelze nahrát do vLLM bez post-processingu |
| Korupce shape (`torch.Size([0])`) bez `stage3_gather_16bit_weights_on_model_save: true` | [transformers #20082](https://github.com/huggingface/transformers/issues/20082) | OOM nebo MISSING klíče při loadu |

### 1.3 Doporučení

Setrvat na **ZeRO-2 + CPU offload optimizéru** jako defaultní cestě. Migraci na ZeRO-3 odložit do uzavření #45313. Místo paměťové úspory přes ZeRO-3 použít **LoRA**, která řeší stejný problém (sharding optimizer states je u plného FT 32 bitů × 4 B params = 32 GB; u LoRA r=32 je řádově ~400 MB) bez zavádění nových rizik.

---

## 2. VRAM rozpočet a strop kontextového okna

### 2.1 Statická spotřeba

Pro `Qwen/Qwen3.5-4B` (≈ 4.0 × 10⁹ parametrů) lze v bf16 kvantifikovat statickou spotřebu na jeden GPU:

| Komponenta | Plný FT, ZeRO-2 + offload | Plný FT, ZeRO-2 (no offload), 3 GPU | LoRA (r=32), ZeRO-2 (no offload) |
|------------|:---:|:---:|:---:|
| Parametry modelu (bf16) | 8.0 GB | 8.0 GB | 8.0 GB (zmrazené) |
| Gradienty (bf16) | 8.0 GB | 8.0 GB / 3 ≈ 2.7 GB | ~0.1 GB (LoRA) |
| Optimizer states (Adam, fp32 master + m + v) | CPU offload (0 GB GPU) | 32 GB / 3 ≈ 10.7 GB | ~0.4 GB (LoRA) |
| **Statický součet** | **~16 GB** | **~21 GB** | **~8.5 GB** |

Tato čísla vychází z: bf16 = 2 B/param, fp32 = 4 B/param, Adam udržuje master weights + první + druhý moment (12 B/param), gradienty bf16 = 2 B/param. ZeRO-2 sharduje gradienty a optimizer states, parametry replikuje.

### 2.2 Dynamická spotřeba aktivací

GRPO trénink generuje `num_generations = G` rolloutů na prompt délky `L_p`, completion délky `L_c`. Aktivace s gradient checkpointingem zhruba odpovídají $\mathcal{O}(B \cdot G \cdot (L_p + L_c) \cdot d_{\text{model}})$, kde $d_{\text{model}}$ pro Qwen3.5-4B je 2560.

Empiricky pro tuto třídu modelu při $B=1$, $G=2$, $L_c=2048$, gradient checkpointing aktivní, lze očekávat ~10–14 GB aktivací. Při zvýšení $L_c$ na 4096 se aktivace přibližně zdvojnásobí.

### 2.3 Strop kontextu

Volné VRAM = 40 − statický součet:

| Režim | Volné VRAM | Praktický strop `max_completion_length` |
|-------|:---:|:---:|
| Plný FT + Zero2 offload | ~24 GB | **2048** (současný, viz [`run_main.sh:43`](../run_main.sh)) |
| Plný FT + Zero2, 3 GPU | ~19 GB | ~2048 (offload eliminován, ale aktivace dominují) |
| LoRA r=32 + Zero2 | ~31 GB | **8192** (≈ 4× nárůst) |

Klíčové pozorování: omezení kontextu **není dáno modelem**, ale rozpočtem aktivací při plném FT. Přechod na LoRA uvolňuje 23 GB statického, které lze realokovat buď na delší kontext, vyšší `num_generations`, nebo větší `per_device_train_batch_size`.

---

## 3. Distribuce GPU a paralelismus

### 3.1 Současná topologie

Z [`run_main.sh`](../run_main.sh) (řádky 22-33):

- **GPU 0,1** → trénink GRPO (`STEP13_GPUS`, `STEP13_NUM_PROCESSES=2`)
- **GPU 2** → vLLM solver pro step0 a step2 (`STEP0_GPU`, `STEP2_GPUS`)
- **GPU 3** → DGX display (vyloučena ze všech workloadů)

Step1 a step3 (samotný GRPO) tak běží jen na **dvou GPU**, zatímco GPU 2 v té době zahálí — vLLM solver se spouští pouze ve step0 (rollout) a step2 (data generation/verify).

### 3.2 Návrh

Pro fáze step1 a step3 alokovat všechny 3 dostupné GPU:

```
STEP13_GPUS=0,1,2
STEP13_NUM_PROCESSES=3
```

Precedent existuje v [`scripts/resume_iter1_step2_and_solver.sh`](../scripts/resume_iter1_step2_and_solver.sh), které tuto konfiguraci již používá. Přidání 3. GPU k ZeRO-2 sníží podíl optimizer states na GPU z $1/2$ na $1/3$, tedy ~3.6 GB úspora per GPU u plného FT (irelevantní u LoRA, kde optimizer states jsou už malé).

### 3.3 Paralelismus

Použitý paralelismus zůstává **data-parallel s ZeRO-2 partition** (ne tensor-parallel). Pro 4B model na A100 40GB to není limitující — model se vejde na jeden GPU. Kdyby byl v budoucnu cíl Qwen3.5-7B nebo větší, přicházela by v úvahu Megatron-style TP přes vLLM `tensor_parallel_size`, což však vyžaduje odlišnou knihovnu a větší zásah.

---

## 4. Gradient accumulation a effective batch size

### 4.1 Současné nastavení

Z [`run_step1.sh:27-29`](../run_step1.sh) a [`run_step3.sh:27-29`](../run_step3.sh):

$$
B_{\text{eff}} = B_{\text{device}} \cdot N_{\text{accum}} \cdot N_{\text{GPU}} \cdot G_{\text{rollouts}}
= 1 \cdot 8 \cdot 2 \cdot 2 = 32
$$

### 4.2 Návrh

Po LoRA přechodu a GPU rozšíření na 3 procesy lze dosáhnout řádově vyšší effective batch:

$$
B_{\text{eff}}^{\text{nov}\acute{\text{y}}} = 1 \cdot 16 \cdot 3 \cdot 4 = 192
$$

Větší effective batch (192) stabilizuje GRPO advantage estimate — relativní výhoda je počítaná napříč $G$ rollouty pro stejný prompt, takže $G = 4$ místo $G = 2$ poskytuje statisticky robustnější relativní rank (variance ∝ $1/G$). Zvýšení `gradient_accumulation_steps` z 8 na 16 dále zlepšuje estimate gradientu při stejné per-step VRAM.

---

## 5. LoRA integrace

### 5.1 Princip

TRL `GRPOTrainer` přijímá volitelný argument `peft_config` (instance `peft.LoraConfig`). Pokud je předán, trainer obalí model jako `PeftModel`, zmrazí původní váhy, zavede nízko-hodnostní adaptéry a v save fázi ukládá pouze adapter weights (~50–100 MB pro r=32).

Pro Qwen3.5-4B s LoRA r=32 na všech projection vrstvách (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) je počet trénovatelných parametrů přibližně:

$$
N_{\text{LoRA}} \approx 7 \cdot 2 \cdot r \cdot d_{\text{model}} \cdot N_{\text{layers}}
= 7 \cdot 2 \cdot 32 \cdot 2560 \cdot 36 \approx 41 \text{ M}
$$

Tedy ~1 % parametrů modelu — paměťová úspora optimizer states je řádově $100 \times$.

### 5.2 Stav repozitáře

[`step1_generator.py:11-20`](../step1_generator.py) a [`step3_solver.py:12-21`](../step3_solver.py) již **importují** `get_peft_config`, `get_quantization_config`, `get_kbit_device_map`, ale ani jeden symbol není dále volán. Volání `GRPOTrainer(...)` v [`step1_generator.py:311-318`](../step1_generator.py) a [`step3_solver.py:96-109`](../step3_solver.py) parametr `peft_config` neuvádí.

### 5.3 Navrhovaná změna v Pythonu

Přidat jediný argument do volání trainer (identicky v obou souborech):

```python
peft_config=get_peft_config(model_args),
```

`get_peft_config` z TRL vrátí `LoraConfig` pokud `model_args.use_peft == True`, jinak `None` (a `GRPOTrainer` pokračuje v plném FT). Tedy zavedení nezpůsobí breaking change — výchozí chování zůstává plný fine-tuning.

### 5.4 Navrhovaná změna v shellech

TRL `ModelConfig` dataclass exposuje vlajky `--use_peft`, `--lora_r`, `--lora_alpha`, `--lora_target_modules` přímo přes `TrlParser`, takže není třeba upravovat Python parser. V [`run_step1.sh`](../run_step1.sh) a [`run_step3.sh`](../run_step3.sh) stačí přidat ENV-konfigurovatelný blok argumentů (obvykle ~5 řádků) a propagovat jej do `accelerate launch`.

Doporučené ENV proměnné a defaulty:

| Proměnná | Default | Význam |
|----------|---------|--------|
| `TOOL_R0_USE_PEFT` | `true` | Zapne LoRA (`false` ponechá plný FT pro kompatibilitu) |
| `TOOL_R0_LORA_R` | `32` | LoRA rank — 32 je standard pro 4B modely; r=64 zvýší kapacitu o 2× |
| `TOOL_R0_LORA_ALPHA` | `64` | Scaling faktor; konvence $\alpha = 2r$ |
| `TOOL_R0_LORA_TARGET_MODULES` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Všechny projection vrstvy attention + MLP |
| `TOOL_R0_LORA_DROPOUT` | `0.05` | Regularizace — Tool-R0 trénuje na malém datasetu (2k vzorků/iter) |

### 5.5 Dopad na pipeline checkpointů

Po LoRA tréninku ukládá `GRPOTrainer.save_model()` pouze adapter weights, nikoliv plný model. To má dva downstream efekty:

1. **vLLM v step2** ([`step2_gen.py`](../step2_gen.py), [`step2_genverify.py`](../step2_genverify.py), [`step2_select_curriculum.py`](../step2_select_curriculum.py)) potřebuje buď (a) `--enable-lora` flag s explicitním adapter cestou, nebo (b) předmerged checkpoint. Volba (b) je jednodušší a zachovává existující kód.

2. **Eval pipeline** ([`eval/run_eval.py`](../eval/run_eval.py)) má stejný požadavek.

Doporučené řešení: rozšířit funkci [`grpo_processing.py::fix_checkpoint_for_vllm`](../grpo_processing.py) o detekci LoRA adapteru (existence `adapter_config.json` v output_dir) a v takovém případě provést `peft.PeftModel.merge_and_unload()` před zápisem konzolidovaného checkpointu. Toto je single-purpose úprava ~15 řádků a zachovává stávající interface volaný z train scripts.

### 5.6 Konvergence LoRA vs. plný FT v RL kontextu

LoRA je v RLHF/GRPO režimu standardní volba (viz Hu et al. 2021, Llama-Guard, Tülu 3 recipes). Empirické pozorování z literatury je, že LoRA dosahuje 95–99 % výkonu plného FT při r ≥ 16 na úkolech instruction-tuning. Pro tool calling, kde se model neučí novou znalost, ale spíše formátový styl a struktury volání, je LoRA očekávatelně neutrální nebo lepší (méně catastrophic forgetting, lépe zachovává irrelevance schopnost).

---

## 6. Roadmap implementace (řazeno dle ROI)

| Pořadí | Úkol | Riziko | Odhadovaný dopad |
|:---:|------|:---:|------|
| 1 | **Krátkodobý kontext fix bez LoRA**: `STEP1/3_MAX_COMPLETION_LENGTH` z 2048 → 3072 a dočasně `num_generations: 2 → 1`. Cíl: ověřit, zda osekávání odpovědí na NESTFUL je hlavní příčina nízkého skóre. | Nízké | Eliminace truncation pro ~80 % NESTFUL případů |
| 2 | **GPU expansion**: nastavit `STEP13_GPUS=0,1,2`, `STEP13_NUM_PROCESSES=3` v [`run_main.sh`](../run_main.sh). | Nízké | +50 % effective batch, nižší podíl optimizer states |
| 3 | **LoRA integrace** (jeden argument do dvou souborů, ENV bloky do dvou shellů — viz §5.3, §5.4). | Střední — nutné ověřit, že se ukládá adapter správně | Uvolní 23 GB VRAM/GPU, umožní `max_completion_length=8192` |
| 4 | **Validační run**: 50 GRPO kroků s LoRA + ZeRO-2, sledovat reward křivku, gradient norm a paměťové špičky. Porovnat s plným FT baseline. | Střední | Důkaz, že LoRA neztrácí konvergenci |
| 5 | **Merge & save** úprava v [`grpo_processing.py`](../grpo_processing.py) — automatický merge LoRA adapteru do plného checkpointu pro downstream vLLM/eval kompatibilitu. | Nízké | Zero-friction přechod pro step2 a eval |
| 6 | **NESTFUL re-run** s `max_completion_length=4096` a LoRA-trénovaným modelem (nový iter). Cíl: kvantifikovat zlepšení vůči baseline 14.31 %. | Nízké | Reálné NESTFUL číslo s validní konfigurací |

---

## 7. Otevřené otázky pro budoucí výzkum

- **Curriculum pro multi-call sekvence**: Současná distribuce v [`step1_generator.py:159-184`](../step1_generator.py) generuje max 3 cally, NESTFUL vyžaduje až 8. Rozšíření distribuce je samostatná změna ~5 řádků, ale interaguje s `max_completion_length` (delší sekvence = delší výstup).

- **Variable reference matching v reward**: Pokud generator produkuje `$var_N.result$`, reward funkce v [`rewards_solver.py`](../rewards_solver.py) musí explicitně rozpoznat tento token jako rovný libovolné gold hodnotě. Bez toho dostává model penalizaci i za korektní výstup.

- **Irrelevance regrese**: Po Tool-R0 finetuningu klesá BFCL irrelevance accuracy z 62 % (Qwen3.5 base) na 2 %. Lze adresovat přidáním negativních příkladů do tréninkového curricula (viz [`step2_select_curriculum.py`](../step2_select_curriculum.py)) — paralelní směr k tomuto dokumentu.

---

## 8. Validační smoke test po implementaci roadmapy

Tato sekce dokumentuje minimální 5-krokový smoke test, který je nutné spustit **před** plným self-play tréninkem (`bash run_main.sh`). Cíl je ověřit, že nová pipeline (migrace na `Qwen/Qwen3-4B-Instruct-2507`, LoRA integrace, auto-merge ve `fix_checkpoint_for_vllm`) funguje end-to-end na izolovaném scope.

### 8.1 Změny realizované v této iteraci roadmapy

| Krok roadmapy | Realizace |
|:---:|------|
| 1, 6 | `STEP1/3_MAX_COMPLETION_LENGTH` 2048 → **3072** v [`run_main.sh`](../run_main.sh), [`run_step1.sh`](../run_step1.sh), [`run_step3.sh`](../run_step3.sh). |
| 2 | `STEP13_GPUS=0,1,2`, `STEP13_NUM_PROCESSES=3` v [`run_main.sh`](../run_main.sh). |
| 3 | `peft_config=get_peft_config(model_args)` v [`step1_generator.py`](../step1_generator.py) a [`step3_solver.py`](../step3_solver.py). ENV → CLI propagace přes `PEFT_ARGS` v shell scriptech. |
| 5 | `_maybe_merge_lora_adapter` jako první krok ve `fix_checkpoint_for_vllm` ([`grpo_processing.py`](../grpo_processing.py)). Adapter side-files se zálohují do `<output_dir>/_lora_adapter/`. |
| — | **Migrace base modelu** `Qwen/Qwen3.5-4B` (VLM, hybridní DeltaNet) → `Qwen/Qwen3-4B-Instruct-2507` (text-only `Qwen3ForCausalLM`). Eliminuje [vLLM #34186](https://github.com/vllm-project/vllm/issues/34186) (silent-zero LoRA) a celou třídu VLM remap problémů. |
| — | TRL bug mitigace v `apply_grpo_peft_workarounds` ([`grpo_processing.py`](../grpo_processing.py)): `gradient_checkpointing_kwargs.use_reentrant=True` ([TRL #3089](https://github.com/huggingface/trl/issues/3089)) a `sync_ref_model=False` ([TRL #3108](https://github.com/huggingface/trl/issues/3108)). |

### 8.2 Spuštění smoke testu

```bash
TOOL_R0_USE_PEFT=true \
TOOL_R0_LORA_R=32 \
STEP13_GPUS=0,1,2 \
STEP13_NUM_PROCESSES=3 \
STEP1_MAX_COMPLETION_LENGTH=3072 \
bash run_step1.sh "Qwen/Qwen3-4B-Instruct-2507" \
    "./qwen3-4b-tool-r0/smoke_iter1_generator" "smoke" 5
```

5 GRPO kroků (`max_steps=5`) na 3 GPU s LoRA r=32. Save se nespustí (default `save_steps=50`), ale `trainer.save_model()` na konci ano — tím proběhne i merge přes `fix_checkpoint_for_vllm`.

### 8.3 Kontrolní body (acceptance criteria)

| # | Kontrolní bod | Jak ověřit | Pokud selže |
|:---:|------|------|------|
| 1 | **Žádný `requires_grad` error** | V `step.log` chybí stack trace `element 0 of tensors does not require grad`. | TRL #3089 mitigace nezapůsobila. Zkontrolovat, že `apply_grpo_peft_workarounds` je volán a že TRL ≥ 0.20. |
| 2 | **LoRA váhy se zapsaly** | `du -h ./qwen3-4b-tool-r0/smoke_iter1_generator/_lora_adapter/adapter_model.safetensors` ≥ 1 MB (typicky ~80 MB pro r=32). | Zkontrolovat [PEFT #2892](https://github.com/huggingface/peft/issues/2892); ověřit, že jedeme na ZeRO-2, nikoli ZeRO-3. |
| 3 | **Plný checkpoint po merge** | `du -h ./qwen3-4b-tool-r0/smoke_iter1_generator/model*.safetensors` celkem ≈ **8 GB** (4B params × 2 B/bf16). | `_maybe_merge_lora_adapter` selhal — zkontrolovat log na řádek `[fix_checkpoint] ERROR: merge_and_unload failed`. |
| 4 | **Správná architektura** | `python -c "import json; print(json.load(open('./qwen3-4b-tool-r0/smoke_iter1_generator/config.json'))['architectures'])"` vrátí `['Qwen3ForCausalLM']`. | Migrace base modelu nezapůsobila — zkontrolovat `TOOL_R0_BASE_MODEL` env i CLI `--model_name_or_path`. |
| 5 | **vLLM načte checkpoint** | `python -c "from vllm import LLM; LLM(model='./qwen3-4b-tool-r0/smoke_iter1_generator', enforce_eager=True, max_model_len=2048)"` doběhne bez `OSError`. | Zkontrolovat, že `_remap_weight_keys` neuhladil keys ke shape mismatchům (text-only Qwen3 by neměl vyžadovat remap). |

### 8.4 Co dělat po úspěšném smoke testu

1. Smazat smoke output: `rm -rf ./qwen3-4b-tool-r0/smoke_iter1_generator`.
2. Spustit plný self-play: `bash run_main.sh` (3 iterace, 50 GRPO kroků na step1+step3, ~8–10 hodin na 3× A100 40GB).
3. Po finálním `iter3_solver/checkpoint-50` re-spustit eval s aktualizovanými configs:

```bash
python -m eval.run_eval --config eval/configs/finetuned.yaml
python -m eval.run_eval --config eval/configs/baseline.yaml
python eval/scripts/compare.py --baseline ... --finetuned ...
```

Cíl: kvantifikovat NESTFUL Partial Match vůči předchozímu Qwen3.5-4B baseline (14.31 %) a Qwen2.5-1.5B-Instruct fine-tuned (13.20 %). Hypotéza: silnější base + delší kontext + LoRA stabilita → ≥ 18 %.

---

## Reference

- Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Rajbhandari et al., 2020, *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Shao et al., 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO formulation), [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- transformers issue [#45313](https://github.com/huggingface/transformers/issues/45313) — Qwen3.5 + ZeRO-3 weight load
- transformers issue [#45127](https://github.com/huggingface/transformers/issues/45127) — model collapse on tied embeddings + modules_to_save
- DeepSpeed issue [#7170](https://github.com/deepspeedai/DeepSpeed/issues/7170)
- TRL issue [#3089](https://github.com/huggingface/trl/issues/3089) — GRPO + PEFT + gradient_checkpointing
- TRL issue [#3108](https://github.com/huggingface/trl/issues/3108) — sync_ref_model silently no-op with LoRA
- PEFT issue [#2892](https://github.com/huggingface/peft/issues/2892) — empty adapter on ZeRO-3 save
- vLLM issue [#34186](https://github.com/vllm-project/vllm/issues/34186) — silent-zero LoRA on Qwen3.5 prefix mismatch
- TRL `GRPOTrainer` PEFT support — [trl/trainer/grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
