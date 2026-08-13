# P43 MT-GRPO training

Aktuální online multi-turn GRPO smyčka pro Qwen3-4B-Instruct-2507. Složka je
samostatná vůči starým experimentům; používá pouze Pilot 4.3 dataset a registry
z vedlejšího `targeted_tool_data_factory`.

## RunPod: nový běh na 4 GPU

Z kořene repozitáře:

```bash
bash experiments/nestful_mtgrpo_minimal/install_deps.sh
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_dynamic_online.sh
```

Rozdělení GPU je záměrné: GPU 0 počítá QLoRA/GRPO update, GPU 1–3 paralelně
generují vLLM rollouty. Výchozí konfigurace je
`configs/qwen3_p43_profile1000_dynamic_online_samplingfix.yaml`.

Skript před načtením modelu ověří:

- existenci profilového datasetu,
- dostupnost Pilot 4.3 tool registru,
- plný gold replay všech tréninkových příkladů.

## Pokračování 127 → 256

```bash
SRC_RUN=/cesta/k/qwen3_p43_profile1000_dynamic_online_samplingfix \
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_256.sh
```

## Pokračování 350 → 550 / 550 → 750 (enrich 30 %)

```bash
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_550_enrich30.sh
bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_750_enrich30.sh
```

Default CKPTs: `adapter_epoch_14` @ 350, `adapter_epoch_19` @ 550.
Konkrétní checkpoint lze přepsat proměnnou `CKPT`. Resume provádí fail-fast
kontrolu kroku, sampler state, reward policy a hashe datasetu.

## Přímé spuštění

```bash
cd experiments/nestful_mtgrpo_minimal
python run.py --mode train \
  --config configs/qwen3_p43_profile1000_dynamic_online_samplingfix.yaml
```

Pro smoke/eval lze použít stejné `--config` s režimem `smoke`, `rollout_eval`
nebo `final_eval`. Oficiální NESTFUL benchmark, dev/test split, scorer a IBM
executable functions zůstávají součástí `data/NESTFUL-main` a `data/splits`.
Trénink P43 dál používá syntetický executor; tyto benchmarkové soubory slouží
pro nezávislou finální evaluaci.

## NESTFUL temp-0 eval (4 GPU vLLM)

Ekvivalent starého `nestful_synthetic_curriculum_v3/scripts/v5/final_eval.sh`:

```bash
cd experiments/nestful_mtgrpo_minimal
CKPT=outputs/qwen3_p43_profile1000_dynamic_online_continue350_enrich30/checkpoints
OUT=outputs/evals/p43_nestful_t0_step350
export WANDB_MODE=disabled
export USE_VLLM=1 EVAL_TP=4 VLLM_GPU_UTIL=0.85
export CUDA_VISIBLE_DEVICES=0,1,2,3
LABEL=final CHECKPOINT=$CKPT/adapter_epoch_XX OUT_DIR=$OUT/step350 \
  bash scripts/final_eval.sh
```

## Testy

```bash
python -m pytest tests -q
```

Horká cesta tréninku je v `grpo_train.py`; vLLM batching a tří-GPU rollout pool
jsou v `vllm_generate.py` a `vllm_dp_pool.py`.
