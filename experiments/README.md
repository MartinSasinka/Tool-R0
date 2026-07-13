# Agentic data generation

Zkopíruj celou složku `experiments/` do kořene repa:

```
<REPO_ROOT>/experiments/
  nestful_synthetic_curriculum_v3/
  nestful_mtgrpo_minimal/      # včetně data/
  nestful_mtgrpo_partial/
```

Potřebuješ: **GPU**, **`OPENROUTER_API_KEY`**, **3 NESTFUL JSONL** (contamination gate):

```
experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl
experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl
experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl
```

## Setup

```bash
cd <REPO_ROOT>
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/setup_agentic_venv.sh
pip install pyyaml
source experiments/nestful_synthetic_curriculum_v3/.venv/bin/activate
export OPENROUTER_API_KEY="sk-or-..."
```

## Generace (4 GPU, 800/stage)

```bash
export STAGES="stage2_2call_agentic_openrouter"
export NUM_GPUS=4
export TOTAL_PER_STAGE=800
export BASE_SEED=56
export TOTAL_SPEND_USD=40
export TOTAL_REQUESTS=8000
export OPENROUTER_MAX_ITERATIONS_PER_STAGE=3000
export OUT_BASE="experiments/nestful_synthetic_curriculum_v3/data/agentic_workers/stage2_mt_800_$(date +%Y%m%d)"

bash experiments/nestful_synthetic_curriculum_v3/scripts/data/launch_multi_gpu_workers.sh
```

## Merge

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/data/merge_agentic_workers.py \
  --workers-glob "experiments/nestful_synthetic_curriculum_v3/data/agentic_workers/stage2_mt_800_*/gpu*" \
  --output-dir experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter
```

Více: `nestful_synthetic_curriculum_v3/docs/AGENTIC_DATA_GENERATION.md`
