# NESTFUL data on RunPod

Git clone **does not include** NESTFUL JSONL by default if `data/` was gitignored.
Training and preflight need:

```
experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl
experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl
experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl
```

## Option A — extract from tar (recommended if you have the archive)

```bash
cd /workspace/Tool-R0/Tool-R0   # adjust to your repo root

# if nestful_mtgrpo_minimal.tar is in repo root or /workspace:
tar -xf nestful_mtgrpo_minimal.tar

# merge data into repo layout (if tar extracts flat minimal folder):
mkdir -p experiments/nestful_mtgrpo_minimal/data
cp -a nestful_mtgrpo_minimal/data/* experiments/nestful_mtgrpo_minimal/data/ 2>/dev/null || \
cp -a /workspace/nestful_mtgrpo_minimal/data/* experiments/nestful_mtgrpo_minimal/data/
```

Verify:

```bash
ls experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl
```

## Option B — only missing splits (if nestful_data.jsonl already exists)

```bash
python experiments/comparison/make_nestful_dev_split.py
```

## After nestful_data.jsonl exists — v3 analysis (once per pod)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analyze_nestful_motifs.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_distribution_audit.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_tool_family_realism.py
```

## Curriculum (synthetic train data — regenerate on pod, not in git)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/generate_motif_synthetic_tasks.py
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3.py
```

Then DRY RUN:

```bash
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```
