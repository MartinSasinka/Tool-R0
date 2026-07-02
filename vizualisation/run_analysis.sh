#!/usr/bin/env bash
# Full trajectory analysis pipeline for curriculum checkpoint comparisons.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-vizualisation/configs/trajectory_analysis_config.json}"
PYTHON="${PYTHON:-python3}"

RUN_DIR="$("$PYTHON" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("$ROOT")))
from vizualisation.scripts.lib.io_utils import load_config, run_dir_from_config
cfg = load_config("$CONFIG")
print(run_dir_from_config(cfg))
PY
)"

echo "Config:  $CONFIG"
echo "Run dir: $RUN_DIR"

run_step() {
  echo ""
  echo "==> $1"
  shift
  "$PYTHON" "$@"
}

run_step "collect_trajectories" vizualisation/scripts/collect_trajectories.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "canonicalize_trajectories" vizualisation/scripts/canonicalize_trajectories.py --run_dir "$RUN_DIR"
run_step "compute_trajectory_metrics" vizualisation/scripts/compute_trajectory_metrics.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "build_feature_space" vizualisation/scripts/build_feature_space.py --run_dir "$RUN_DIR"
run_step "reduce_dimensions" vizualisation/scripts/reduce_dimensions.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_skill_profiles" vizualisation/scripts/plot_skill_profiles.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_checkpoint_gains" vizualisation/scripts/plot_checkpoint_gains.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_embedding_map" vizualisation/scripts/plot_embedding_map.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_centroid_shift" vizualisation/scripts/plot_centroid_shift.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_distance_to_gold" vizualisation/scripts/plot_distance_to_gold.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "plot_error_distribution" vizualisation/scripts/plot_error_distribution.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "generate_report" vizualisation/scripts/generate_report.py --config "$CONFIG" --run_dir "$RUN_DIR"
run_step "inspect_run" vizualisation/scripts/inspect_run.py --run_dir "$RUN_DIR"

echo ""
echo "Done. Report: $RUN_DIR/reports/analysis_report.md"
