#!/usr/bin/env bash
set -euo pipefail

# Run Experiment 1 / ID 5 only.
# Usage: ./scripts/run_experiments_1_4.sh [TIMESTEPS] [MAX_ITER]

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
CONDA_PREFIX="/Users/timothy/opt/miniconda3/envs/py312"
PY="$CONDA_PREFIX/bin/python"

TIMESTEPS=${1:-1000000}
MAX_ITER=${2:-100}
SEED=42

echo "Root: $ROOT"
echo "Python: $PY"
echo "Timesteps: $TIMESTEPS, Max iterations: $MAX_ITER"

# Optional: rebuild the PyO3 extension before runs (uncomment to enable)
# echo "Rebuilding rust extension (maturin)..."
# "$PY" -m maturin develop --release --manifest-path "$ROOT/rust_alns/Cargo.toml"

run_experiment(){
  local action=$1
  local tag=$2
  echo "\n=== Running $tag ($action) ==="
  # Let the Python runner generate a fresh timestamped run id (do not export
  # RL_RUN_ID here to avoid accidental reuse of old runs).
  "$PY" "$ROOT/scripts/run_experiment_rl_improvement_mvp.py" \
    --action-module "$action" \
    --train-timesteps "$TIMESTEPS" \
    --max-iterations "$MAX_ITER" \
    --run-tag "$tag" \
    --seed "$SEED"
}

run_experiment destroy_repair_many_v1 exp1_id5_destroy_repair_many_v1

echo "\nExperiment 1 / ID 5 finished. Check logs/ for results." 
