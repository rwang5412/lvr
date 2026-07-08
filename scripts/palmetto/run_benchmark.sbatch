#!/usr/bin/env bash
# Benchmark accuracy eval (evaluation/evaluation.py) — V*Bench / BLINK / MMVP letter-accuracy.
# This is the "brittleness guard", SEPARATE from the causal harness (run_harness.py).
#
# NOTE: our evaluation.py is NOT CLI-parameterized. It reads module-level CHKPT_PATHS and
# DATASET_CONFIG and logs to evaluation/results_log/eval_<timestamp>.log. So before submitting,
# edit those in evaluation/evaluation.py (checkpoint path, which benchmarks, dataset dirs).
# This launcher just gives it a clean, cached, GPU environment.
#
# Usage:
#   mkdir -p /scratch/haizhow/logs            # once
#   # 1) edit CHKPT_PATHS / DATASET_CONFIG in evaluation/evaluation.py
#   # 2) sbatch scripts/palmetto/run_benchmark.sbatch
#
#SBATCH --job-name=lvr-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=/scratch/haizhow/logs/benchmark_%j.out
#SBATCH --error=/scratch/haizhow/logs/benchmark_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/haizhow/lvr}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/haizhow/cache}"
ENV_NAME="${ENV_NAME:-train}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONNOUSERSITE=1

[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL:-0}" != "1" ]] && { echo "ERROR: use sbatch (or ALLOW_LOCAL=1)"; exit 1; }
[[ -d "$PROJECT_ROOT" ]] || { echo "ERROR: PROJECT_ROOT missing: $PROJECT_ROOT"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no GPU"; exit 1; }

mkdir -p "$CACHE_ROOT" /scratch/haizhow/logs
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; set -u
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "=== benchmark eval (evaluation.py) ===  results -> evaluation/results_log/ + per-benchmark out dirs"
python -u evaluation/evaluation.py
echo "Done. See evaluation/results_log/ for the timestamped log."
