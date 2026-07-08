#!/usr/bin/env bash
# Causal harness (evaluation/run_harness.py) — mediation / latent battery on a held-out set.
# Output: $RESULT_ROOT/$OUT.txt (+ .json with per_example). Job log: /scratch/haizhow/logs/harness_%j.out
#
# Usage:
#   mkdir -p /scratch/haizhow/logs                 # once (SBATCH --output dir must pre-exist)
#   # baseline (base ckpt, no bottleneck):
#   sbatch scripts/palmetto/run_harness.sbatch
#   # a different held-out draw:
#   sbatch --export=ALL,HELDOUT=data/lvr_data/heldout_val_seed2024.json,OUT=baseline_val_s2024 \
#          scripts/palmetto/run_harness.sbatch
#   # bottleneck arm (after Step-1 FT):
#   sbatch --export=ALL,CHECKPOINT=/scratch/haizhow/stage1_bottleneck_checkpoints/.../checkpoint-final,\
# USE_BOTTLENECK=1,OUT=step1_bottleneck_val scripts/palmetto/run_harness.sbatch
#
#SBATCH --job-name=lvr-harness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=/scratch/haizhow/logs/harness_%j.out
#SBATCH --error=/scratch/haizhow/logs/harness_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/haizhow/lvr}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/scratch/haizhow/vcot_dl}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/haizhow/results/harness}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/haizhow/cache}"
ENV_NAME="${ENV_NAME:-train}"

CHECKPOINT="${CHECKPOINT:-/home/haizhow/lvr/weights/LVR-7B}"
HELDOUT="${HELDOUT:-data/lvr_data/heldout_val_clean.json}"
OUT="${OUT:-baseline_val}"
CORRUPTION="${CORRUPTION:-mean}"
LIMIT="${LIMIT:-300}"
USE_BOTTLENECK="${USE_BOTTLENECK:-0}"
SKIP_FLIP="${SKIP_FLIP:-0}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONNOUSERSITE=1

[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL:-0}" != "1" ]] && { echo "ERROR: use sbatch (or ALLOW_LOCAL=1)"; exit 1; }
[[ -d "$PROJECT_ROOT" ]] || { echo "ERROR: PROJECT_ROOT missing: $PROJECT_ROOT"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no GPU (nvidia-smi absent)"; exit 1; }

mkdir -p "$RESULT_ROOT" "$CACHE_ROOT" /scratch/haizhow/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; set -u

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

ARGS=(--checkpoint "$CHECKPOINT" --image-folder "$IMAGE_FOLDER"
      --heldout "$HELDOUT" --out "$RESULT_ROOT/$OUT"
      --corruption "$CORRUPTION" --limit "$LIMIT")
[[ "$USE_BOTTLENECK" == "1" ]] && ARGS+=(--use-bottleneck)
[[ "$SKIP_FLIP" == "1" ]] && ARGS+=(--skip-flip)

echo "=== harness ===  ckpt=$CHECKPOINT  heldout=$HELDOUT  out=$RESULT_ROOT/$OUT  bottleneck=$USE_BOTTLENECK"
python -u evaluation/run_harness.py "${ARGS[@]}"
echo "Done. Report -> $RESULT_ROOT/$OUT.txt  (+ .json)"
