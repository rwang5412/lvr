#!/usr/bin/env bash
# CapImagine do(Z) accuracy harness on one checkpoint + one dataset. 1 GPU, inference only.
#
# Benchmarks (vstar/hrbench_4k/mme_realworld/...) download from HF -> they MUST be pre-cached on the
# LOGIN node first (compute nodes have no internet). One-time, on login:
#   conda activate train; export HF_HOME=/scratch/haizhow/cache/huggingface
#   python -c "from datasets import load_dataset; load_dataset('DreamMr/HR-Bench','hrbench_4k'); \
#              load_dataset('yifanzhang114/MME-RealWorld-Lite'); load_dataset('craigwu/vstar_bench')"
#
# Usage:
#   mkdir -p /scratch/haizhow/logs
#   # gqa (in-domain, free-form):
#   sbatch --export=ALL,CHECKPOINT=/scratch/haizhow/ckpts/bottleneck_7b/checkpoint-833,DATASET=gqa,USE_BOTTLENECK=0 \
#          scripts/palmetto/run_capimagine.sbatch
#   # benchmark (paper-style, multiple-choice):
#   sbatch --export=ALL,CHECKPOINT=/scratch/haizhow/ckpts/bottleneck_7b/checkpoint-833,DATASET=hrbench_4k,USE_BOTTLENECK=0 \
#          scripts/palmetto/run_capimagine.sbatch
#
#SBATCH --job-name=lvr-capimagine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=/scratch/haizhow/logs/capimagine_%j.out
#SBATCH --error=/scratch/haizhow/logs/capimagine_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/haizhow/lvr}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/haizhow/cache}"
ENV_NAME="${ENV_NAME:-train}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/path/to/checkpoint}"
DATASET="${DATASET:-gqa}"
RECORDS="${RECORDS:-data/lvr_data/heldout_val_clean.json}"   # gqa only
IMAGE_FOLDER="${IMAGE_FOLDER:-/scratch/haizhow/vcot_dl}"      # gqa only
USE_BOTTLENECK="${USE_BOTTLENECK:-0}"
LIMIT="${LIMIT:-300}"
LVR_STEPS="${LVR_STEPS:-16}"
OUT="${OUT:-/scratch/haizhow/results/capimagine/${DATASET}_bn${USE_BOTTLENECK}_${SLURM_JOB_ID:-local}}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # benchmarks must be pre-cached on login (no internet here)
export PYTHONNOUSERSITE=1

mkdir -p /scratch/haizhow/logs /scratch/haizhow/results/capimagine "$CACHE_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; module load cuda/12.3 gcc/12.3.0; set -u
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "=== capimagine ===  ckpt=$CHECKPOINT  dataset=$DATASET  use_bottleneck=$USE_BOTTLENECK  limit=$LIMIT  out=$OUT"
python evaluation/capimagine_harness.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --records "$RECORDS" \
    --image-folder "$IMAGE_FOLDER" \
    --use-bottleneck "$USE_BOTTLENECK" \
    --limit "$LIMIT" \
    --lvr-steps "$LVR_STEPS" \
    --out "$OUT"
echo "Done. Report -> $OUT.json"
