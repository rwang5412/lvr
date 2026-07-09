#!/usr/bin/env bash
# Step-1 bottleneck SFT (continue-FT from the LVR base, USE_BOTTLENECK=True, existing losses only).
# Checkpoints -> $OUTPUT_DIR (scratch). Job log: /scratch/haizhow/logs/train_bottleneck_%j.out
#
# Usage:
#   mkdir -p /scratch/haizhow/logs            # once
#   sbatch scripts/palmetto/train_bottleneck.sbatch
#   # override step count / slice:
#   sbatch --export=ALL,MAX_STEPS=300,DATA_PATH=data/meta_data_bottleneck_slice.json \
#          scripts/palmetto/train_bottleneck.sbatch
#
#SBATCH --job-name=lvr-bottleneck-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G   # 7B ZeRO-3 full CPU offload (params+optimizer via cpu_adam) needs ~300G; 200G OOMs
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:4
#SBATCH --output=/scratch/haizhow/logs/train_bottleneck_%j.out
#SBATCH --error=/scratch/haizhow/logs/train_bottleneck_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/haizhow/lvr}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/haizhow/cache}"
ENV_NAME="${ENV_NAME:-train}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-/home/haizhow/lvr/weights/LVR-7B}"   # base LVR ckpt (continue-FT init)
DATA_PATH="${DATA_PATH:-data/meta_data_bottleneck_slice.json}"           # meta -> slice_train.json + image root
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/haizhow/stage1_arms/bottleneck/}"      # both arms live under stage1_arms/
MAX_STEPS="${MAX_STEPS:-833}"                                            # 1 epoch @ 4 GPU on 80k: 80000/(1*4*8*3pack)
NUM_DEVICES="${NUM_DEVICES:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"                                # global batch = 1 * NUM_DEVICES * GRAD_ACCUM
LR="${LR:-1e-5}"
LAMBDA_LVR="${LAMBDA_LVR:-0.1}"
LST="${LST:-4096}"
MAX_INSTANCE_PER_BATCH="${MAX_INSTANCE_PER_BATCH:-4}"

export WANDB_PROJECT="${WANDB_PROJECT:-LVR-7B-Step1-Bottleneck}"
export WANDB_MODE="${WANDB_MODE:-offline}"        # compute nodes have no internet
export HF_HOME="${CACHE_ROOT}/huggingface"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # reduce fragmentation on long packed seqs

[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL:-0}" != "1" ]] && { echo "ERROR: use sbatch (or ALLOW_LOCAL=1)"; exit 1; }
[[ -d "$PROJECT_ROOT" ]] || { echo "ERROR: PROJECT_ROOT missing: $PROJECT_ROOT"; exit 1; }
[[ -d "$CHECKPOINT_NAME" ]] || { echo "ERROR: base checkpoint missing: $CHECKPOINT_NAME"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no GPU"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT" /scratch/haizhow/logs
source "$(conda info --base)/etc/profile.d/conda.sh"
# module load + conda under `set +u`: Lmod/conda touch unbound vars that -u would abort on.
set +u; conda activate "$ENV_NAME"; module load cuda/12.3 gcc/12.3.0; set -u
cd "$PROJECT_ROOT"
# BOTH paths: code mixes `from src.X` (needs root) and `from train.X` (needs root/src).
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# cpu_adam is JIT-compiled at startup. With >1 rank the concurrent ninja builds race and one rank
# fails ("Error building extension 'cpu_adam'"). Use a per-job, NODE-LOCAL cache (correct -march=native
# for this node; no stale/wrong-arch reuse) and PRE-BUILD single-process so the ranks only load the .so.
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_${SLURM_JOB_ID:-$$}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "[prebuild] compiling cpu_adam single-process into $TORCH_EXTENSIONS_DIR ..."
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load()" && echo "[prebuild] cpu_adam OK"

echo "=== bottleneck FT ===  init=$CHECKPOINT_NAME  data=$DATA_PATH  max_steps=$MAX_STEPS  gpus=$NUM_DEVICES  out=$OUTPUT_DIR"
deepspeed --num_gpus "$NUM_DEVICES" src/train/train_lvr.py \
    --run_name "Step1_bottleneck_${MAX_STEPS}steps" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --coconut True \
    --loss_lvr_fct mse \
    --deepspeed scripts/zero3_offload.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvr_head False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --loss_lvr_lambda "$LAMBDA_LVR" \
    --bf16 True --fp16 False \
    --disable_flash_attn2 True \
    --use_bottleneck True \
    --online_checkpoint False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((5120 * 28 * 28)) \
    --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --tf32 False --gradient_checkpointing True \
    --report_to wandb --lazy_preprocess True \
    --save_strategy steps --save_steps 100 --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --enable_data_packing True \
    --max_packed_tokens $((MAX_INSTANCE_PER_BATCH * LST)) \
    --random_seed 42 \
    --long_seq_threshold "$LST" \
    --max_instance_per_batch "$MAX_INSTANCE_PER_BATCH"

echo "Done. Checkpoints -> $OUTPUT_DIR"
