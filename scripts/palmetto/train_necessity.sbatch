#!/usr/bin/env bash
# Step-1 NECESSITY SFT (continue-FT from the LVR base). NO bottleneck: keeps L_answer + L_patch and adds
# the answer-margin necessity term between the with-latent and a latent-ABLATED forward (in-distribution
# shuffle/mean ablation). This is the no-mismatch pivot away from bottleneck+distill.
#
# MEMORY: the ablated forward runs WITH grad (unlike distill's no_grad teacher) -> ~2x activations. If it
# OOMs, LOWER MAX_INSTANCE_PER_BATCH (e.g. 2) before anything else.
#
# Usage:
#   mkdir -p /scratch/haizhow/logs            # once
#   # A100-40GB (default: CPU offload + cpu_adam):
#   sbatch --export=ALL,MAX_STEPS=833,NECESSITY_WEIGHT=1.0,OUTPUT_DIR=/scratch/haizhow/ckpts/necessity_7b \
#          scripts/palmetto/train_necessity.sbatch
#   # H100-80GB (no offload, no cpu_adam, faster):
#   sbatch --gpus=h100:4 --export=ALL,DEEPSPEED_CONFIG=scripts/zero3.json,MAX_STEPS=833,NECESSITY_WEIGHT=1.0,OUTPUT_DIR=/scratch/haizhow/ckpts/necessity_7b \
#          scripts/palmetto/train_necessity.sbatch
#
#SBATCH --job-name=lvr-necessity-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G   # for the CPU-offload default; zero3.json (H100, no offload) needs far less
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:4
#SBATCH --output=/scratch/haizhow/logs/train_necessity_%j.out
#SBATCH --error=/scratch/haizhow/logs/train_necessity_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/haizhow/lvr}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/haizhow/cache}"
ENV_NAME="${ENV_NAME:-train}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-/home/haizhow/lvr/weights/LVR-7B}"   # base LVR ckpt (continue-FT init)
DATA_PATH="${DATA_PATH:-data/meta_data_bottleneck_slice.json}"           # meta -> slice_train.json + image root
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/haizhow/stage1_arms/necessity/}"
MAX_STEPS="${MAX_STEPS:-833}"
NUM_DEVICES="${NUM_DEVICES:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-5}"
LAMBDA_LVR="${LAMBDA_LVR:-0.1}"
NECESSITY_WEIGHT="${NECESSITY_WEIGHT:-1.0}"                              # lambda: strength of the necessity term
NECESSITY_MARGIN="${NECESSITY_MARGIN:-0.5}"                             # M (nats): calibrated start (base clean NLL ~0.08, gap ~0.07); fix M, tune lambda
NECESSITY_ABLATION="${NECESSITY_ABLATION:-shuffle}"                     # in-distribution ablation: shuffle | mean (NOT zeros)
LST="${LST:-4096}"
MAX_INSTANCE_PER_BATCH="${MAX_INSTANCE_PER_BATCH:-3}"                   # LOWER (2) if the 2nd grad forward OOMs
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}"      # H100: pass scripts/zero3.json (no offload)

export WANDB_PROJECT="${WANDB_PROJECT:-LVR-7B-Step1-Necessity}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # necessity holds 2 grad forwards -> reduce fragmentation

[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL:-0}" != "1" ]] && { echo "ERROR: use sbatch (or ALLOW_LOCAL=1)"; exit 1; }
[[ -d "$PROJECT_ROOT" ]] || { echo "ERROR: PROJECT_ROOT missing: $PROJECT_ROOT"; exit 1; }
[[ -d "$CHECKPOINT_NAME" ]] || { echo "ERROR: base checkpoint missing: $CHECKPOINT_NAME"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no GPU"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT" /scratch/haizhow/logs
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$ENV_NAME"; module load cuda/12.3 gcc/12.3.0; set -u
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# cpu_adam only needed when the deepspeed config offloads the optimizer to CPU. Pre-build single-process so
# ranks don't race; skipped entirely for a no-offload config (zero3.json on H100).
if grep -q '"offload_optimizer"' "$DEEPSPEED_CONFIG"; then
    export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_ext_${SLURM_JOB_ID:-$$}"
    mkdir -p "$TORCH_EXTENSIONS_DIR"
    echo "[prebuild] $DEEPSPEED_CONFIG offloads optimizer -> compiling cpu_adam single-process ..."
    python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load()"
    echo "[prebuild] cpu_adam OK"
else
    echo "[prebuild] $DEEPSPEED_CONFIG has no CPU offload -> skipping cpu_adam (fused GPU Adam)"
fi

echo "=== necessity FT ===  init=$CHECKPOINT_NAME  data=$DATA_PATH  max_steps=$MAX_STEPS  gpus=$NUM_DEVICES  necessity_weight=$NECESSITY_WEIGHT  margin=$NECESSITY_MARGIN  ablation=$NECESSITY_ABLATION  out=$OUTPUT_DIR"
deepspeed --num_gpus "$NUM_DEVICES" src/train/train_lvr.py \
    --run_name "Step1_necessity_w${NECESSITY_WEIGHT}_m${NECESSITY_MARGIN}_${MAX_STEPS}steps" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --coconut True \
    --loss_lvr_fct mse \
    --deepspeed "$DEEPSPEED_CONFIG" \
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
    --use_bottleneck False \
    --distill_weight 0.0 \
    --necessity_weight "$NECESSITY_WEIGHT" \
    --necessity_margin "$NECESSITY_MARGIN" \
    --necessity_ablation "$NECESSITY_ABLATION" \
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
