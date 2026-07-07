#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=a100:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --job-name=bottleneck_ft
#SBATCH --output=bottleneck_ft_%j.out
#
# Run with:  sbatch scripts/finetune_lvr_stage1_bottleneck.sh
# (deepspeed launches across the 8 GPUs the SBATCH line allocates on one node.)
#
# Step-1: continue-finetune the LVR base with the answer->image BOTTLENECK ON, existing losses only
# (L_answer + L_patch, NO new distillation loss). Trains on a diverse multi-source slice for a short
# reroute (~200-450 steps). Harness the result on the CLEAN val held-out and compare to the base.
#
# This is the DECISION experiment: if causality moves + accuracy holds + probe recovers, you may not
# need a distillation loss at all. Do NOT build the loss until this says you need it.

# ---- model: load the LVR BASE as init (continue-FT), not Qwen from scratch ----
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"          # arch + processor
CHECKPOINT_NAME="/home/haizhow/lvr/weights/LVR-7B"  # base LVR checkpoint ($CKPT)
export WANDB_PROJECT="LVR-7B-Step1-Bottleneck"

# ---- data: the diverse slice from make_slice + the vcot_dl image root ----
DATA_PATH="data/meta_data_bottleneck_slice.json"   # points at slice_train.json + /scratch/haizhow/vcot_dl
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=4
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))
RANDOM_SEED=42

# ---- steps: SET FROM make_slice's ESTIMATED-STEPS report (keep ~200-450) ----
MAX_STEPS=417                                       # 80k slice x1 epoch ~= 417 steps
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=8

LR=1e-5
LVR_HEAD=False
USE_BOTTLENECK=True                                 # <-- the point of this run

LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1
MAX_TOKEN=5120
MIN_TOKEN=128

RUN_NAME="Step1_bottleneck_${MAX_STEPS}steps_${LVR_LOSS_FCT}Lambda${LAMBDA_LVR}"
ONLINE=False
OUTPUT_DIR="stage1_bottleneck_checkpoints/"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate train
cd ~/lvr

deepspeed src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --use_bottleneck $USE_BOTTLENECK \
    --online_checkpoint $ONLINE \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH
