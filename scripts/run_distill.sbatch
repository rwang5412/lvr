#!/bin/bash
#SBATCH --job-name=lvr-distill
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# --- environment (the recipe we validated interactively) ---
source ~/.bashrc
module load cuda/12.3 gcc/12.3.0
conda deactivate 2>/dev/null; conda activate train      # NOT qwen-venv
cd /home/haizhow/lvr
export PYTHONPATH="$(pwd):$(pwd)/src"
export WANDB_MODE=offline

# --- the distill arm: bottleneck ON + self-distillation KL (weight 0.5) ---
# continue-FT from the existing LVR checkpoint on the held-out-excluded slice; 1 epoch
# (~2500 steps at 4 GPUs = 80k / (1 x 4 x 8)). Checkpoints go next to the input weights.
deepspeed src/train/train_lvr.py \
    --run_name Stage1_distill_bn_w0.5 \
    --coconut True \
    --loss_lvr_fct mse \
    --deepspeed scripts/zero3_offload.json \
    --model_id weights/LVR-7B \
    --data_path data/meta_data_bottleneck_slice.json \
    --remove_unused_columns False \
    --lvr_head False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps 2500 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --loss_lvr_lambda 0.1 \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --use_bottleneck True \
    --distill_weight 0.5 \
    --online_checkpoint False \
    --output_dir weights/distill_bn_w0.5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((5120 * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --enable_data_packing True \
    --max_packed_tokens 16384 \
    --random_seed 42 \
    --long_seq_threshold 4096 \
    --max_instance_per_batch 4
