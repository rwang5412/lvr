# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Training code for **Latent Visual Reasoning (LVR)** on top of Qwen2.5-VL (3B / 7B). Forked from [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune). Two training stages:

1. **Stage-1 SFT** (`src/train/train_lvr.py`) — supervised finetune with an auxiliary LVR loss on latent visual tokens.
2. **Stage-2 GRPO<sub>latent</sub>** (`src/train/train_grpo.py`) — RL finetune over LVR rollouts.

A vanilla SFT path (no LVR loss) also exists at `src/train/train_sft.py`.

## Environment

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation   # must come last
```

`transformers>=4.54.0` is required for the RL path (abstract model architecture change). See README "Known Issues".

## Running training / eval

All training is launched via `deepspeed` through shell scripts in `scripts/`:

```bash
bash scripts/finetune_lvr_stage1_7b.sh   # Stage-1 SFT, 7B
bash scripts/finetune_lvr_stage2_7b.sh   # Stage-2 GRPO, 7B (consumes a Stage-1 checkpoint)
bash scripts/vanilla_sft_7b.sh           # Plain SFT baseline
# 3B variants exist with the same names
```

DeepSpeed configs: `scripts/zero{2,3}{,_offload}.json`. Stage-1 7B uses `zero3_offload.json`; Stage-2 uses `zero2.json`.

Evaluation (BLINK / V*Bench / MMVP) uses max-step decoding by default:

```bash
PYTHONPATH=src python evaluation/evaluation.py
```

Other decoding strategies live in `src/model/qwen_lvr_model.py` — search for `decoding_strategy`.

There is **no test suite, linter, or formatter wired up** in this repo. Don't invent commands for these.

## Architecture — the big picture

### Model: `QwenWithLVR` (`src/model/qwen_lvr_model.py`)

Subclass of `Qwen2_5_VLForConditionalGeneration` that adds:
- LVR special tokens (`<|lvr_start|>`, `<|lvr|>`, `<|lvr_end|>`, `<|lvr_latent_end|>` — see `src/constants.py`).
- Optional LVR head (`src/model/lvr_heads.py`) when `--lvr_head True`.
- Custom decoding loops (max-step / steps / latent-end-token) used at inference and during GRPO rollouts.

### Monkey patches (applied before training starts)

These are not optional — the training scripts call them at startup. When debugging behavior that doesn't match stock Qwen2.5-VL, check here first:

- `src/train/monkey_patch_patch_emb.py` — replaces the 3D conv patch embedding to avoid NaNs from numeric instability. README explicitly calls this out.
- `src/train/monkey_patch_forward.py` / `monkey_patch_forward_lvr.py` / `monkey_patch_forward_lvr_rl.py` — mixed-modality forward variants for plain SFT, LVR SFT, and LVR RL respectively. Each training entry point installs the matching one.
- `src/train/monkey_patch_dataloader.py` — replaces the HF `Trainer` dataloader to support data packing.

### Trainers (`src/trainer/`)

Three HF `Trainer` subclasses, selected by entry point:
- `QwenSFTTrainer` — vanilla SFT.
- `QwenLVRSFTTrainer` — Stage-1; adds the LVR loss term (`--loss_lvr_fct {mse,l1}`, `--loss_lvr_lambda`).
- `QwenGRPOTrainer` — Stage-2; subclasses `trl.GRPOTrainer`. **Liger GRPO loss and vLLM backend are not yet supported** (README).

### Datasets (`src/dataset/`)

LLaVA-style JSON with image paths relative to `--image_folder`. The Stage-1 format uses `<image>` and `<lvr>` placeholders plus optional `bboxes`; the Stage-2 GRPO format **omits `<image>` tokens** (different from typical setups — see README).

Key modules:
- `lvr_sft_dataset.py` — unpacked LVR SFT.
- `lvr_sft_dataset_packed.py` — InternVL-style packing of short instances; long instances pass through. Enabled with `--enable_data_packing True`. When packing is on, `BATCH_PER_DEVICE` must be 1; concurrency comes from `--max_instance_per_batch` and `--long_seq_threshold`.
- `grpo_dataset.py` — Stage-2 prompt dataset.

### Reward functions (`src/train/reward_funcs.py`)

Any function ending in `_reward` is auto-discovered by `src.utils.load_reward_funcs` and registered with the GRPO trainer. Add new rewards here; no registration plumbing needed.

### Custom system prompts (`src/constants.py`)

Append to this file rather than threading new prompt strings through the call sites. `SYSTEM_MESSAGE` is the default; `LVR_SYSTEM_MESSAGE` is the answer-formatting prompt used in LVR runs.

### Checkpointing

`src/s3_checkpoints_lvr.py` provides `OCIFolderCheckpointHandler` — uploads checkpoints to OCI Object Storage (S3-compatible). Toggled per-run via `--online_checkpoint`. Credentials are read from env vars: `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`, `ENDPOINT_URL`, `BUCKET_NAME`, `REGION_NAME`, plus `CACHE_DIR` for the local staging dir. When `online_checkpoint=True`, `output_dir` is rewritten to a temp local dir and the original value becomes the remote prefix. `--save_total_limit` only bounds local copies.

## Training arg conventions worth knowing

- `--image_min_pixels` / `--image_max_pixels` are passed as **pixel counts**, but scripts compute them as `tokens * 28 * 28` (Qwen's 28-pixel patch). Keep that pattern when editing.
- `--coconut True` enables the continuous-reasoning (LVR) code paths in `QwenWithLVR`.
- Stage-2 GRPO is **sensitive to `--temperature`**; the default 0.9 is a tuned value, not arbitrary.
- The 7B Stage-2 script references `STAGE1_STEPS=2500` and expects a Stage-1 checkpoint at `stage1_checkpoints/.../checkpoint-2500/` — update both when chaining runs.
