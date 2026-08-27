#!/usr/bin/env bash
set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-}"
export TRAIN_FILES="${TRAIN_FILES:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-}"

# ===== SP (Ulysses Sequence Parallel) =====
export SP_SIZE="${SP_SIZE:-}"

# ===== Batch / length =====
export TRAIN_BATCH_SIZE=8            
export MICRO_BATCH_SIZE=1            
export MAX_LEN="${MAX_LEN:-}"
export MAX_TOKEN_LEN_PER_GPU=25000      

# ===== Optim =====
export TOTAL_EPOCHS=1
export MAX_STEPS=null              
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export WEIGHT_DECAY=0.0
export WARMUP_RATIO=0.03
export MAX_GRAD_NORM=1.0
export LR_SCHEDULER=cosine
export SEED=42

# ===== Logging / ckpt =====
export LOG_BACKEND=wandb
export PROJECT_NAME="${PROJECT_NAME:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
export SAVE_FREQ=10
export EVAL_FREQ=-1

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=29500 \
  -m verl.trainer.sft_trainer \
  --config-path config \
  --config-name sft_trainer_engine \
  model.path="$BASE_MODEL" \
  model.tokenizer_path="$TOKENIZER_PATH" \
  +model.override_config.attn_implementation=flash_attention_2 \
  +model.override_config.model_config.dtype=bfloat16 \
  +model.override_config.model_config.vision_config.dtype=bfloat16 \
  +model.override_config.use_cache=false \
  +model.override_config.model_config.use_cache=false \
  +model.override_config.gradient_checkpointing=true \
  +model.override_config.model_config.gradient_checkpointing=true \
  engine.model_dtype=bfloat16 \
  engine.use_torch_compile=false \
  engine.ulysses_sequence_parallel_size="$SP_SIZE" \
  data.train_files="[$TRAIN_FILES]" \
  data.val_files="[]" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  data.use_dynamic_bsz=true \
  data.max_length="$MAX_LEN" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.pad_mode=no_padding \
  data.truncation=right \
  +data.skip_invalid_samples=true \
  +data.max_skip_attempts=1 \
  +data.skip_warn_every=20 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$MAX_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$EVAL_FREQ" \
  trainer.logger="$LOG_BACKEND" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.seed="$SEED" \
  trainer.resume_from_path= \
  optim.lr="$LEARNING_RATE" \
  optim.weight_decay="$WEIGHT_DECAY" \
  optim.lr_warmup_steps_ratio="$WARMUP_RATIO" \
  optim.lr_scheduler_type="$LR_SCHEDULER" \
  optim.clip_grad="$MAX_GRAD_NORM"

echo "SFT finished. Sharded checkpoints saved to $OUTPUT_DIR; merge the final global_step directory for RL use."
