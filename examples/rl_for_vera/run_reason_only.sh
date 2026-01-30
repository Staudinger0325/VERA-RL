#!/usr/bin/env bash
# k8s: pin Ray/c10d to pod IP + NIC to avoid Service DNS in Gloo/NCCL.

export VERL_LOGGING_LEVEL=INFO
# 正常启动训练即可
RAY_RUNTIME_ENV_ARGS=()
RAY_START_ARGS=(--head)
if [[ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]]; then
  IFNAME=""
  if command -v ip >/dev/null 2>&1; then
    IFNAME="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
  fi
  IFNAME="${IFNAME:-eth0}"
  POD_IP=""
  if command -v ip >/dev/null 2>&1; then
    POD_IP="$(ip -4 addr show dev "$IFNAME" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
  fi
  POD_IP="${POD_IP:-$(hostname -i | awk '{print $1}')}"
  export RAY_OVERRIDE_NODE_IP_ADDRESS="$POD_IP"
  export GLOO_SOCKET_IFNAME="$IFNAME"
  export NCCL_SOCKET_IFNAME="$IFNAME"
  RAY_START_ARGS+=(--node-ip-address "$POD_IP")
  RAY_RUNTIME_ENV_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_OVERRIDE_NODE_IP_ADDRESS=$POD_IP"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=$IFNAME"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=$IFNAME"
  )
fi

RAY_RUNTIME_ENV_ARGS+=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES='1'"
)

export RAY_TMPDIR=
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
ray start "${RAY_START_ARGS[@]}"
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
set -euo pipefail

# Resolve repo root so Hydra config path is stable regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/verl/trainer/config"

# ===== Paths =====
export BASE_MODEL=
export TOKENIZER_PATH=
export TRAIN_RL_PATH=

export STAGE1_OUTPUT_DIR=
export STAGE1_ROLLOUT_DIR=

export STAGE2_OUTPUT_DIR=
export STAGE2_DROP_LIST=

# ===== Shared rollouts (override in stage blocks if needed) =====
export ROLLOUT_ENGINE=vllm
export ROLLOUT_DTYPE=bfloat16
export ROLLOUT_TP=1
export ROLLOUT_DP=8
export ROLLOUT_GPU_MEM_UTIL=0.5
export ROLLOUT_MAX_MODEL_LEN=52048
export ROLLOUT_MAX_BATCHED_TOKENS=52048
export ROLLOUT_MAX_SEQS=1
export ROLLOUT_PROMPT_LEN=50000
export ROLLOUT_RESP_LEN=2048
export VLLM_SERVER_START_TIMEOUT_S=3600   # 30min

# ===== Shared data config =====
export DATA_PROMPT_KEY=messages
export DATA_IMAGE_KEY=images
export DATA_TRUNCATION=right
export DATA_DATALOADER_WORKERS=32

# ===== Logging =====
export PROJECT_NAME=two_stage_rl_main
export STAGE1_EXPERIMENT=stage1_main
export STAGE2_EXPERIMENT=stage2_main

# ===== Reward manager =====
export REWARD_MANAGER_NAME="${REWARD_MANAGER_NAME:-llm_judge}"
REWARD_MANAGER_ARGS=(
  reward_model.reward_manager="$REWARD_MANAGER_NAME"
  reward_model.enable=false
  reward_model.use_reward_loop=false
)

# ===== Drop thresholds (Stage2 filtering) =====
export MEAN_THRESHOLD=0.8
export VAR_THRESHOLD=0.02
export MIN_COUNT=8

# ===== Stage1 hyperparameters =====
export STAGE1_TOTAL_STEPS=50
export STAGE1_TRAIN_BATCH_SIZE=32
export STAGE1_PPO_MINI_BATCH=2
export STAGE1_PPO_MICRO_BATCH=1
export STAGE1_PPO_MAX_TOKENS=52048
export STAGE1_LR=2e-6
export STAGE1_WD=0.0
export STAGE1_WARMUP_RATIO=0.03
export STAGE1_CLIP_LOW=0.1
export STAGE1_CLIP_HIGH=0.5
export STAGE1_SEED=42
export STAGE1_SAVE_FREQ=1
export STAGE1_NGPUS=8

# ===== Stage1 resume (optional) =====
STAGE1_RESUME_FROM="${STAGE1_RESUME_FROM:-}"
if [[ "$STAGE1_RESUME_FROM" == "latest" ]]; then
  latest_file="$STAGE1_OUTPUT_DIR/latest_checkpointed_iteration.txt"
  if [[ -f "$latest_file" ]]; then
    step="$(cat "$latest_file")"
    STAGE1_RESUME_FROM="$STAGE1_OUTPUT_DIR/global_step_${step}"
  else
    echo "Missing $latest_file; set STAGE1_RESUME_FROM explicitly." >&2
    exit 1
  fi
fi
STAGE1_RESUME_ARGS=()
if [[ -n "$STAGE1_RESUME_FROM" ]]; then
  STAGE1_RESUME_ARGS=(
    trainer.resume_mode=resume_path
    trainer.resume_from_path="$STAGE1_RESUME_FROM"
  )
fi

# ===== Stage2 hyperparameters =====
export STAGE2_TOTAL_STEPS=30
export STAGE2_TRAIN_BATCH_SIZE=32
export STAGE2_PPO_MINI_BATCH=1
export STAGE2_PPO_MICRO_BATCH=1
export STAGE2_PPO_MAX_TOKENS=52048
export STAGE2_LR=2e-6
export STAGE2_WD=0.0
export STAGE2_WARMUP_RATIO=0.03
export STAGE2_CLIP_LOW=0.1
export STAGE2_CLIP_HIGH=0.5
export STAGE2_SEED=42
export STAGE2_SAVE_FREQ=10
export STAGE2_NGPUS=8

echo "[stage1] train on full train_RL.parquet (reason-only reward)"
python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=exp/ppo_two_stage_stage1 \
  reward_model.reward_kwargs.category_weights.scan.w_loc=0 \
  reward_model.reward_kwargs.category_weights.scan.w_reason=1 \
  reward_model.reward_kwargs.category_weights.scan.w_unrel=0 \
  reward_model.reward_kwargs.category_weights.verify.w_loc=0 \
  reward_model.reward_kwargs.category_weights.verify.w_reason=1 \
  reward_model.reward_kwargs.category_weights.verify.w_unrel=0 \
  reward_model.reward_kwargs.category_weights.reason.w_loc=0 \
  reward_model.reward_kwargs.category_weights.reason.w_reason=1 \
  reward_model.reward_kwargs.category_weights.reason.w_unrel=0 \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE" \
  actor_rollout_ref.rollout.dtype="$ROLLOUT_DTYPE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_DP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.prompt_length="$ROLLOUT_PROMPT_LEN" \
  actor_rollout_ref.rollout.response_length="$ROLLOUT_RESP_LEN" \
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_SEQS" \
  actor_rollout_ref.rollout.load_format=hf \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size="$STAGE1_PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$STAGE1_PPO_MICRO_BATCH" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$STAGE1_PPO_MAX_TOKENS" \
  actor_rollout_ref.actor.use_asymmetric_clip=true \
  actor_rollout_ref.actor.clip_ratio_low="$STAGE1_CLIP_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$STAGE1_CLIP_HIGH" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="$STAGE1_LR" \
  actor_rollout_ref.actor.optim.weight_decay="$STAGE1_WD" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$STAGE1_WARMUP_RATIO" \
  actor_rollout_ref.actor.optim.total_training_steps="$STAGE1_TOTAL_STEPS" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.enable_activation_offload=true \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.enable_activation_offload=true \
  +actor_rollout_ref.rollout.enable_sleep_mode=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_type=lru \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.enforce_eager=False \
  data.train_files="[$TRAIN_RL_PATH]" \
  data.prompt_key="$DATA_PROMPT_KEY" \
  data.image_key="$DATA_IMAGE_KEY" \
  data.video_key=null \
  +data.strip_assistant_messages=true \
  data.max_prompt_length="$ROLLOUT_PROMPT_LEN" \
  data.max_response_length="$ROLLOUT_RESP_LEN" \
  data.train_batch_size="$STAGE1_TRAIN_BATCH_SIZE" \
  data.filter_overlong_prompts=true \
  data.truncation="$DATA_TRUNCATION" \
  data.dataloader_num_workers="$DATA_DATALOADER_WORKERS" \
  ray_kwargs.ray_init.num_cpus=null \
  "${RAY_RUNTIME_ENV_ARGS[@]}" \
  "${REWARD_MANAGER_ARGS[@]}" \
  "${STAGE1_RESUME_ARGS[@]}" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$STAGE1_EXPERIMENT" \
  trainer.default_local_dir="$STAGE1_OUTPUT_DIR" \
  trainer.rollout_data_dir="$STAGE1_ROLLOUT_DIR" \
  trainer.total_training_steps="$STAGE1_TOTAL_STEPS" \
  trainer.total_epochs=100000 \
  trainer.save_freq="$STAGE1_SAVE_FREQ" \
  trainer.n_gpus_per_node="$STAGE1_NGPUS"

echo "[stage2] build drop list from stage1 rollouts"
python3 scripts/build_stage2_drop_list.py \
  --rollout_dir "$STAGE1_ROLLOUT_DIR" \
  --out_path "$STAGE2_DROP_LIST" \
  --mean_threshold "$MEAN_THRESHOLD" \
  --var_threshold "$VAR_THRESHOLD" \
  --min_count "$MIN_COUNT"

STAGE1_CKPT="${STAGE1_CKPT:-}"
if [[ -z "$STAGE1_CKPT" ]]; then
  latest_file="$STAGE1_OUTPUT_DIR/latest_checkpointed_iteration.txt"
  if [[ -f "$latest_file" ]]; then
    step="$(cat "$latest_file")"
    STAGE1_CKPT="$STAGE1_OUTPUT_DIR/global_step_${step}"
  else
    echo "Missing $latest_file; set STAGE1_CKPT explicitly." >&2
    exit 1
  fi
fi

echo "[stage2] resume from $STAGE1_CKPT with drop list (reason-only reward)"
python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=exp/ppo_two_stage_stage2 \
  reward_model.reward_kwargs.category_weights.scan.w_loc=0 \
  reward_model.reward_kwargs.category_weights.scan.w_reason=1 \
  reward_model.reward_kwargs.category_weights.scan.w_unrel=0 \
  reward_model.reward_kwargs.category_weights.verify.w_loc=0 \
  reward_model.reward_kwargs.category_weights.verify.w_reason=1 \
  reward_model.reward_kwargs.category_weights.verify.w_unrel=0 \
  reward_model.reward_kwargs.category_weights.reason.w_loc=0 \
  reward_model.reward_kwargs.category_weights.reason.w_reason=1 \
  reward_model.reward_kwargs.category_weights.reason.w_unrel=0 \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE" \
  actor_rollout_ref.rollout.dtype="$ROLLOUT_DTYPE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_DP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.prompt_length="$ROLLOUT_PROMPT_LEN" \
  actor_rollout_ref.rollout.response_length="$ROLLOUT_RESP_LEN" \
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_SEQS" \
  actor_rollout_ref.rollout.load_format=hf \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size="$STAGE2_PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$STAGE2_PPO_MICRO_BATCH" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$STAGE2_PPO_MAX_TOKENS" \
  actor_rollout_ref.actor.use_asymmetric_clip=true \
  actor_rollout_ref.actor.clip_ratio_low="$STAGE2_CLIP_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$STAGE2_CLIP_HIGH" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="$STAGE2_LR" \
  actor_rollout_ref.actor.optim.weight_decay="$STAGE2_WD" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$STAGE2_WARMUP_RATIO" \
  actor_rollout_ref.actor.optim.total_training_steps="$STAGE2_TOTAL_STEPS" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.enable_activation_offload=true \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.enable_activation_offload=true \
  +actor_rollout_ref.rollout.enable_sleep_mode=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_type=lru \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.enforce_eager=False \
  data.train_files="[$TRAIN_RL_PATH]" \
  data.prompt_key="$DATA_PROMPT_KEY" \
  data.image_key="$DATA_IMAGE_KEY" \
  +data.strip_assistant_messages=true \
  data.max_prompt_length="$ROLLOUT_PROMPT_LEN" \
  data.max_response_length="$ROLLOUT_RESP_LEN" \
  data.train_batch_size="$STAGE2_TRAIN_BATCH_SIZE" \
  data.filter_overlong_prompts=true \
  data.truncation="$DATA_TRUNCATION" \
  data.dataloader_num_workers=0 \
  ray_kwargs.ray_init.num_cpus=null \
  "${RAY_RUNTIME_ENV_ARGS[@]}" \
  "${REWARD_MANAGER_ARGS[@]}" \
  data.sampler.class_path=pkg://verl.experimental.dataset.two_stage_sampler \
  data.sampler.class_name=TwoStageRewardFilterSampler \
  data.sampler.drop_list_path="$STAGE2_DROP_LIST" \
  data.sampler.include_categories=null \
  data.sampler.keep_if_group_id_missing=true \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$STAGE2_EXPERIMENT" \
  trainer.default_local_dir="$STAGE2_OUTPUT_DIR" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$STAGE1_CKPT" \
  trainer.total_training_steps="$STAGE2_TOTAL_STEPS" \
  trainer.total_epochs=100000 \
  trainer.save_freq="$STAGE2_SAVE_FREQ" \
  trainer.n_gpus_per_node="$STAGE2_NGPUS"
