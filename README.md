# VERA-RL

Code for **Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Papers**.

This repository is built on [verl](https://github.com/volcengine/verl).

## Resources

| Artifact | Hugging Face repository |
| --- | --- |
| VERA-13K dataset | [`Stdg/VERA-13K`](https://huggingface.co/datasets/Stdg/VERA-13K) |
| SFT checkpoint | [`Stdg/VERA-SFT`](https://huggingface.co/Stdg/VERA-SFT) |
| SFT+RL checkpoint | [`Staudinger/VERA-RL`](https://huggingface.co/Staudinger/VERA-RL) |

## Setup

Install a CUDA-compatible verl environment, then install this repository:

```bash
pip install -e ".[vllm,gpu,geo]"
```

Avoid PyTorch 2.9.0 when training Qwen3-VL-8B-Instruct because its vision `Conv3d` may fall back to very slow kernels. PyTorch 2.8.x is recommended for the provided environment.

## Download Data and Checkpoints

Download the released data and checkpoints:

```bash
hf download Stdg/VERA-13K \
  --type dataset \
  --local-dir data/VERA-13K

hf download Stdg/VERA-SFT \
  --local-dir checkpoints/vera-rl-sft

hf download Staudinger/VERA-RL \
  --local-dir checkpoints/vera-rl-sft-rl
```

The training scripts use:

```text
data/VERA-13K/train_SFT.parquet
data/VERA-13K/train_RL.parquet
```

## Reward API

Set `reward_kwargs.judge.model` in `verl/trainer/config/reward_model/reward_model.yaml` and provide the credential through the environment:

```bash
export ARK_API_KEY="YOUR_API_KEY"
```

Keep `reward_kwargs.judge.api_key` empty. Reward prompts are defined in `verl/workers/reward_manager/llm_prompts.py`.

## Training

Run all commands from the repository root. Training parameters are defined in the launch scripts; paths can be supplied through environment variables.

### SFT

```bash
mkdir -p logs
export BASE_MODEL="$PWD/checkpoints/Qwen3-VL-8B-Instruct"
export TOKENIZER_PATH="$BASE_MODEL"
export TRAIN_FILES="$PWD/data/VERA-13K/train_SFT.parquet"
export OUTPUT_DIR="$PWD/outputs/vera-rl-sft"
export SP_SIZE=YOUR_SEQUENCE_PARALLEL_SIZE
export MAX_LEN=YOUR_MAX_SEQUENCE_LENGTH
export PROJECT_NAME=vera-rl
export EXPERIMENT_NAME=sft

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash sft_train.sh 2>&1 \
  | tee "logs/sft_$(date +%Y%m%d_%H%M%S).log"
```

Merge the final SFT checkpoint before using it for RL.

### RL environment

The following variables are shared by the RL launchers:

```bash
export BASE_MODEL="$PWD/checkpoints/vera-rl-sft"
export TOKENIZER_PATH="$BASE_MODEL"
export TRAIN_RL_PATH="$PWD/data/VERA-13K/train_RL.parquet"
export ROLLOUT_MAX_MODEL_LEN=52048
export ROLLOUT_MAX_BATCHED_TOKENS=52048
export ROLLOUT_PROMPT_LEN=50000
mkdir -p logs
```

### Main RL

```bash
ray stop --force
export STAGE1_OUTPUT_DIR="$PWD/outputs/main_rl"
export STAGE1_ROLLOUT_DIR="$PWD/outputs/main_rl_rollouts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/rl_for_vera/run_main.sh 2>&1 \
  | tee "logs/main_rl_$(date +%Y%m%d_%H%M%S).log"
```

### Scan-only RL

```bash
ray stop --force
export STAGE1_OUTPUT_DIR="$PWD/outputs/scan_only_rl"
export STAGE1_ROLLOUT_DIR="$PWD/outputs/scan_only_rl_rollouts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/rl_for_vera/run_scan.sh 2>&1 \
  | tee "logs/scan_only_rl_$(date +%Y%m%d_%H%M%S).log"
```

### Curricular RL

Run the main phase first, then resume its checkpoint with the Scan-only launcher:

```bash
ray stop --force
export STAGE1_OUTPUT_DIR="$PWD/outputs/curricular_main"
export STAGE1_ROLLOUT_DIR="$PWD/outputs/curricular_main_rollouts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  STAGE1_TOTAL_STEPS=15 \
  bash examples/rl_for_vera/run_main.sh 2>&1 \
  | tee "logs/curricular_main_$(date +%Y%m%d_%H%M%S).log"

ray stop --force
export STAGE1_RESUME_FROM="$PWD/outputs/curricular_main/global_step_15"
export STAGE1_OUTPUT_DIR="$PWD/outputs/curricular_rl"
export STAGE1_ROLLOUT_DIR="$PWD/outputs/curricular_rl_rollouts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  STAGE1_TOTAL_STEPS=30 \
  bash examples/rl_for_vera/run_scan.sh 2>&1 \
  | tee "logs/curricular_rl_$(date +%Y%m%d_%H%M%S).log"
```

Additional ablations are available in `examples/rl_for_vera/run_reason_only.sh` and `examples/rl_for_vera/run_no_unrelated.sh`.

## Merge FSDP Checkpoints

SFT checkpoint:

```bash
PYTHONPATH=. python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/sft/global_step_FINAL \
  --target_dir checkpoints/vera-rl-sft
```

RL actor checkpoint:

```bash
PYTHONPATH=. python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/rl/global_step_FINAL/actor \
  --target_dir /path/to/merged_huggingface_model
```

## License

This project retains the upstream verl notices. See [LICENSE](LICENSE) and [Notice.txt](Notice.txt).
