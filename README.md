# Anonymous Code Repository (ACL May 2026 Submission)

This anonymous repository contains the code for the ACl May 2026 submission:

**Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Papers**

Our implementation is based on **verl**. Please first follow the official verl documentation to set up the environment.

We observed an environment-related slowdown when training with **Qwen3-VL-8B-Instruct** as the base model. With **PyTorch 2.9.0**, the Qwen3-VL vision patch embedding layer (`model.visual.patch_embed.proj`, implemented as `nn.Conv3d`) may fall back to slow 3DConv kernels such as `aten::slow_conv_dilated3d`, making multimodal training updates extremely slow. This appears to be an upstream PyTorch `Conv3d`/cuDNN regression rather than an issue in this repository. If you encounter this problem, we recommend avoiding PyTorch 2.9.0 and using **PyTorch 2.8.x** or a newer fixed version such as **PyTorch 2.10+**. See PyTorch issues [#174051](https://github.com/pytorch/pytorch/issues/174051), [#166122](https://github.com/pytorch/pytorch/issues/166122), and Qwen3-VL issue [#1701](https://github.com/QwenLM/Qwen3-VL/issues/1701).

## Overview of Added Code and Scripts for VERA-RL

### 1) Reward Model API Setup

We use **Volc Engine**’s official **Seed-1.6** service as a rule-based reward model to score rollouts.

To enable it, fill in the API key and the corresponding model endpoint in:

`verl/trainer/config/reward_model/reward_model.yaml`

The prompt field in the YAML is optional, since prompts are already defined in:

`verl/workers/reward_manager/llm_prompts.py`

### 2) Training Scripts

We provide the following scripts:

- **SFT training script**:  
  `sft_train.sh`

- **RL-related scripts** (all RL scripts are under):  
  `examples/rl_for_vera`  
  To use them, you need to set key paths such as `BASE_MODEL`, `TOKENIZER_PATH`, `TRAIN_FILES`, `OUTPUT_DIR`, etc.

Example configuration:

- For reproducing **SFT**, set:
  - `BASE_MODEL`: path to **Qwen3-VL-8B-Instruct**
  - `TOKENIZER_PATH`: path to `tokenizer.json` (usually the same as `BASE_MODEL`)
  - `TRAIN_FILES`: path to `train_SFT.parquet`
  - `OUTPUT_DIR`: output directory for saving checkpoints

- Reproducing **RL** follows the same logic.

### 3) Dataset and Models

Due to storage limitations of the anonymous repository, we temporarily host the **VERA-13K** dataset on Hugging Face:

https://huggingface.co/datasets/anonymous0x7e/VERA-13K
