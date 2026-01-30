# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("zero")
class ZeroRewardManager(AbstractRewardManager):
    """Reward manager that returns zero rewards for all samples."""

    def __init__(self, tokenizer=None, num_examine=0, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
