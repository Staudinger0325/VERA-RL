"""Helper to debug LLM judge API calls.

Usage:
  python -m verl.workers.reward_manager.api_debug \\
    --input "question text" \\
    --output "model answer" \\
    --category scan \\
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \\
    --model Qwen-30B-A3B \\
    --prompt-file /path/to/prompt.txt

Prompts:
  - If --prompt-file is given, it is used as fallback prompt_template.
  - Per-category prompts are loaded from verl/workers/reward_manager/llm_prompts.py.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import httpx

from verl.workers.reward_manager.llm_judge import LLMJudgeClient, LLMJudgeConfig
from verl.workers.reward_manager.llm_prompts import CATEGORY_PROMPTS


async def _run_once(args: argparse.Namespace) -> dict[str, Any]:
    prompt_text = Path(args.prompt_file).read_text() if args.prompt_file else None
    judge_cfg = LLMJudgeConfig(
        api_url=args.api_url,
        model=args.model,
        api_key=args.api_key,
        prompt_template=prompt_text,
        category_prompts={k: v for k, v in CATEGORY_PROMPTS.items() if v},
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    client = LLMJudgeClient(judge_cfg)
    sample = {
        "input_text": args.input,
        "output_text": args.output,
        "category": args.category,
    }
    async with httpx.AsyncClient() as http_client:
        parsed = await client.score_sample(http_client, sample)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Debug LLM judge API call.")
    parser.add_argument("--input", required=True, help="Input text/prompt")
    parser.add_argument("--output", required=True, help="Model output text")
    parser.add_argument("--category", required=True, choices=["scan", "verify", "reason"])
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", default="Qwen-30B-A3B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-file", default=None, help="Path to fallback prompt template")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=None)
    args = parser.parse_args()

    parsed = asyncio.run(_run_once(args))
    print("Parsed judge JSON:")
    print(parsed)


if __name__ == "__main__":
    main()

