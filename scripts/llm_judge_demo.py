#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import httpx
from omegaconf import OmegaConf


def _load_category_prompts() -> dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    prompts_path = script_dir / ".." / "verl" / "workers" / "reward_manager" / "llm_prompts.py"
    prompts_path = prompts_path.resolve()
    spec = importlib.util.spec_from_file_location("llm_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "CATEGORY_PROMPTS", {})
    if not isinstance(prompts, dict):
        raise RuntimeError("CATEGORY_PROMPTS is missing or invalid in llm_prompts.py")
    return {str(k): str(v) for k, v in prompts.items()}


def _default_config_dir() -> str:
    return str(
        (
            Path(__file__).resolve().parent
            / ".."
            / "verl"
            / "trainer"
            / "config"
        ).resolve()
    )


def _default_config_name(config_dir: str) -> str:
    candidate = Path(config_dir) / "exp" / "ppo_two_stage_stage1.yaml"
    if candidate.exists():
        return "exp/ppo_two_stage_stage1"
    return "ppo_trainer"


def _load_reward_judge_cfg(config_dir: str, config_name: str, overrides: list[str]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(config_dir)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    try:
        judge_node = cfg.reward_model.reward_kwargs.judge
    except Exception:
        return {}
    if judge_node is None:
        return {}
    judge_cfg = OmegaConf.to_container(judge_node, resolve=True) or {}
    if not isinstance(judge_cfg, dict):
        return {}
    return judge_cfg


@dataclass
class LLMJudgeConfig:
    """Configuration for the external LLM judge."""

    api_url: str
    model: str
    api_key: str | None = None
    prompt_template: str | None = None
    category_prompts: dict[str, str] | None = None
    temperature: float = 0.0
    timeout: float = 30.0
    reasoning_effort: str | None = None
    max_retries: int | None = None
    retry_interval: float = 1.0
    max_concurrency: int = 8


def _safe_json_loads(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except Exception:
        pass

    try:
        start = payload.index("{")
        end = payload.rindex("}") + 1
    except ValueError:
        return None

    try:
        return json.loads(payload[start:end])
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _build_eval_input(prompt: str, q: str, e: str, a: str) -> str:
    return (
        _safe_text(prompt).rstrip()
        + "\n\nQ:\n"
        + _safe_text(q).strip()
        + "\n\nE:\n"
        + _safe_text(e).strip()
        + "\n\nA:\n"
        + _safe_text(a).strip()
    )


class LLMJudgeClient:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.category_prompts = {k.lower(): v for k, v in (self.config.category_prompts or {}).items() if v}
        if not self.category_prompts and not self.config.prompt_template:
            raise ValueError("LLM judge prompts are empty.")
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    def _render_prompt_template(self, template: str, template_vars: dict[str, str]) -> str:
        sentinel_l = "__VERL_LBRACE__"
        sentinel_r = "__VERL_RBRACE__"
        escaped = template.replace("{{", sentinel_l).replace("}}", sentinel_r)
        escaped = escaped.replace("{", "{{").replace("}", "}}")
        for key in template_vars.keys():
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
        escaped = escaped.replace(sentinel_l, "{{").replace(sentinel_r, "}}")
        return escaped.format_map(template_vars)

    def _build_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        cat = str(sample["category"]).lower()
        question_text = sample.get("question_text", "")
        gold_text = sample.get("gold_text", "")
        template_vars = defaultdict(
            str,
            {
            "input": sample["input_text"],
            "raw_input": sample["input_text"],
            "output": sample["output_text"],
            "model_output": sample["output_text"],
            "q": question_text,
            "question": question_text,
            "e": gold_text,
            "gold": gold_text,
            "a": sample["output_text"],
            "category": cat,
            "Category": cat,
            },
        )
        prompt_template = self.category_prompts.get(cat) or self.config.prompt_template
        if not prompt_template:
            raise ValueError(f"Missing judge prompt for category={cat}")
        rendered_prompt = self._render_prompt_template(prompt_template, template_vars)
        content = _build_eval_input(rendered_prompt, question_text, gold_text, sample["output_text"])
        return [{"role": "user", "content": content}]

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        consistency = payload.get("consistency", None)
        contains_target_error = payload.get("contains_target_error", payload.get("contain_targets_error", None))

        if contains_target_error is None:
            contains_target_error = payload.get("existence", payload.get("existance", None))

        location = payload.get("location", None)
        reasoning = payload.get("reasoning", {})
        unrelated_errors = payload.get("unrelated_errors", None)
        if unrelated_errors is None and isinstance(reasoning, dict):
            unrelated_errors = reasoning.get("unrelated_errors", None)

        if consistency is None and contains_target_error is None:
            consistency = 1.0

        if unrelated_errors is None or not isinstance(reasoning, dict):
            return None

        try:
            location_present = isinstance(location, dict)
            location = location if location_present else {}
            normalized = {
                "consistency": float(consistency),
                "contains_target_error": float(contains_target_error or 0),
                "location": {
                    "gold_steps": int(location.get("gold_steps", 0) or 0),
                    "hit_steps": int(location.get("hit_steps", 0) or 0),
                    "extra_steps": int(location.get("extra_steps", 0) or 0),
                },
                "location_present": location_present,
                "reasoning": {
                    "gold_steps": int(reasoning.get("gold_steps", 0) or 0),
                    "reached_steps": int(reasoning.get("reached_steps", 0) or 0),
                },
                "unrelated_errors": int(unrelated_errors or 0),
            }
        except Exception:
            return None

        if normalized["consistency"] <= 0:
            normalized["contains_target_error"] = 0.0
            normalized["location"] = {"gold_steps": 0, "hit_steps": 0, "extra_steps": 0}
            normalized["reasoning"] = {"gold_steps": 0, "reached_steps": 0}
            normalized["unrelated_errors"] = 0

        return normalized

    def _parse_response_json(self, response_json: Any) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        if isinstance(response_json, dict):
            if {
                "consistency",
                "contains_target_error",
                "reasoning",
                "unrelated_errors",
            } <= response_json.keys() or {
                "consistency",
                "contain_targets_error",
                "reasoning",
                "unrelated_errors",
            } <= response_json.keys() or {
                "existence",
                "location",
                "reasoning",
                "unrelated_errors",
            } <= response_json.keys() or {
                "existance",
                "location",
                "reasoning",
                "unrelated_errors",
            } <= response_json.keys():
                candidates.append(response_json)
            if "choices" in response_json:
                choices = response_json.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if isinstance(content, dict):
                        candidates.append(content)
                    elif isinstance(content, str):
                        parsed = _safe_json_loads(content)
                        if parsed:
                            candidates.append(parsed)
        elif isinstance(response_json, str):
            parsed = _safe_json_loads(response_json)
            if parsed:
                candidates.append(parsed)

        for candidate in candidates:
            normalized = self._normalize_payload(candidate)
            if normalized is not None:
                return normalized
        return None

    async def _score_once(self, client: httpx.AsyncClient, sample: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": self._build_messages(sample),
            "temperature": self.config.temperature,
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort

        async with self._semaphore:
            resp = await client.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
        resp.raise_for_status()
        return resp.json()


def _compute_reward_from_judge(
    judge_json: dict[str, Any],
    category: str,
    category_weights: dict[str, dict[str, float]],
    lambda_unrel: float,
) -> tuple[float, dict[str, float]]:
    cat = category.lower()
    weights = category_weights.get(cat, category_weights.get("default", {}))
    s_consistency = float(judge_json.get("consistency", 0.0))
    s_contains_target_error = float(judge_json.get("contains_target_error", 0.0))
    loc = judge_json.get("location", {}) or {}
    loc_present = bool(judge_json.get("location_present", True))
    reasoning = judge_json.get("reasoning", {}) or {}
    unrelated_errors = max(int(judge_json.get("unrelated_errors", 0)), 0)

    if loc_present:
        g = int(loc.get("gold_steps", 0) or 0)
        h = int(loc.get("hit_steps", 0) or 0)
        e = int(loc.get("extra_steps", 0) or 0)
        fn = max(g - h, 0)
        denom = max(2 * h + e + fn, 1)
        s_loc = (2.0 * h) / denom
    else:
        s_loc = 0.0

    rg = int(reasoning.get("gold_steps", 0) or 0)
    rr = int(reasoning.get("reached_steps", 0) or 0)
    if rg <= 0:
        s_reason = 1.0
    else:
        s_reason = min(max(rr / float(rg), 0.0), 1.0)

    p_unrel = pow(2.718281828, -lambda_unrel * unrelated_errors)

    if s_consistency <= 0:
        s_loc = 0.0
        s_reason = 0.0
        p_unrel = 0.0
    elif s_contains_target_error <= 0:
        s_loc = 0.0
        s_reason = 0.0

    weighted_sum = (
        weights.get("w_loc", 0.0) * s_loc
        + weights.get("w_reason", 0.0) * s_reason
        + weights.get("w_unrel", 0.0) * p_unrel
    )
    raw_reward = s_consistency * weighted_sum

    extra = {
        "judge/s_consistency": s_consistency,
        "judge/s_contains_target_error": s_contains_target_error,
        "judge/s_exist": s_consistency,
        "judge/s_loc": s_loc,
        "judge/s_reason": s_reason,
        "judge/p_unrel": p_unrel,
        "judge/weighted_sum": weighted_sum,
    }
    return raw_reward, extra


CATEGORY_PROMPTS_OVERRIDE = _load_category_prompts()

# Sample input for the demo.
CATEGORY = "reason"
SAMPLE = {
    "input_text": "A report claims the total is 120, but the table lists 60 and 50.",
    "output_text": "There is an error: the stated total (120) does not match 60 + 50 = 110.",
    "category": CATEGORY,
    "question_text": "Is there a numerical inconsistency in the report?",
    "gold_text": "Answer: ERROR. The total is inconsistent with the table entries.",
    "meta": {},
}

# Reward settings (same defaults as LLM judge manager).
LAMBDA_UNRELATED = 0.4
CATEGORY_WEIGHTS = {
    "scan": {"w_loc": 0.4, "w_reason": 0.4, "w_unrel": 0.2},
    "verify": {"w_loc": 0.0, "w_reason": 0.6, "w_unrel": 0.4},
    "reason": {"w_loc": 0.0, "w_reason": 0.6, "w_unrel": 0.4},
}


def _init_progress(total: int):
    if total <= 1:
        return None
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc="llm_judge_demo", leave=False, ascii=True, dynamic_ncols=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM judge demo using composed trainer config.")
    parser.add_argument(
        "--config-dir",
        default=_default_config_dir(),
        help="Hydra config directory (default: verl/trainer/config)",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Hydra config name (e.g. exp/ppo_two_stage_stage1)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (can be used multiple times)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Number of samples to send (default: env LLM_JUDGE_DEMO_BATCH or 1)",
    )
    return parser.parse_args()


async def _run_demo(args: argparse.Namespace) -> int:
    config_name = args.config_name or os.getenv("LLM_JUDGE_CONFIG_NAME")
    if not config_name:
        config_name = _default_config_name(args.config_dir)
    overrides = list(args.override)
    env_overrides = os.getenv("LLM_JUDGE_OVERRIDES")
    if env_overrides:
        overrides.extend([item for item in env_overrides.split(",") if item])

    judge_cfg = _load_reward_judge_cfg(args.config_dir, config_name, overrides)
    if not judge_cfg:
        print(
            "Missing reward_model.reward_kwargs.judge in composed config. "
            "Check --config-name or add overrides.",
            file=sys.stderr,
        )
        return 2

    allowed_keys = {field.name for field in fields(LLMJudgeConfig)}
    judge_cfg = {k: v for k, v in judge_cfg.items() if k in allowed_keys}

    env_api_url = os.getenv("LLM_JUDGE_API_URL")
    if env_api_url:
        judge_cfg["api_url"] = env_api_url
    env_model = os.getenv("LLM_JUDGE_MODEL")
    if env_model:
        judge_cfg["model"] = env_model
    env_api_key = os.getenv("LLM_JUDGE_API_KEY")
    if env_api_key:
        judge_cfg["api_key"] = env_api_key

    if not judge_cfg.get("api_url"):
        print(
            "Missing judge api_url. Set reward_kwargs.judge.api_url in reward_model.yaml "
            "or LLM_JUDGE_API_URL.",
            file=sys.stderr,
        )
        return 2
    if not judge_cfg.get("model"):
        print(
            "Missing judge model. Set reward_kwargs.judge.model in reward_model.yaml "
            "or LLM_JUDGE_MODEL.",
            file=sys.stderr,
        )
        return 2

    judge_cfg["prompt_template"] = judge_cfg.get("prompt_template") or None
    default_prompts = {k.lower(): v for k, v in CATEGORY_PROMPTS_OVERRIDE.items() if v}
    user_prompts = {k.lower(): v for k, v in (judge_cfg.get("category_prompts") or {}).items() if v}
    judge_cfg["category_prompts"] = {**default_prompts, **user_prompts} or None

    if not judge_cfg["category_prompts"] and not judge_cfg["prompt_template"]:
        print("No prompts available; fill CATEGORY_PROMPTS or reward_kwargs.judge.prompt_template.", file=sys.stderr)
        return 2

    client = LLMJudgeClient(LLMJudgeConfig(**judge_cfg))

    batch_env = os.getenv("LLM_JUDGE_DEMO_BATCH")
    batch_size = args.batch if args.batch is not None else int(batch_env or "1")
    batch_size = max(batch_size, 1)
    samples = [SAMPLE] * batch_size
    parsed_list: list[dict[str, Any] | None] = [None] * batch_size

    async with httpx.AsyncClient() as http_client:
        if batch_size == 1:
            raw = await client._score_once(client=http_client, sample=SAMPLE)
            parsed_list[0] = client._parse_response_json(raw)
            print("RAW_RESPONSE:", json.dumps(raw))
        else:
            progress = _init_progress(batch_size)

            async def _score_with_idx(idx: int, sample: dict[str, Any]):
                return idx, await client._score_once(client=http_client, sample=sample)

            tasks = [asyncio.create_task(_score_with_idx(idx, sample)) for idx, sample in enumerate(samples)]
            try:
                for task in asyncio.as_completed(tasks):
                    idx, raw = await task
                    parsed_list[idx] = client._parse_response_json(raw)
                    if progress is not None:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

    if any(parsed is None for parsed in parsed_list):
        print("PARSED_RESPONSE: null", file=sys.stderr)
        return 1

    parsed = parsed_list[0]
    print("PARSED_RESPONSE:", json.dumps(parsed))

    reward, extras = _compute_reward_from_judge(
        parsed,
        category=CATEGORY,
        category_weights=CATEGORY_WEIGHTS,
        lambda_unrel=LAMBDA_UNRELATED,
    )
    print("REWARD:", reward)
    print("REWARD_EXTRAS:", json.dumps(extras))

    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run_demo(args))


if __name__ == "__main__":
    raise SystemExit(main())
