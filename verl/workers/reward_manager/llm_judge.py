from __future__ import annotations

import asyncio
import logging
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.llm_prompts import CATEGORY_PROMPTS

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class LLMJudgeConfig:
    """Configuration for the external LLM judge."""

    api_url: str
    model: str = "Qwen-30B-A3B"
    api_key: str | None = None
    prompt_template: str | None = None  # fallback prompt
    category_prompts: dict[str, str] | None = None  # per-category prompts, lower-cased keys recommended
    temperature: float = 0.0
    timeout: float = 30.0
    reasoning_effort: str | None = None
    max_retries: int | None = None  # kept for API failures; JSON parse will ignore this and retry indefinitely
    retry_interval: float = 1.0
    max_concurrency: int = 256
    show_progress: bool = True


DEFAULT_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "scan": {"w_loc": 0.4, "w_reason": 0.4, "w_unrel": 0.2},
    "verify": {"w_loc": 0.0, "w_reason": 0.6, "w_unrel": 0.4},
    "reason": {"w_loc": 0.0, "w_reason": 0.6, "w_unrel": 0.4},
}


def _safe_json_loads(payload: str) -> dict[str, Any] | None:
    """Attempt to parse a JSON object from a string."""

    try:
        return json.loads(payload)
    except Exception:
        pass

    # Try extracting the first JSON object in the string
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


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_QUESTION_WINDOW_CHARS = 4000
_QUESTION_MARKS = ("?", "？")
_SENT_BOUNDARY_CHARS = ".!?。！？\n"


def _second_sentence(text: str) -> str:
    cleaned = _safe_text(text).strip()
    if not cleaned:
        return ""
    parts = _SENT_SPLIT_RE.split(cleaned)
    if len(parts) >= 2:
        return parts[1].strip()
    return cleaned


def _third_line_question(text: str) -> str:
    cleaned = _safe_text(text)
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) >= 3:
        candidate = lines[2].strip()
        if candidate:
            return candidate
    saw_non_empty = False
    saw_blank_after_non_empty = False
    for line in lines:
        if line.strip():
            if saw_blank_after_non_empty:
                return line.strip()
            saw_non_empty = True
        else:
            if saw_non_empty:
                saw_blank_after_non_empty = True
    return ""


def _first_question_sentence(text: str) -> str:
    cleaned = _safe_text(text).strip()
    if not cleaned:
        return ""

    window = cleaned[:_QUESTION_WINDOW_CHARS]

    # Prefer explicit "Q:" if present.
    q_match = re.search(r"(?:^|\n)\s*Q\s*[:：]\s*", window)
    if q_match:
        tail = window[q_match.end() :]
        candidate = re.split(r"\n\s*\n", tail, maxsplit=1)[0].strip()
        if candidate:
            return candidate

    first_idx = None
    for mark in _QUESTION_MARKS:
        idx = window.find(mark)
        if idx != -1:
            first_idx = idx if first_idx is None else min(first_idx, idx)
    if first_idx is None:
        return ""

    start = max(window.rfind(ch, 0, first_idx) for ch in _SENT_BOUNDARY_CHARS) + 1
    candidate = window[start : first_idx + 1].strip()
    return candidate.lstrip(" \t-–*#")


class LLMJudgeClient:
    """Async client for calling the external LLM judge."""

    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.category_prompts = {k.lower(): v for k, v in (self.config.category_prompts or {}).items() if v}
        if not self.category_prompts and not self.config.prompt_template:
            raise ValueError(
                "LLM judge prompts are empty. Please fill verl/workers/reward_manager/llm_prompts.py "
                "or set reward_model.reward_kwargs.judge.category_prompts/prompt_template."
            )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._progress_enabled = bool(config.show_progress)

    def _render_prompt_template(self, template: str, template_vars: dict[str, str]) -> str:
        sentinel_l = "__VERL_LBRACE__"
        sentinel_r = "__VERL_RBRACE__"
        escaped = template.replace("{{", sentinel_l).replace("}}", sentinel_r)
        escaped = escaped.replace("{", "{{").replace("}", "}}")
        for key in template_vars.keys():
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
        escaped = escaped.replace(sentinel_l, "{{").replace(sentinel_r, "}}")
        return escaped.format_map(template_vars)

    def _init_progress(self, total: int):
        if not self._progress_enabled or total <= 0:
            return None
        try:
            from tqdm import tqdm
        except Exception:
            return None
        return tqdm(total=total, desc="llm_judge", leave=False, ascii=True, dynamic_ncols=True)

    def _log_progress(self, done: int, total: int) -> None:
        if not self._progress_enabled or total <= 0:
            return
        step = max(1, total // 10)
        if done == 1 or done % step == 0 or done == total:
            logger.info("llm_judge progress %s/%s", done, total)

    def _summarize_response(self, raw: Any) -> str:
        if isinstance(raw, dict):
            if "error" in raw:
                return f"error={raw.get('error')}"
            return f"dict_keys={list(raw.keys())}"
        if isinstance(raw, str):
            return f"str_len={len(raw)}"
        return f"type={type(raw).__name__}"

    def _build_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Render the user message using the configured template."""

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
        """Validate and normalize the judge JSON to the expected schema."""

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
            # Backward compatibility: old schema without consistency.
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
        """Extract the judge JSON from the raw response."""

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
        api_key = self.config.api_key or os.environ.get("ARK_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

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

    async def score_sample(self, client: httpx.AsyncClient, sample: dict[str, Any]) -> dict[str, Any]:
        """Call judge API for one sample with retry on JSON parsing failure."""

        attempt = 0
        last_err: Exception | None = None
        while True:
            attempt += 1
            try:
                raw = await self._score_once(client, sample)
                parsed = self._parse_response_json(raw)
                if parsed is not None:
                    return parsed
                last_err = ValueError(f"Failed to parse judge JSON: {raw}")
                if attempt == 1 or attempt % 10 == 0:
                    logger.warning(
                        "llm_judge parse failed (attempt=%s/%s, category=%s, response=%s)",
                        attempt,
                        self.config.max_retries or "inf",
                        sample.get("category"),
                        self._summarize_response(raw),
                    )
            except Exception as exc:  # noqa: PERF203
                last_err = exc
                if attempt == 1 or attempt % 10 == 0:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    logger.warning(
                        "llm_judge request failed (attempt=%s/%s, category=%s, status=%s): %s",
                        attempt,
                        self.config.max_retries or "inf",
                        sample.get("category"),
                        status,
                        exc,
                    )

            # 对 JSON 解析失败无限重试；仅对 API 异常遵循 max_retries（若配置了）。
            is_parse_error = isinstance(last_err, ValueError)
            if (
                self.config.max_retries is not None
                and self.config.max_retries > 0
                and attempt >= self.config.max_retries
                and not is_parse_error
            ):
                raise RuntimeError(
                    f"LLM judge failed after {attempt} attempt(s) for category={sample['category']}"
                ) from last_err
            await asyncio.sleep(self.config.retry_interval)

    async def score_batch(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run judge calls concurrently for a batch of samples."""

        async with httpx.AsyncClient() as client:
            results: list[dict[str, Any]] = [None] * len(samples)
            progress = self._init_progress(len(samples))

            async def _score_with_idx(idx: int, sample: dict[str, Any]):
                return idx, await self.score_sample(client, sample)

            tasks = [asyncio.create_task(_score_with_idx(idx, sample)) for idx, sample in enumerate(samples)]
            try:
                completed = 0
                for task in asyncio.as_completed(tasks):
                    idx, result = await task
                    results[idx] = result
                    completed += 1
                    if progress is not None:
                        progress.update(1)
                    else:
                        self._log_progress(completed, len(samples))
            finally:
                if progress is not None:
                    progress.close()

            return results


def _compute_reward_from_judge(
    judge_json: dict[str, Any],
    category: str,
    category_weights: dict[str, dict[str, float]],
    lambda_unrel: float,
) -> tuple[float, dict[str, float]]:
    """Compute reward and intermediate scores from judge JSON."""

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

    p_unrel = math.exp(-lambda_unrel * unrelated_errors)

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


def _run_async(coro: Any) -> Any:
    """Run async coroutine from sync context with fallback if loop exists."""

    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "event loop is running" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@register("llm_judge")
class LLMJudgeRewardManager(AbstractRewardManager):
    """Reward manager that queries an external LLM judge and computes structured rewards."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        judge: dict[str, Any] | None = None,
        category_weights: dict[str, dict[str, float]] | None = None,
        lambda_unrelated: float = 0.4,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        judge_cfg = judge or {}
        if not isinstance(judge_cfg, dict):
            try:
                from omegaconf import OmegaConf

                judge_cfg = OmegaConf.to_container(judge_cfg, resolve=True) or {}
            except Exception:
                judge_cfg = dict(judge_cfg)
        else:
            judge_cfg = dict(judge_cfg)

        if "prompt_template" not in judge_cfg:
            judge_cfg["prompt_template"] = None
        default_prompts = {k.lower(): v for k, v in CATEGORY_PROMPTS.items() if v}
        category_prompts = judge_cfg.get("category_prompts") or {}
        user_prompts = {k.lower(): v for k, v in category_prompts.items() if v}
        merged_prompts = {**default_prompts, **user_prompts}
        judge_cfg["category_prompts"] = merged_prompts or None
        self.judge_config = LLMJudgeConfig(**judge_cfg)
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
        self.lambda_unrelated = lambda_unrelated
        self._client = LLMJudgeClient(self.judge_config)

    def _decode_text_pair(self, data_item) -> tuple[str, str, int]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        if torch.is_tensor(valid_prompt_length):
            if valid_prompt_length.numel() != 1:
                valid_prompt_length = valid_prompt_length.sum()
            valid_prompt_length = valid_prompt_length.item()
        valid_prompt_length = int(valid_prompt_length)
        if valid_prompt_length < 0:
            valid_prompt_length = 0
        if valid_prompt_length > prompt_length:
            valid_prompt_length = prompt_length
        valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        if torch.is_tensor(valid_response_length):
            if valid_response_length.numel() != 1:
                valid_response_length = valid_response_length.sum()
            valid_response_length = valid_response_length.item()
        valid_response_length = int(valid_response_length)
        if valid_response_length < 0:
            valid_response_length = 0
        if valid_response_length > response_ids.shape[-1]:
            valid_response_length = response_ids.shape[-1]
        valid_response_ids = response_ids[:valid_response_length]

        def _to_int_ids(ids: Any) -> list[int]:
            if torch.is_tensor(ids):
                if ids.is_floating_point():
                    ids = ids.to(torch.long)
                return ids.tolist()
            if isinstance(ids, (list, tuple)):
                return [int(x) for x in ids]
            return [int(ids)]

        prompt_str = self.tokenizer.decode(_to_int_ids(valid_prompt_ids), skip_special_tokens=True)
        response_str = self.tokenizer.decode(_to_int_ids(valid_response_ids), skip_special_tokens=True)
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token and response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]

        return prompt_str, response_str, int(valid_response_length)

    def _flatten_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    kind = item.get("type")
                    if kind == "text":
                        parts.append(str(item.get("text", "")))
                    elif kind == "image":
                        parts.append("<image>")
                    elif kind == "video":
                        parts.append("<video>")
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _extract_question(self, data_item, prompt_str: str) -> str:
        non_tensor = data_item.non_tensor_batch
        base_text = ""

        raw_prompt = non_tensor.get("raw_prompt")
        if isinstance(raw_prompt, list):
            for msg in raw_prompt:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user":
                    base_text = self._flatten_content(msg.get("content", ""))
                    break
        elif isinstance(raw_prompt, str):
            base_text = raw_prompt

        if not base_text:
            for key in ("prompt", "input", "question", "query"):
                value = non_tensor.get(key)
                if value:
                    base_text = self._flatten_content(value)
                    break

        if not base_text:
            base_text = prompt_str

        question = _third_line_question(base_text)
        if question:
            return question

        question = _first_question_sentence(base_text)
        if question:
            return question
        return _second_sentence(base_text)

    def _extract_gold(self, data_item) -> str:
        non_tensor = data_item.non_tensor_batch
        reward_model = non_tensor.get("reward_model", {}) or {}
        if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
            return str(reward_model.get("ground_truth"))
        for key in ("ground_truth", "gold", "answer"):
            value = non_tensor.get(key)
            if value:
                return str(value)
        return ""

    def _extract_category(self, data_item) -> str:
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        question_kind = (
            extra_info.get("question_kind")
            or data_item.non_tensor_batch.get("question_kind")
        )
        if question_kind is not None:
            return str(question_kind)

        category = (
            extra_info.get("Category")
            or extra_info.get("category")
            or data_item.non_tensor_batch.get("Category")
            or data_item.non_tensor_batch.get("category")
        )
        if category is None:
            raise ValueError(
                "Category/question_kind is missing in data sample; expected one of 'scan', 'verify', or 'reason'."
            )
        return str(category)

    def _extract_group_id(self, data_item, fallback_idx: int, category: str) -> tuple[str, dict[str, str | None]]:
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        def _norm(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value)
            return text if text else None

        orig_id = _norm(extra_info.get("orig_id")) or _norm(extra_info.get("id"))
        orig_index = _norm(extra_info.get("orig_index")) or _norm(extra_info.get("index"))
        question_kind = (
            _norm(extra_info.get("question_kind"))
            or _norm(data_item.non_tensor_batch.get("question_kind"))
            or _norm(extra_info.get("Category"))
            or _norm(data_item.non_tensor_batch.get("Category"))
            or _norm(category)
        )

        parts = [p for p in (orig_id, orig_index, question_kind) if p is not None]
        if parts:
            return "::".join(parts), {
                "orig_id": orig_id,
                "orig_index": orig_index,
                "question_kind": question_kind,
            }

        request_id = _norm(data_item.non_tensor_batch.get("request_id"))
        if request_id is not None:
            return f"request_id::{request_id}", {"orig_id": None, "orig_index": None, "question_kind": None}

        return f"row_{fallback_idx}", {"orig_id": None, "orig_index": None, "question_kind": None}

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[Any]] = defaultdict(list)

        samples: list[dict[str, Any]] = []
        sample_lens: list[int] = []
        categories: list[str] = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_str, response_str, response_len = self._decode_text_pair(data_item)
            category = self._extract_category(data_item)
            group_id, group_meta = self._extract_group_id(data_item, fallback_idx=i, category=category)
            question_text = self._extract_question(data_item, prompt_str)
            gold_text = self._extract_gold(data_item)

            samples.append(
                {
                    "input_text": prompt_str,
                    "output_text": response_str,
                    "category": category,
                    "question_text": question_text,
                    "gold_text": gold_text,
                    "meta": data_item.non_tensor_batch,
                }
            )
            sample_lens.append(response_len)
            categories.append(category)

        logger.info("llm_judge scoring %s samples", len(samples))
        judge_results = _run_async(self._client.score_batch(samples))

        for i, judge_json in enumerate(judge_results):
            logger.info("llm_judge scored %s/%s (category=%s)", i + 1, len(judge_results), categories[i])
            reward, extras = _compute_reward_from_judge(
                judge_json, category=categories[i], category_weights=self.category_weights, lambda_unrel=self.lambda_unrelated
            )
            reward_tensor[i, sample_lens[i] - 1] = reward

            reward_extra_info["judge/raw_response"].append(judge_json)
            reward_extra_info["judge/reward"].append(reward)
            reward_extra_info["Category"].append(categories[i])
            reward_extra_info["group_id"].append(group_id)
            reward_extra_info["orig_id"].append(group_meta.get("orig_id"))
            reward_extra_info["orig_index"].append(group_meta.get("orig_index"))
            reward_extra_info["question_kind"].append(group_meta.get("question_kind"))
            for k, v in extras.items():
                reward_extra_info[k].append(v)

            if i < self.num_examine:
                print("[prompt]", samples[i]["input_text"])
                print("[response]", samples[i]["output_text"])
                print("[category]", categories[i])
                print("[judge_json]", judge_json)
                print("[reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
