#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import datasets
import torch
from transformers import AutoProcessor, AutoTokenizer

# Allow running as a standalone script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from verl.utils.dataset.vision_utils import process_image, process_video


@dataclass(frozen=True)
class CountConfig:
    model_path: str
    tokenizer_path: str | None
    prompt_key: str
    image_key: str
    video_key: str | None
    image_patch_size: int
    apply_chat_template_kwargs: dict[str, Any]
    tool_config_path: str | None
    disable_processor: bool


_STATE: dict[str, Any] = {}


def _load_tool_schemas(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    from verl.tools.utils.tool_registry import initialize_tools_from_config

    tool_list = initialize_tools_from_config(path)
    return [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]


def _get_state(cfg: CountConfig) -> dict[str, Any]:
    cache_key = (
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.disable_processor,
        tuple(sorted(cfg.apply_chat_template_kwargs.items())),
        cfg.tool_config_path,
    )
    cached = _STATE.get("state")
    if cached and cached["cache_key"] == cache_key:
        return cached

    tokenizer_path = cfg.tokenizer_path or cfg.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    processor = None
    if not cfg.disable_processor:
        try:
            processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        except Exception:
            processor = None

    tool_schemas = _load_tool_schemas(cfg.tool_config_path)

    cached = {
        "cache_key": cache_key,
        "tokenizer": tokenizer,
        "processor": processor,
        "tool_schemas": tool_schemas,
    }
    _STATE["state"] = cached
    return cached


def _extract_type(example: dict) -> str:
    extra = example.get("extra_info") or {}
    if isinstance(extra, dict):
        for key in ("Type", "type"):
            val = extra.get(key)
            if val is not None and str(val):
                return str(val)
    for key in ("Type", "type", "ability", "data_source", "question_kind"):
        val = example.get(key)
        if val is not None and str(val):
            return str(val)
    return "unknown"


def _build_messages(example: dict, prompt_key: str, image_key: str, video_key: str | None) -> list[dict]:
    raw_messages = example.get(prompt_key)
    if not raw_messages:
        return []
    messages = copy.deepcopy(raw_messages)

    has_videos = video_key is not None and video_key in example and example.get(video_key)
    if not has_videos:
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and "<video>" in content:
                message["content"] = content.replace("<video>", "")
            elif isinstance(content, list):
                filtered = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        continue
                    filtered.append(item)
                message["content"] = filtered

    if image_key in example or (video_key is not None and video_key in example):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                elif segment == "<video>":
                    content_list.append({"type": "video"})
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

    return messages


def _count_tokens(example: dict, cfg: CountConfig) -> int:
    state = _get_state(cfg)
    tokenizer = state["tokenizer"]
    processor = state["processor"]
    tool_schemas = state["tool_schemas"]

    messages = _build_messages(example, cfg.prompt_key, cfg.image_key, cfg.video_key)
    if processor is not None:
        apply_kwargs = dict(cfg.apply_chat_template_kwargs)
        if tool_schemas is not None:
            apply_kwargs["tools"] = tool_schemas
        raw_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )

        images = None
        if cfg.image_key in example and example.get(cfg.image_key):
            images = [process_image(image, image_patch_size=cfg.image_patch_size) for image in example[cfg.image_key]]

        videos = None
        videos_kwargs: dict[str, Any] = {}
        if cfg.video_key is not None and cfg.video_key in example and example.get(cfg.video_key):
            videos, video_metadata = zip(
                *[
                    process_video(video, image_patch_size=cfg.image_patch_size, return_video_metadata=True)
                    for video in example[cfg.video_key]
                ],
                strict=True,
            )
            videos = list(videos)
            video_metadata = list(video_metadata)
            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

        model_inputs = processor(
            text=[raw_prompt],
            images=images,
            videos=videos,
            videos_kwargs=videos_kwargs,
            return_tensors="pt",
        )
        return int(model_inputs["input_ids"].shape[-1])

    apply_kwargs = dict(cfg.apply_chat_template_kwargs)
    if tool_schemas is not None:
        apply_kwargs["tools"] = tool_schemas
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kwargs,
    )
    model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
    return int(model_inputs["input_ids"].shape[-1])


def _compute_row(example: dict, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = CountConfig(**cfg_dict)
    type_name = _extract_type(example)
    try:
        token_count = _count_tokens(example, cfg)
        err = None
    except Exception as exc:  # noqa: PERF203
        token_count = -1
        err = f"{type(exc).__name__}: {exc}"
    return {"token_count": token_count, "type_name": type_name, "error": err}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    counters: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "sum": 0.0, "max": 0, "errors": 0})
    for row in rows:
        t = row["type_name"]
        token_count = row["token_count"]
        if token_count is None or token_count < 0:
            counters[t]["errors"] += 1
            continue
        counters[t]["count"] += 1
        counters[t]["sum"] += float(token_count)
        counters[t]["max"] = max(counters[t]["max"], token_count)

    for t, c in sorted(counters.items()):
        count = c["count"]
        avg = c["sum"] / count if count else 0.0
        stats[t] = {
            "count": int(count),
            "avg_tokens": avg,
            "max_tokens": int(c["max"]),
            "errors": int(c["errors"]),
        }
    return stats


def _print_summary(label: str, stats: dict[str, dict[str, float]]) -> None:
    print(f"\n== {label} ==")
    if not stats:
        print("No rows found.")
        return
    for t, s in stats.items():
        print(
            f"{t}\tcount={s['count']}\tavg_tokens={s['avg_tokens']:.2f}"
            f"\tmax_tokens={s['max_tokens']}\terrors={s['errors']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Count tokens per sample and aggregate by Type.")
    parser.add_argument("--data-files", nargs="+", required=True, help="Parquet file paths.")
    parser.add_argument("--model-path", required=True, help="Model path for AutoProcessor.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path (defaults to model path).")
    parser.add_argument("--prompt-key", default="messages", help="Prompt/messages column name.")
    parser.add_argument("--image-key", default="images", help="Image column name.")
    parser.add_argument("--video-key", default="videos", help="Video column name (use 'null' to disable).")
    parser.add_argument("--image-patch-size", type=int, default=14, help="Image patch size for vision utils.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for datasets.map.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit number of samples per file.")
    parser.add_argument("--output-jsonl", default=None, help="Optional JSONL output path for per-sample counts.")
    parser.add_argument("--output-summary", default=None, help="Optional JSON output path for summary.")
    parser.add_argument("--apply-chat-template-kwargs", default=None, help="JSON dict for apply_chat_template kwargs.")
    parser.add_argument("--tool-config-path", default=None, help="Optional tool config for chat template.")
    parser.add_argument("--disable-processor", action="store_true", help="Force text-only tokenizer path.")
    args = parser.parse_args()

    video_key = None if args.video_key in ("null", "None", "") else args.video_key
    apply_kwargs = {}
    if args.apply_chat_template_kwargs:
        apply_kwargs = json.loads(args.apply_chat_template_kwargs)

    cfg_dict = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "prompt_key": args.prompt_key,
        "image_key": args.image_key,
        "video_key": video_key,
        "image_patch_size": args.image_patch_size,
        "apply_chat_template_kwargs": apply_kwargs,
        "tool_config_path": args.tool_config_path,
        "disable_processor": args.disable_processor,
    }

    all_rows: list[dict[str, Any]] = []
    for data_file in args.data_files:
        dataset = datasets.load_dataset("parquet", data_files=data_file)["train"]
        if args.max_samples > 0 and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))

        mapped = dataset.map(
            _compute_row,
            fn_kwargs={"cfg_dict": cfg_dict},
            num_proc=args.num_proc,
            desc=f"Counting tokens for {os.path.basename(data_file)}",
        )
        rows = [
            {"token_count": tc, "type_name": tn, "error": err}
            for tc, tn, err in zip(mapped["token_count"], mapped["type_name"], mapped["error"])
        ]
        stats = _summarize(rows)
        _print_summary(data_file, stats)

        if args.output_jsonl:
            with open(args.output_jsonl, "a", encoding="utf-8") as f:
                for idx, row in enumerate(rows):
                    payload = {"file": data_file, "index": idx, **row}
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        all_rows.extend(rows)

    if args.output_summary:
        summary = _summarize(all_rows)
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
