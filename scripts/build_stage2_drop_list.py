#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _build_group_id(record: dict[str, Any]) -> str | None:
    group_id = _norm(record.get("group_id"))
    if group_id:
        return group_id

    extra_info = record.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    orig_id = _norm(record.get("orig_id")) or _norm(extra_info.get("orig_id")) or _norm(extra_info.get("id"))
    orig_index = _norm(record.get("orig_index")) or _norm(extra_info.get("orig_index")) or _norm(extra_info.get("index"))
    question_kind = (
        _norm(record.get("question_kind"))
        or _norm(extra_info.get("question_kind"))
        or _norm(record.get("Category"))
        or _norm(record.get("category"))
    )

    parts = [p for p in (orig_id, orig_index, question_kind) if p is not None]
    if parts:
        return "::".join(parts)
    return None


def _pick_score(record: dict[str, Any], preferred_key: str | None) -> float | None:
    if preferred_key:
        value = record.get(preferred_key)
        if value is not None:
            return float(value)

    for key in ("judge/reward", "judge/weighted_sum", "score"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def iter_jsonl_files(rollout_dir: str):
    pattern = os.path.join(rollout_dir, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage2 drop list from Stage1 rollout logs.")
    parser.add_argument("--rollout_dir", required=True, help="Directory of Stage1 rollout JSONL logs.")
    parser.add_argument("--out_path", required=True, help="Output JSONL drop list path.")
    parser.add_argument("--mean_threshold", type=float, default=0.9)
    parser.add_argument("--var_threshold", type=float, default=0.02)
    parser.add_argument("--min_count", type=int, default=8)
    parser.add_argument("--score_key", default=None, help="Preferred score key (default uses judge/reward).")
    args = parser.parse_args()

    groups: dict[str, list[float]] = defaultdict(list)
    group_meta: dict[str, dict[str, Any]] = {}

    total = 0
    skipped = 0
    for path in iter_jsonl_files(args.rollout_dir):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)
                group_id = _build_group_id(record)
                score = _pick_score(record, args.score_key)
                if group_id is None or score is None:
                    skipped += 1
                    continue
                groups[group_id].append(float(score))
                if group_id not in group_meta:
                    group_meta[group_id] = {
                        "Category": record.get("Category") or record.get("category"),
                        "question_kind": record.get("question_kind"),
                    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    kept = 0
    dropped = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for group_id, scores in groups.items():
            if len(scores) < args.min_count:
                continue
            mean_val = sum(scores) / float(len(scores))
            var_val = sum((x - mean_val) ** 2 for x in scores) / float(len(scores))
            if mean_val > args.mean_threshold and var_val < args.var_threshold:
                dropped += 1
                payload = {
                    "group_id": group_id,
                    "mean": mean_val,
                    "var": var_val,
                    "count": len(scores),
                }
                payload.update(group_meta.get(group_id, {}))
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                kept += 1

    print(
        f"[DONE] total_records={total} skipped_records={skipped} "
        f"groups={len(groups)} dropped={dropped} kept={kept} out={args.out_path}"
    )


if __name__ == "__main__":
    main()
