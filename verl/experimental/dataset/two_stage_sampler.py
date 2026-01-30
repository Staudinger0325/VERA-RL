from __future__ import annotations

import json
import os
import random
from typing import Iterable

from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


def _norm(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _build_group_id_from_row(row: dict) -> str | None:
    extra_info = row.get("extra_info", {}) or {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    orig_id = _norm(extra_info.get("orig_id")) or _norm(extra_info.get("id"))
    orig_index = _norm(extra_info.get("orig_index")) or _norm(extra_info.get("index"))
    question_kind = (
        _norm(row.get("question_kind"))
        or _norm(extra_info.get("question_kind"))
        or _norm(row.get("Category"))
        or _norm(row.get("category"))
    )

    parts = [p for p in (orig_id, orig_index, question_kind) if p is not None]
    if parts:
        return "::".join(parts)
    return None


def _load_drop_list(path: str | None) -> set[str]:
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"drop_list_path not found: {path}")

    drop_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            group_id = payload.get("group_id")
            if group_id:
                drop_ids.add(str(group_id))
    return drop_ids


class TwoStageRewardFilterSampler(AbstractSampler):
    """Sampler that filters samples by a Stage2 drop list and optional category filter."""

    def __init__(self, data_source, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config

        sampler_cfg = data_config.get("sampler", {}) or {}
        self.drop_list_path = sampler_cfg.get("drop_list_path", None)
        self.include_categories = sampler_cfg.get("include_categories", None)
        self.keep_if_group_id_missing = sampler_cfg.get("keep_if_group_id_missing", True)

        self.drop_ids = _load_drop_list(self.drop_list_path)
        self.include_categories = (
            {str(c).lower() for c in self.include_categories} if self.include_categories else None
        )

        self.indices = self._build_indices()

        if data_config.get("shuffle", True):
            seed = data_config.get("seed", None)
            rng = random.Random(seed)
            rng.shuffle(self.indices)

    def _category_allowed(self, row: dict) -> bool:
        if not self.include_categories:
            return True
        cat = (
            _norm(row.get("question_kind"))
            or _norm(row.get("Category"))
            or _norm(row.get("category"))
            or _norm(row.get("extra_info", {}).get("Category"))
        )
        if cat is None:
            return False
        return cat.lower() in self.include_categories

    def _build_indices(self) -> list[int]:
        indices: list[int] = []
        for idx in range(len(self.data_source)):
            row = self.data_source.dataframe[idx]
            if not self._category_allowed(row):
                continue

            group_id = _build_group_id_from_row(row)
            if group_id is None:
                if self.keep_if_group_id_missing:
                    indices.append(idx)
                continue

            if group_id in self.drop_ids:
                continue
            indices.append(idx)
        return indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
