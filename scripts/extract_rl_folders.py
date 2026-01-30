#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from typing import Iterable

import pyarrow.parquet as pq


def _iter_extra_info_paths(parquet_path: str) -> Iterable[str]:
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["extra_info"]):
        rows = batch.to_pylist()
        for row in rows:
            extra = row.get("extra_info") or {}
            if isinstance(extra, dict):
                path = extra.get("path")
                if path:
                    yield str(path)


def _infer_src_root(sample_path: str) -> str | None:
    marker = f"{os.sep}deepseek_ocr_outputs{os.sep}"
    idx = sample_path.find(marker)
    if idx == -1:
        return None
    return sample_path[: idx + len("deepseek_ocr_outputs") + 1]


def _collect_unique_dirs(parquet_path: str, src_root: str | None) -> tuple[str, list[str]]:
    src_root_resolved = src_root
    unique_dirs: set[str] = set()
    for path in _iter_extra_info_paths(parquet_path):
        if src_root_resolved is None:
            src_root_resolved = _infer_src_root(path)
            if src_root_resolved is None:
                raise ValueError(
                    f"Cannot infer src root from path: {path}. "
                    "Pass --src-root explicitly."
                )
        if not os.path.normpath(path).startswith(os.path.normpath(src_root_resolved)):
            raise ValueError(
                f"Path {path} is not under src root {src_root_resolved}. "
                "Pass --src-root to override."
            )
        unique_dirs.add(path)
    if src_root_resolved is None:
        raise ValueError("No paths found in parquet; cannot infer src root.")
    return src_root_resolved, sorted(unique_dirs)


def _copy_dirs(
    src_root: str,
    dst_root: str,
    dirs: list[str],
    dry_run: bool,
) -> None:
    os.makedirs(dst_root, exist_ok=True)
    for src_dir in dirs:
        rel = os.path.relpath(src_dir, src_root)
        dst_dir = os.path.join(dst_root, rel)
        if dry_run:
            print(f"[dry-run] {src_dir} -> {dst_dir}")
            continue
        if not os.path.exists(src_dir):
            print(f"[skip] missing: {src_dir}")
            continue
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract folders referenced by train_RL.parquet into a new root."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to train_RL.parquet.",
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Root directory that contains deepseek_ocr_outputs. "
        "If omitted, inferred from extra_info.path.",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Destination root directory (e.g., deep_seek_ocr_outputs_RL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    args = parser.parse_args()

    src_root, dirs = _collect_unique_dirs(args.parquet, args.src_root)
    print(f"src_root: {src_root}")
    print(f"unique folders: {len(dirs)}")
    _copy_dirs(src_root, args.dst_root, dirs, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
