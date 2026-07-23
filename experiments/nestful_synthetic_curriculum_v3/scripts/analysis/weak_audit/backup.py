"""Backup real audit artifacts before retry."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from weak_audit.io_utils import sha256_file, write_json


BACKUP_FILES = [
    "pass_a_annotations_raw.jsonl",
    "pass_b_annotations_raw.jsonl",
    "pass_a_annotations.jsonl",
    "pass_b_annotations.jsonl",
    "invalid_annotations.jsonl",
    "annotation_agreement.csv",
    "ANNOTATION_AGREEMENT.md",
    "cluster_counts.csv",
    "cluster_examples.json",
    "WEAK_MODEL_SUMMARY.md",
    "HIGH_PRIORITY_CASES.jsonl",
    "HIGH_PRIORITY_CASES.md",
    "REAL_RUN_STATUS.md",
]


def backup_before_retry(out_dir: Path, backup_dir: Path) -> dict:
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    hashes = {}
    for name in BACKUP_FILES:
        src = out_dir / name
        if not src.is_file():
            continue
        dst = backup_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
        hashes[name] = sha256_file(dst)
    manifest = {
        "source_dir": str(out_dir),
        "backup_dir": str(backup_dir),
        "files": copied,
        "sha256": hashes,
    }
    write_json(backup_dir / "MANIFEST_SHA256.json", manifest)
    return manifest
