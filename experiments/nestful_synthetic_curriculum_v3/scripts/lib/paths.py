"""Single source of truth for dataset locations and legacy-dataset detection.

Canonical vs legacy status per audits/DATASET_AUDIT.md and MASTER_AUDIT_REPORT.md:
dataset A (curriculum_v3_1/filtered) is canonical; dataset B
(nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic) is legacy and must never be
used silently. When the P3 archive move happens, only this module changes.
"""
from __future__ import annotations

import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../experiments/nestful_synthetic_curriculum_v3/scripts/lib -> repo root
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
V3_ROOT = os.path.join(REPO_ROOT, "experiments", "nestful_synthetic_curriculum_v3")
MINIMAL_ROOT = os.path.join(REPO_ROOT, "experiments", "nestful_mtgrpo_minimal")

# --- canonical datasets (A) -------------------------------------------------
CANONICAL_CURRICULUM_DIR = os.path.join(V3_ROOT, "outputs", "curriculum_v3_1", "filtered")
CANONICAL_STAGE_FILES = {
    1: os.path.join(CANONICAL_CURRICULUM_DIR, "stage1_1call_atomic.jsonl"),
    2: os.path.join(CANONICAL_CURRICULUM_DIR, "stage2_2call_dependency.jsonl"),
    3: os.path.join(CANONICAL_CURRICULUM_DIR, "stage3_3call_composition.jsonl"),
    4: os.path.join(CANONICAL_CURRICULUM_DIR, "stage4_4to6call_persistence.jsonl"),
}
CURRICULUM_MANIFEST = os.path.join(
    V3_ROOT, "outputs", "curriculum_v3_1", "curriculum_v3_1_manifest.json"
)

# --- NESTFUL evaluation data -------------------------------------------------
NESTFUL_DEV = os.path.join(MINIMAL_ROOT, "data", "splits", "nestful_dev.jsonl")       # 200, selection only
NESTFUL_TEST = os.path.join(MINIMAL_ROOT, "data", "splits", "nestful_test.jsonl")     # 1661, headline
NESTFUL_FULL = os.path.join(MINIMAL_ROOT, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")  # 1861 = dev+test
NESTFUL_DATASETS = {"nestful_dev": NESTFUL_DEV, "nestful_test": NESTFUL_TEST, "nestful_full": NESTFUL_FULL}

# --- legacy (B) — guarded ----------------------------------------------------
# Archived by cleanup Phase K (was nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic).
LEGACY_DATASET_B_DIR = os.path.join(
    V3_ROOT, "archive", "legacy_dataset_B_filtered_toolr0_synthetic")
_LEGACY_MARKERS = ("filtered_toolr0_synthetic",)

# Default output root for eval batches (TARGET_ARCHITECTURE.md §1/§4).
EVAL_OUTPUT_ROOT = os.path.join(V3_ROOT, "outputs", "evals")


def is_legacy_dataset_path(path: str) -> bool:
    """True if `path` points into the legacy dataset B tree."""
    norm = os.path.normpath(path).replace("\\", "/")
    return any(marker in norm for marker in _LEGACY_MARKERS)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def count_jsonl_rows(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def dataset_info(path: str) -> dict:
    """{name, path (repo-relative), sha256, n_rows} for a dataset file."""
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT).replace("\\", "/")
    return {
        "name": os.path.basename(path),
        "path": rel,
        "sha256": sha256_file(path),
        "n_rows": count_jsonl_rows(path),
    }


if __name__ == "__main__":
    print(f"repo_root = {REPO_ROOT}")
    for label, p in {**{f"stage{k}": v for k, v in CANONICAL_STAGE_FILES.items()},
                     **NESTFUL_DATASETS}.items():
        status = "OK" if os.path.isfile(p) else "MISSING"
        print(f"  {label:14s} [{status}] {os.path.relpath(p, REPO_ROOT)}")
    print(f"legacy B dir present: {os.path.isdir(LEGACY_DATASET_B_DIR)} "
          f"({os.path.relpath(LEGACY_DATASET_B_DIR, REPO_ROOT)})")
