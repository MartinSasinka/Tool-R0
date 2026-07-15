"""Single source of truth for dataset locations and legacy-dataset detection.

Canonical vs legacy status per audits/DATASET_AUDIT.md and MASTER_AUDIT_REPORT.md:
dataset A (curriculum_v3_1/filtered) is canonical; dataset B
(nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic) is legacy and must never be
used silently. When the P3 archive move happens, only this module changes.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

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


# --- agentic synthetic-tool datasets (executor.mode=gold_replay REQUIRED) ---
# Tool names in these datasets come from ``nestful_like_generator.TOOLS``
# (LLM-invented, e.g. "units_per_box", "percentage_of") and are NOT entries in
# the real NESTFUL IBM function registry. Since that registry IS present in
# this repo (used for genuine NESTFUL data), ``executor.mode=auto`` resolves
# to ``full`` and either (a) hard-fails every episode on the first call with
# ``unknown_function`` (most tool names), or worse (b) silently executes a
# DIFFERENT real IBM function on a name collision (e.g. "rectangle_area" is
# also a real registry entry) and scores against its output instead of the
# synthetic gold trace. Either way the reward is corrupted. Always force
# ``executor.mode=gold_replay`` for these datasets — see
# docs/AGENTIC_DATA_GENERATION.md.
_AGENTIC_MARKERS = ("agentic_openrouter", "agentic_hybrid", "agentic_workers",
                   "nestful_like_agentic", "curriculum_v4")

# v5 agentic datasets (lib/agentic_data + lib/synthetic_tools.py, ~163 tools):
# their tool names ARE understood by the trainer's real executor in
# executor.mode="synthetic" (unlike the legacy v4 generator's tools, which
# only ever worked under gold_replay). Checked BEFORE _AGENTIC_MARKERS by
# is_agentic_synthetic_dataset_path()/is_agentic_synthetic_dataset() below —
# a v5 path must never be misclassified as a legacy v4 gold_replay-only one.
_V5_SYNTHETIC_MARKERS = ("curriculum_v5_agentic", "agentic_v5", "v5_agentic")


def is_v5_agentic_synthetic_dataset_path(path: str) -> bool:
    """Path-based heuristic: True if `path` looks like a v5 agentic dataset
    (``lib/synthetic_tools.py`` registry — real ``executor.mode=synthetic``
    execution, NOT gold_replay)."""
    norm = os.path.normpath(path).replace("\\", "/")
    return any(marker in norm for marker in _V5_SYNTHETIC_MARKERS)


def is_agentic_synthetic_dataset_path(path: str) -> bool:
    """Path-based heuristic: True if `path` looks like a LEGACY v4 agentic
    dataset (``lib/nestful_like_generator.py`` — gold_replay only). Returns
    False for v5 agentic datasets even though both trees contain the
    substring "agentic" — see ``is_v5_agentic_synthetic_dataset_path``."""
    if is_v5_agentic_synthetic_dataset_path(path):
        return False
    norm = os.path.normpath(path).replace("\\", "/")
    return any(marker in norm for marker in _AGENTIC_MARKERS)


def peek_dataset_source(path: str) -> Optional[str]:
    """Best-effort read of the first row's ``source`` field, or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return None
                src = row.get("source")
                return str(src) if src is not None else None
    except OSError:
        return None
    return None


def is_agentic_synthetic_dataset(path: str) -> bool:
    """True when `path`'s tools are the LEGACY v4 agentic generator's
    synthetic tools (not real IBM NESTFUL functions, and NOT understood by
    the v5 real synthetic executor either) — i.e. the executor MUST run in
    gold_replay mode. False for v5 agentic datasets (source contains
    "v5"/"synthetic_tools" — see ``is_v5_agentic_synthetic_dataset``), which
    should use ``executor.mode=synthetic`` instead.

    Checks the path first (cheap), then falls back to peeking at the first
    row's ``source`` field (handles datasets copied/renamed off the standard
    ``curriculum_v4_*`` tree).
    """
    if is_v5_agentic_synthetic_dataset(path):
        return False
    if is_agentic_synthetic_dataset_path(path):
        return True
    source = peek_dataset_source(path)
    return (bool(source) and "agentic" in source.lower()
            and "v5" not in source.lower())


def is_v5_agentic_synthetic_dataset(path: str) -> bool:
    """True when `path`'s tools come from the v5 ``lib/synthetic_tools.py``
    registry (real ``executor.mode=synthetic`` execution required/expected —
    never gold_replay)."""
    if is_v5_agentic_synthetic_dataset_path(path):
        return True
    source = peek_dataset_source(path)
    return bool(source) and "v5" in source.lower() and "agentic" in source.lower()


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
