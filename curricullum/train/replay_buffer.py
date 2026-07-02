#!/usr/bin/env python3
"""Data replay buffer for curriculum training.

Samples a fraction of rows from all prior-stage JSONL files and writes them
to a temporary file that train_grpo_stage.py appends to its training dataset.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import List, Optional


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_replay(
    stage: int,
    dataset_size: int,
    replay_fraction: float,
    seed: int,
    data_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Sample replay rows from stages 1..(stage-1) and write to a JSONL file.

    Args:
        stage: Current stage number (1-indexed). Stages 1..(stage-1) are used.
        dataset_size: Size of the current stage's training dataset (after expansion).
        replay_fraction: Fraction of dataset_size to draw as replay.
        seed: Random seed for reproducibility.
        data_dir: Directory containing epoch_N_Ncall.jsonl files.
        output_path: If None, writes to a temp file (caller must delete).

    Returns:
        Path to replay JSONL, or None if stage == 1 (no prior stages).
    """
    if stage <= 1:
        return None

    n_replay = max(1, int(dataset_size * replay_fraction))
    rng = random.Random(seed)

    # Collect rows from all prior stages
    pool: List[str] = []
    for prior_stage in range(1, stage):
        jsonl = data_dir / f"epoch_{prior_stage}_{prior_stage}call.jsonl"
        if not jsonl.is_file():
            print(f"[replay] WARNING: prior stage file not found: {jsonl}", flush=True)
            continue
        rows = list(_iter_jsonl(jsonl))
        pool.extend(json.dumps(r, ensure_ascii=False) for r in rows)

    if not pool:
        return None

    sampled = rng.choices(pool, k=n_replay)

    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="replay_stage")
        os.close(fd)
        output_path = Path(tmp)

    output_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")
    print(
        f"[replay] stage={stage}  sampled {len(sampled)} rows from {stage - 1} prior stages"
        f"  (pool={len(pool)}, fraction={replay_fraction}) -> {output_path}",
        flush=True,
    )
    return output_path
