"""Shared deterministic stratified-sampling helpers for the reward ablation
data-prep scripts (prepare_train_subset_160.py, prepare_nestful_diagnostic_500.py).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def largest_remainder_allocation(cell_sizes: Dict[Any, int], total: int) -> Dict[Any, int]:
    """Proportional allocation across cells summing exactly to `total`,
    never exceeding a cell's available size. Deterministic tie-breaking by
    sorted string key so results don't depend on dict iteration order."""
    n_source = sum(cell_sizes.values())
    if n_source == 0:
        return {k: 0 for k in cell_sizes}
    raw = {k: (total * v / n_source) for k, v in cell_sizes.items()}
    floor_alloc = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floor_alloc.values())
    order = sorted(raw.keys(), key=lambda k: (-(raw[k] - floor_alloc[k]), str(k)))
    for k in order[:max(remainder, 0)]:
        floor_alloc[k] += 1
    for k in floor_alloc:
        floor_alloc[k] = min(floor_alloc[k], cell_sizes[k])
    deficit = total - sum(floor_alloc.values())
    guard = 0
    while deficit > 0:
        spare = sorted(
            [k for k in cell_sizes if floor_alloc[k] < cell_sizes[k]],
            key=lambda k: (-(cell_sizes[k] - floor_alloc[k]), str(k)),
        )
        if not spare:
            break
        for k in spare:
            if deficit <= 0:
                break
            floor_alloc[k] += 1
            deficit -= 1
        guard += 1
        if guard > 10000:
            break
    return floor_alloc


def stratified_select(
    ids_by_cell: Dict[Any, List[str]],
    alloc: Dict[Any, int],
    seed: int,
) -> List[str]:
    """Deterministic within-cell random selection (sorted ids, seeded
    shuffle) — reproducible independent of file/dict ordering."""
    rng = random.Random(seed)
    selected: List[str] = []
    for cell_key in sorted(ids_by_cell.keys(), key=str):
        ids = sorted(ids_by_cell[cell_key])
        rng.shuffle(ids)
        selected.extend(ids[: alloc.get(cell_key, 0)])
    return sorted(set(selected))


def group_by(items: Sequence[Tuple[str, Any]]) -> Dict[Any, List[str]]:
    out: Dict[Any, List[str]] = defaultdict(list)
    for sid, key in items:
        out[key].append(sid)
    return dict(out)
