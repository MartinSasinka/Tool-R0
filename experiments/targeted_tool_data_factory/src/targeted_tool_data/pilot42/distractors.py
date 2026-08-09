"""Behaviourally non-equivalent distractor construction."""
from __future__ import annotations

import random
from typing import Any, Dict, List

from .. import registry as reg
from ..capability import behaviourally_equivalent, family_of, signatures_compatible
from .generate import _tool


def build_distractors(gold_primitive_ids: List[str], count: int = 4,
                      hardness: str = "hard", seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    for sid, primitive in reg.all_primitives().items():
        if sid in gold_primitive_ids:
            continue
        for gold_sid in gold_primitive_ids:
            gold = reg.get(gold_sid)
            if not signatures_compatible(gold, primitive):
                continue
            if behaviourally_equivalent(gold, primitive):
                continue
            same_family = family_of(sid) == family_of(gold_sid)
            if hardness == "hard" and not same_family:
                continue
            candidates.append(sid)
            break
    rng.shuffle(candidates)
    rows = []
    for sid in candidates[:count]:
        row = _tool(sid)
        row["is_distractor"] = True
        row["distractor_type"] = (
            "same_capability_non_equivalent" if hardness == "hard"
            else "schema_compatible_non_equivalent")
        row["hardness"] = hardness
        rows.append(row)
    return rows


def verify_non_equivalence(gold_sid: str, distractor_sid: str) -> bool:
    gold, distractor = reg.get(gold_sid), reg.get(distractor_sid)
    return signatures_compatible(gold, distractor) and not behaviourally_equivalent(
        gold, distractor)


def attach_distractors(record: Dict[str, Any], *, count: int = 4) -> Dict[str, Any]:
    gold_ids = [n["primitive_id"] for n in
                (record.get("semantic_program") or {}).get("nodes", [])]
    seed = abs(hash(record.get("task_id") or "")) % (10 ** 9)
    distractors = build_distractors(gold_ids, count=count, hardness="hard", seed=seed)
    if len(distractors) < count:
        distractors += build_distractors(
            gold_ids, count=count - len(distractors), hardness="medium",
            seed=seed + 1)
    tools = list(record.get("tools") or [])
    names = {t.get("name") for t in tools}
    for d in distractors:
        if d["name"] not in names:
            tools.append(d)
            names.add(d["name"])
    out = dict(record)
    out["tools"] = tools
    out["distractor_tool_ids"] = [d["semantic_id"] for d in distractors]
    out["distractor_hardness"] = [d.get("hardness") for d in distractors]
    return out
