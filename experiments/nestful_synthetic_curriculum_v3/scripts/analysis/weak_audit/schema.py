"""Annotation JSON schema validation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from weak_audit.constants import (
    RECOMMENDED_FIXES,
    REWARD_COMPONENTS,
    ROOT_CAUSES,
    SHORTER_PATH_VERDICTS,
)


def validate_annotation(obj: dict, *, expected_task_id: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    if expected_task_id and obj.get("task_id") != expected_task_id:
        errors.append(f"task_id mismatch: {obj.get('task_id')} != {expected_task_id}")
    if not obj.get("task_id"):
        errors.append("missing task_id")
    rc = obj.get("root_cause")
    if rc not in ROOT_CAUSES:
        errors.append(f"invalid root_cause: {rc}")
    sp = obj.get("shorter_path_verdict")
    if sp not in SHORTER_PATH_VERDICTS:
        errors.append(f"invalid shorter_path_verdict: {sp}")
    rf = obj.get("recommended_fix")
    if rf not in RECOMMENDED_FIXES:
        errors.append(f"invalid recommended_fix: {rf}")
    rr = obj.get("responsible_reward_component")
    if rr not in REWARD_COMPONENTS:
        errors.append(f"invalid responsible_reward_component: {rr}")
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errors.append("confidence must be float in [0,1]")
    ev = obj.get("evidence")
    if not isinstance(ev, str) or len(ev) > 500:
        errors.append("evidence must be short string")
    fdt = obj.get("first_divergence_turn")
    if fdt is not None and not isinstance(fdt, int):
        errors.append("first_divergence_turn must be int or null")
    for bool_field in ("observation_used_correctly", "reward_ordering_correct"):
        val = obj.get(bool_field)
        if val is not None and not isinstance(val, bool):
            errors.append(f"{bool_field} must be bool or null")
    return errors


def is_valid(obj: dict, **kw) -> bool:
    return not validate_annotation(obj, **kw)


ANNOTATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "weak_audit_annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string"},
            "first_divergence_turn": {"type": ["integer", "null"]},
            "root_cause": {"type": "string", "enum": sorted(ROOT_CAUSES)},
            "shorter_path_verdict": {
                "type": "string",
                "enum": sorted(SHORTER_PATH_VERDICTS),
            },
            "observation_used_correctly": {"type": ["boolean", "null"]},
            "reward_ordering_correct": {"type": ["boolean", "null"]},
            "responsible_reward_component": {
                "type": "string",
                "enum": sorted(REWARD_COMPONENTS),
            },
            "recommended_fix": {"type": "string", "enum": sorted(RECOMMENDED_FIXES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string", "maxLength": 500},
        },
        "required": [
            "task_id",
            "first_divergence_turn",
            "root_cause",
            "shorter_path_verdict",
            "observation_used_correctly",
            "reward_ordering_correct",
            "responsible_reward_component",
            "recommended_fix",
            "confidence",
            "evidence",
        ],
    },
}
