"""Validate, repair, and compare Pass A/B annotations."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from weak_audit.constants import REPAIR_PROMPT, SYSTEM_PROMPT
from weak_audit.schema import validate_annotation


def parse_json_text(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group())
    if not isinstance(obj, dict):
        raise ValueError("annotation must be object")
    return obj


def validate_raw_rows(
    raw_rows: List[dict],
    *,
    repair_fn=None,
) -> Tuple[List[dict], List[dict]]:
    valid: List[dict] = []
    invalid: List[dict] = []
    for raw in raw_rows:
        if raw.get("error"):
            invalid.append({**raw, "validation_status": "invalid_after_repair"})
            continue
        text = raw.get("response") or ""
        status = "valid"
        obj = None
        try:
            obj = parse_json_text(text)
        except Exception:
            if repair_fn is not None:
                try:
                    repaired = repair_fn(raw, text)
                    obj = parse_json_text(repaired)
                    status = "repaired"
                except Exception:
                    status = "invalid_after_repair"
            else:
                status = "invalid_after_repair"
        if obj is None:
            invalid.append({**raw, "validation_status": status})
            continue
        errs = validate_annotation(obj, expected_task_id=raw.get("task_id"))
        if errs:
            invalid.append({**raw, "validation_status": "invalid_after_repair", "errors": errs})
            continue
        valid.append({
            **obj,
            "validation_status": status,
            "model_id": raw.get("model_id"),
            "pass": raw.get("pass"),
            "request_hash": raw.get("request_hash"),
        })
    return valid, invalid


def remap_pass_b(obj: dict, mapping: Dict[str, str]) -> dict:
    """Pass B uses anonymous trajectories; annotation fields stay semantic."""
    return obj


def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a.count(c) / n for c in cats) ** 2
    pb = sum(b.count(c) / n for c in cats) ** 2
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def compare_passes(
    pass_a: Dict[str, dict],
    pass_b: Dict[str, dict],
) -> dict:
    fields = [
        "first_divergence_turn", "root_cause", "shorter_path_verdict",
        "observation_used_correctly", "reward_ordering_correct",
        "responsible_reward_component", "recommended_fix",
    ]
    ids = sorted(set(pass_a) & set(pass_b))
    per_field = {}
    exact = 0
    root_changed = 0
    reward_changed = 0
    conf_diffs: List[float] = []
    for f in fields:
        matches = sum(1 for tid in ids if pass_a[tid].get(f) == pass_b[tid].get(f))
        per_field[f] = {
            "agreement_rate": matches / len(ids) if ids else None,
            "n": len(ids),
        }
    for tid in ids:
        a, b = pass_a[tid], pass_b[tid]
        if all(a.get(f) == b.get(f) for f in fields):
            exact += 1
        if a.get("root_cause") != b.get("root_cause"):
            root_changed += 1
        if a.get("reward_ordering_correct") != b.get("reward_ordering_correct"):
            reward_changed += 1
        conf_diffs.append(abs(float(a.get("confidence") or 0) - float(b.get("confidence") or 0)))
    kappa = cohen_kappa(
        [pass_a[t].get("root_cause", "unclear") for t in ids],
        [pass_b[t].get("root_cause", "unclear") for t in ids],
    )
    return {
        "n_tasks": len(ids),
        "exact_agreement_rate": exact / len(ids) if ids else None,
        "per_field_agreement": per_field,
        "root_cause_kappa": kappa,
        "root_cause_changed_rate": root_changed / len(ids) if ids else None,
        "reward_ordering_changed_rate": reward_changed / len(ids) if ids else None,
        "mean_confidence_abs_diff": sum(conf_diffs) / len(conf_diffs) if conf_diffs else None,
    }
