"""Validate case packets against source evals."""
from __future__ import annotations

from typing import Dict, List, Tuple

from weak_audit.io_utils import read_jsonl


def validate_packets(
    packets_path,
    selected_ids: List[str],
    arms: Dict[str, Dict[str, dict]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    packets = read_jsonl(packets_path)
    ids = [p["task_id"] for p in packets]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task_id in case_packets.jsonl")
    missing = set(selected_ids) - set(ids)
    if missing:
        errors.append(f"missing packets for {len(missing)} selected ids")
    extra = set(ids) - set(selected_ids)
    if extra:
        errors.append(f"unexpected packets not in selection: {len(extra)}")
    for pkt in packets:
        tid = pkt["task_id"]
        tools = pkt.get("relevant_tools") or []
        if len(tools) > 40:
            errors.append(f"{tid}: suspiciously many relevant_tools ({len(tools)})")
        for arm in ("C0", "E1", "E2"):
            src = arms[arm].get(tid)
            if not src:
                errors.append(f"{tid}: missing source eval for {arm}")
                continue
            from scripts.analysis.two_phase_root_cause_analysis import official_win  # noqa
            expected = official_win(src) == 1.0
            got = (pkt.get(arm) or {}).get("official_win")
            if got != expected:
                errors.append(
                    f"{tid} {arm}: official_win mismatch got={got} expected={expected}"
                )
    return len(errors) == 0, errors
