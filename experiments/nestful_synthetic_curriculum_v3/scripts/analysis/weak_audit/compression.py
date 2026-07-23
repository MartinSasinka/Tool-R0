"""Token estimation and packet compression."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from weak_audit.constants import TOKEN_HARD, TOKEN_TARGET


def estimate_tokens(obj: Any) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
        return len(enc.encode(text))
    except Exception:
        text = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
        return max(1, len(text) // 4)


def truncate_text_value(val: str, max_len: int = 400) -> Any:
    if len(val) <= max_len:
        return val
    head = val[: max_len // 2]
    tail = val[-max_len // 4 :]
    return {
        "truncated": True,
        "original_length": len(val),
        "type": "string",
        "preview_start": head,
        "preview_end": tail,
    }


def preserve_observation(obs: Any, *, max_text: int = 800) -> Any:
    if obs is None or isinstance(obs, (bool, int, float)):
        return obs
    if isinstance(obs, str):
        return truncate_text_value(obs, max_text) if len(obs) > max_text else obs
    if isinstance(obs, list):
        if len(json.dumps(obs, ensure_ascii=False)) > max_text * 2:
            return {
                "truncated": True,
                "type": "list",
                "original_length": len(obs),
                "value": obs[:8] if len(obs) > 8 else obs,
            }
        return obs
    if isinstance(obs, dict):
        if len(json.dumps(obs, ensure_ascii=False)) > max_text * 2:
            return {
                "truncated": True,
                "type": "object",
                "original_length": len(obs),
                "keys": list(obs.keys())[:12],
                "preview": {k: obs[k] for k in list(obs.keys())[:4]},
            }
        return obs
    return obs


def shorten_tool_description(desc: str, max_len: int = 180) -> str:
    if len(desc) <= max_len:
        return desc
    return desc[: max_len - 3] + "..."


def compress_packet(
    packet: dict,
    *,
    target: int = TOKEN_TARGET,
    hard: int = TOKEN_HARD,
) -> Tuple[dict, dict]:
    """Return (compressed_packet, log_entry)."""
    original = copy.deepcopy(packet)
    before = estimate_tokens(original)
    removed: List[str] = []
    pkt = copy.deepcopy(packet)

    def _apply_tool_shortening() -> None:
        for tl in pkt.get("relevant_tools") or []:
            d = tl.get("description") or ""
            if len(d) > 180:
                tl["description"] = shorten_tool_description(d)
                removed.append("tool_description_shortened")

    def _drop_unused_tools() -> None:
        used = set()
        for arm in ("C0", "E1", "E2"):
            for c in (pkt.get(arm) or {}).get("calls") or []:
                if c.get("name"):
                    used.add(c["name"])
        for gc in (pkt.get("gold_metadata") or {}).get("gold_calls") or []:
            if gc.get("name"):
                used.add(gc["name"])
        before_n = len(pkt.get("relevant_tools") or [])
        pkt["relevant_tools"] = [
            t for t in (pkt.get("relevant_tools") or []) if t.get("name") in used
        ]
        if len(pkt["relevant_tools"]) < before_n:
            removed.append("unused_tools_dropped")

    def _truncate_question() -> None:
        q = pkt.get("question") or ""
        if len(q) > 900:
            pkt["question"] = q[:850] + "…"
            removed.append("question_truncated")

    steps = [_apply_tool_shortening, _drop_unused_tools, _truncate_question]
    after = before
    for step in steps:
        if after <= target:
            break
        step()
        after = estimate_tokens(pkt)

    over_hard = after > hard
    log = {
        "task_id": packet.get("task_id"),
        "tokens_before": before,
        "tokens_after": after,
        "target": target,
        "hard_limit": hard,
        "removed_parts": removed,
        "over_hard_limit": over_hard,
    }
    return pkt, log
