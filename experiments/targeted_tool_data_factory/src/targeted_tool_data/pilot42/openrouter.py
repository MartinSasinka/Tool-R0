"""Pilot4.2 writer/critic routing with replay and a hard USD budget."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

from ..pilot41.openrouter import (CRITIC_SCHEMA, WRITER_SCHEMA, BudgetExceeded,
                                   OpenRouterSession as _BaseSession,
                                   assert_pinned_model, load_openrouter_config as _load)


def load_openrouter_config(path: Path | None = None) -> Dict[str, Any]:
    cfg = _load(path)
    cfg.update({"temperature": 0.5, "max_total_cost_usd": 20.0,
                "allow_fallbacks": False})
    cfg.setdefault("second_critic_model", "google/gemini-2.5-flash-lite")
    assert_pinned_model(cfg["second_critic_model"])
    if float(cfg["max_total_cost_usd"]) > 20:
        raise ValueError("Pilot4.2 budget may not exceed $20")
    return cfg


def needs_second_critic(record: Dict[str, Any], *, disagreement: bool = False,
                        rewritten: bool = False, seed: int = 0) -> bool:
    if len(record.get("gold_calls") or []) >= 6:
        return True
    if record.get("domain") in ("text_processing", "file_path", "url_processing"):
        return True
    if disagreement or rewritten:
        return True
    rng = random.Random(f"p42-second-critic:{seed}:{record.get('task_id')}")
    return rng.random() < .15


class OpenRouterSession(_BaseSession):
    """Base transport with Pilot4.2 routing; API keys never enter payload logs."""

    def critique_twice_if_needed(self, record: Dict[str, Any], contract: Dict[str, Any],
                                 query: str, findings: Dict[str, Any], *,
                                 disagreement: bool = False,
                                 rewritten: bool = False) -> list[Dict[str, Any]]:
        first = self.critique(contract, query, findings)
        rows = [first]
        if needs_second_critic(record, disagreement=disagreement, rewritten=rewritten):
            rows.append(self.critique(contract, query, findings, use_audit_model=True))
        return rows


def redact_secret(value: Any) -> Any:
    """Recursively replace accidental key-like values before diagnostic logging."""
    if isinstance(value, dict):
        return {k: ("<REDACTED>" if k.lower() in ("authorization", "api_key", "key")
                    else redact_secret(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secret(v) for v in value]
    if isinstance(value, str) and ("sk-" in value or value == os.getenv("OPENROUTER_API_KEY")):
        return "<REDACTED>"
    return value
