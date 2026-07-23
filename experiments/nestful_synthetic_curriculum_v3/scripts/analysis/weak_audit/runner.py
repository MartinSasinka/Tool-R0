"""Batch LLM runner for weak-model audit annotations."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from weak_audit.compression import estimate_tokens
from weak_audit.constants import SYSTEM_PROMPT
from weak_audit.io_utils import append_jsonl, read_jsonl, sha256_text
from weak_audit.schema import ANNOTATION_JSON_SCHEMA


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_APPEND_LOCK = threading.Lock()


def _load_done(raw_path) -> Set[str]:
    done: Set[str] = set()
    if not raw_path.exists():
        return done
    for row in read_jsonl(raw_path):
        if row.get("task_id") and not row.get("error"):
            done.add(f"{row['task_id']}:{row.get('pass', '?')}")
    return done


def _make_client(base_url: Optional[str], api_key_env: str):
    import sys
    from pathlib import Path
    v3 = Path(__file__).resolve().parents[3]
    scripts = v3 / "scripts" / "data"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from openrouter_client import OpenRouterClient  # noqa: WPS433
    return OpenRouterClient(
        max_retries=int(os.environ.get("WEAK_AUDIT_MAX_RETRIES", "3")),
    )


def _call_model(
    client,
    model: str,
    system: str,
    user_payload: dict,
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    use_json_schema: bool,
    provider_slug: Optional[str] = None,
    mock_handler: Optional[Callable] = None,
) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    if mock_handler is not None:
        text = mock_handler(messages)
        return {
            "text": text,
            "cached": False,
            "cost_usd": 0.0,
            "requested_model": model,
            "response_model": model,
            "provider": "mock",
            "fallback_used": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "reported_cost": 0.0,
        }
    return client.chat(
        role="weak_solver",
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        json_schema=ANNOTATION_JSON_SCHEMA if use_json_schema else None,
        reasoning_effort=reasoning_effort or "none",
        provider_slug=provider_slug,
    )


def run_annotations(
    inputs: List[dict],
    *,
    output_raw: str,
    model: str,
    pass_label: str,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENROUTER_API_KEY",
    concurrency: int = 4,
    temperature: float = 0.0,
    max_output_tokens: int = 250,
    max_retries: int = 3,
    resume: bool = True,
    limit: Optional[int] = None,
    mock_handler: Optional[Callable] = None,
    reasoning_effort: Optional[str] = "none",
    use_json_schema: bool = True,
    provider_slug: Optional[str] = None,
) -> dict:
    out_path = __import__("pathlib").Path(output_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if resume else set()
    if limit is not None:
        inputs = inputs[:limit]

    if mock_handler is None and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"{api_key_env} not set; use --mock for offline tests or set API key"
        )

    client = None if mock_handler else _make_client(base_url, api_key_env)
    stats = {"ok": 0, "error": 0, "skipped": 0}

    def _one(inp: dict) -> None:
        tid = inp["task_id"]
        key = f"{tid}:{pass_label}"
        if key in done:
            stats["skipped"] += 1
            return
        t0 = time.time()
        req_hash = sha256_text(json.dumps(inp, sort_keys=True, ensure_ascii=False))
        row_base = {
            "task_id": tid,
            "pass": pass_label,
            "model_id": model,
            "request_hash": req_hash,
            "prompt_token_estimate": estimate_tokens(inp),
            "timestamp": _now(),
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = _call_model(
                    client,
                    model,
                    inp.get("system_prompt") or SYSTEM_PROMPT,
                    inp["case"],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    use_json_schema=use_json_schema,
                    provider_slug=provider_slug,
                    mock_handler=mock_handler,
                )
                text = resp.get("text") or ""
                row = {
                    **row_base,
                    "response": text,
                    "latency_s": round(time.time() - t0, 3),
                    "retry_count": attempt,
                    "cost_usd": resp.get("cost_usd"),
                    "cached": resp.get("cached", False),
                    "requested_model": resp.get("requested_model", model),
                    "response_model": resp.get("response_model"),
                    "provider": resp.get("provider"),
                    "fallback_used": resp.get("fallback_used"),
                    "prompt_tokens": resp.get("prompt_tokens"),
                    "completion_tokens": resp.get("completion_tokens"),
                    "reasoning_tokens": resp.get("reasoning_tokens"),
                    "total_tokens": resp.get("total_tokens"),
                    "reported_cost": resp.get("reported_cost"),
                }
                with _APPEND_LOCK:
                    append_jsonl(out_path, row)
                stats["ok"] += 1
                return
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(60.0, 2 ** attempt))
        with _APPEND_LOCK:
            append_jsonl(out_path, {**row_base, "error": last_err, "retry_count": max_retries})
        stats["error"] += 1

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        futs = [pool.submit(_one, inp) for inp in inputs]
        for _ in as_completed(futs):
            pass
    return stats
