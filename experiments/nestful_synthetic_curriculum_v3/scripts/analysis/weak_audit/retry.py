"""Retry only invalid annotation pairs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from weak_audit.agreement import validate_raw_rows
from weak_audit.io_utils import append_jsonl, read_json, read_jsonl, write_jsonl
from weak_audit.runner import run_annotations


def _load_inputs(out_dir: Path) -> Dict[Tuple[str, str], dict]:
    idx: Dict[Tuple[str, str], dict] = {}
    for label in ("A", "B"):
        for row in read_jsonl(out_dir / f"pass_{label.lower()}_inputs.jsonl"):
            idx[(row["task_id"], label)] = row
    return idx


def _load_done(raw_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not raw_path.is_file():
        return done
    for row in read_jsonl(raw_path):
        if row.get("task_id") and not row.get("error"):
            done.add(f"{row['task_id']}:{row.get('pass')}")
    return done


def build_retry_inputs(out_dir: Path) -> List[dict]:
    manifest = read_json(out_dir / "invalid_retry_manifest.json")
    inputs_idx = _load_inputs(out_dir)
    valid_pairs = set()
    for label in ("A", "B"):
        for row in read_jsonl(out_dir / f"pass_{label.lower()}_annotations.jsonl"):
            valid_pairs.add((row["task_id"], label))

    retry_inputs: List[dict] = []
    for pair in manifest.get("pairs") or []:
        tid = pair["task_id"]
        pl = pair["pass_label"]
        key = (tid, pl)
        if key in valid_pairs:
            raise RuntimeError(f"refusing retry for already-valid pair {key}")
        inp = inputs_idx.get(key)
        if not inp:
            raise RuntimeError(f"missing pass input for {key}")
        retry_inputs.append(inp)
    return retry_inputs


def run_invalid_retry(
    out_dir: Path,
    *,
    model: str,
    provider: Optional[str] = None,
    api_key_env: str = "OPENROUTER_API_KEY",
    concurrency: int = 1,
    temperature: float = 0.0,
    reasoning_effort: str = "none",
    max_output_tokens: int = 500,
    max_retries: int = 3,
    resume: bool = True,
    use_json_schema: bool = True,
    dry_run: bool = False,
    repair_fn=None,
) -> dict:
    if dry_run:
        inputs = build_retry_inputs(out_dir)
        return {
            "dry_run": True,
            "n_pairs": len(inputs),
            "pairs": sorted({(i["task_id"], i["pass"]) for i in inputs}),
            "provider": provider,
        }

    if not os.environ.get(api_key_env):
        raise RuntimeError(f"{api_key_env} not set")

    raw_path = out_dir / "retry_invalid_raw.jsonl"
    validated_path = out_dir / "retry_invalid_validated.jsonl"
    failed_path = out_dir / "retry_invalid_failed.jsonl"

    inputs = build_retry_inputs(out_dir)
    done = _load_done(raw_path) if resume else set()
    todo = [i for i in inputs if f"{i['task_id']}:{i['pass']}" not in done]

    stats = {"requested": len(inputs), "todo": len(todo), "ok": 0, "error": 0}

    # Process sequentially by pass batches using runner per pass
    by_pass: Dict[str, List[dict]] = {"A": [], "B": []}
    for inp in todo:
        by_pass[inp["pass"]].append(inp)

    for pl, batch in by_pass.items():
        if not batch:
            continue
        run_stats = run_annotations(
            batch,
            output_raw=str(raw_path),
            model=model,
            pass_label=pl,
            api_key_env=api_key_env,
            concurrency=concurrency,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            resume=True,
            reasoning_effort=reasoning_effort,
            use_json_schema=use_json_schema,
            provider_slug=provider,
        )
        stats["ok"] += run_stats.get("ok", 0)
        stats["error"] += run_stats.get("error", 0)

    raw_rows = read_jsonl(raw_path)
    manifest_pairs = {
        (p["task_id"], p["pass_label"])
        for p in read_json(out_dir / "invalid_retry_manifest.json").get("pairs") or []
    }
    raw_rows = [
        r for r in raw_rows
        if (r.get("task_id"), r.get("pass")) in manifest_pairs
    ]

    valid, invalid = validate_raw_rows(raw_rows, repair_fn=repair_fn)
    write_jsonl(validated_path, valid)
    write_jsonl(failed_path, invalid)
    stats["validated"] = len(valid)
    stats["failed"] = len(invalid)
    stats["cost_usd"] = sum(
        float(r.get("reported_cost") or r.get("cost_usd") or 0) for r in raw_rows
    )
    stats["prompt_tokens"] = sum(int(r.get("prompt_tokens") or 0) for r in raw_rows)
    stats["completion_tokens"] = sum(int(r.get("completion_tokens") or 0) for r in raw_rows)
    return stats
