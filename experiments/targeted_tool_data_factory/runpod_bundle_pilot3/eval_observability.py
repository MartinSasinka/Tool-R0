"""Phase-Q adapter: turn a finished sharded eval into audit artifacts.

Pilot3's `eval_manifest.json` was a copy of the score summary, so a C0/D1 pair
could not be checked for backend, prompt or scorer identity after the fact —
which is exactly how a backend confound survived into a headline number. This
module writes the identity-bearing manifest plus per-task rows from artifacts
the eval already produced. It runs after generation and never calls a model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_FACTORY_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_FACTORY_SRC) not in sys.path and _FACTORY_SRC.is_dir():
    sys.path.insert(0, str(_FACTORY_SRC))

try:
    from targeted_tool_data.observability import (EvalRunLogger, sha256_file,
                                                  sha256_obj, sha256_text)
    OBSERVABILITY_AVAILABLE = True
except Exception:  # noqa: BLE001
    OBSERVABILITY_AVAILABLE = False


def _git(args: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _adapter_hash(adapter_dir: Optional[Path]) -> Optional[str]:
    if not adapter_dir or not Path(adapter_dir).is_dir():
        return None
    parts = []
    for p in sorted(Path(adapter_dir).rglob("*")):
        if p.is_file() and p.suffix in (".safetensors", ".bin", ".json"):
            parts.append(f"{p.name}:{sha256_file(p)}")
    return sha256_text("\n".join(parts)) if parts else None


def _tools_of(row: Dict[str, Any]) -> Any:
    return row.get("tools") or row.get("offered_tools") or []


def _prompt_text(row: Dict[str, Any]) -> str:
    return str(row.get("input") or row.get("question") or "")


def write_eval_artifacts(*, out_dir: Path, run_id: str, arm: str,
                         input_rows: Sequence[Dict[str, Any]],
                         trajectories: Sequence[Dict[str, Any]],
                         dataset_path: Optional[Path],
                         checkpoint: Optional[Path],
                         backend: str, generation: Optional[Dict[str, Any]] = None,
                         merged_lora: bool = False,
                         model_revision: str = "",
                         shard_manifest: Optional[Dict[str, Any]] = None,
                         scorer_config: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    if not OBSERVABILITY_AVAILABLE:
        return {"written": False,
                "reason": "targeted_tool_data.observability not importable"}

    repo_root = Path(__file__).resolve().parents[3]
    logger = EvalRunLogger(out_dir=Path(out_dir), run_id=run_id,
                           repo_root=repo_root)

    sample_ids = [str(r.get("sample_id")) for r in input_rows]
    tool_hashes = [sha256_obj(_tools_of(r)) for r in input_rows]
    by_id = {str(r.get("sample_id")): r for r in trajectories}

    logger.write_manifest(
        model_revision=model_revision,
        adapter_path=str(checkpoint) if checkpoint else None,
        adapter_hash=_adapter_hash(checkpoint),
        merged_lora=merged_lora,
        backend=backend,
        dataset_path=dataset_path,
        sample_ids=sample_ids,
        shard_manifest=shard_manifest or {},
        generation=generation or {},
        chat_template_sha="",
        tool_schema_hash=sha256_text("\n".join(tool_hashes)),
        parser_commit=_git(["rev-parse", "HEAD"], repo_root),
        scorer_commit=_git(["rev-parse", "HEAD"], repo_root),
        scorer_config=scorer_config or {},
        extra={"arm": arm})

    for r in input_rows:
        sid = str(r.get("sample_id"))
        prompt = _prompt_text(r)
        logger.log_input({
            "sample_id": sid,
            "shard_id": (by_id.get(sid) or {}).get("shard_id"),
            "raw_prompt_hash": sha256_text(prompt),
            "offered_tools_hash": sha256_obj(_tools_of(r)),
            "input_token_ids_hash": (by_id.get(sid) or {}).get(
                "input_token_ids_hash"),
            "n_gold_calls": len(r.get("output") or []),
            "prompt_chars": len(prompt),
        })

    for t in trajectories:
        sid = str(t.get("sample_id"))
        text = str(t.get("generated_text") or t.get("raw_output") or "")
        parsed = t.get("parsed_calls") or t.get("predicted_calls") or []
        logger.log_trajectory({
            "sample_id": sid, "shard_id": t.get("shard_id"),
            "generated_text": text, "generated_text_hash": sha256_text(text),
            "parsed_calls": parsed, "observations": t.get("observations"),
            "stop_reason": t.get("stop_reason"),
            "parse_status": t.get("parse_status")
                            or ("ok" if parsed else "no_calls_parsed"),
            "execution_status": t.get("execution_status"),
            "runtime_sec": t.get("runtime_sec"),
        })
        src = by_id.get(sid) or {}
        logger.log_task_score({
            "sample_id": sid, "shard_id": t.get("shard_id"),
            "raw_prompt_hash": sha256_text(_prompt_text(src)),
            "input_token_ids_hash": t.get("input_token_ids_hash"),
            "offered_tools_hash": sha256_obj(_tools_of(src)),
            "generated_text_hash": sha256_text(text),
            "parsed_calls": parsed,
            "stop_reason": t.get("stop_reason"),
            "parse_status": t.get("parse_status")
                            or ("ok" if parsed else "no_calls_parsed"),
            "execution_status": t.get("execution_status"),
            "official_win": t.get("official_win", t.get("win")),
            "n_predicted_calls": len(parsed),
            "n_gold_calls": len(src.get("output") or []),
            "answer_match": t.get("answer_match"),
            "solution_equivalent": t.get("solution_equivalent"),
            "runtime_sec": t.get("runtime_sec"),
            "diagnostics": {k: v for k, v in t.items()
                            if k.startswith(("diag_", "metric_"))},
        })

    scores_path = logger.close()
    return {"written": True, "out_dir": str(out_dir),
            "task_scores": str(scores_path),
            "n_inputs": len(input_rows), "n_trajectories": len(trajectories)}
