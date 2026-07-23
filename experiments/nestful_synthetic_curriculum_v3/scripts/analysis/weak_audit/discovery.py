"""Discovery report for weak-model audit inputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from weak_audit.io_utils import sha256_file
from weak_audit.paths import AuditPaths


def _exists_meta(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sample_eval_schema(eval_dir: Path) -> Optional[dict]:
    p = eval_dir / "final_eval_trajectories.jsonl"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as fh:
        line = fh.readline()
    if not line.strip():
        return None
    row = json.loads(line)
    traj = row.get("_traj") or {}
    turn0 = (traj.get("turns") or [{}])[0] if traj.get("turns") else {}
    return {
        "row_keys": sorted(row.keys()),
        "traj_keys": sorted(traj.keys()),
        "turn_keys": sorted(turn0.keys()) if turn0 else [],
        "official_win_row": row.get("official_win"),
        "official_win_traj": traj.get("official_win"),
        "pred_answer": traj.get("pred_answer"),
        "reward_train_strict": traj.get("reward_train_strict"),
    }


def _count_jsonl(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _adapter_hash(ckpt_dir: Path) -> Optional[str]:
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        p = ckpt_dir / name
        if p.is_file():
            return sha256_file(p)[:16]
    return None


def build_discovery(paths: AuditPaths) -> dict:
    manifest = {}
    if paths.run_manifest.is_file():
        manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    analysis = {}
    if paths.analysis_json.is_file():
        analysis = json.loads(paths.analysis_json.read_text(encoding="utf-8"))

    inputs = {
        "nestful_test": _exists_meta(paths.nestful_test),
        "eval_C0": _exists_meta(paths.eval_c0 / "final_eval_trajectories.jsonl"),
        "eval_E1": _exists_meta(paths.eval_e1 / "final_eval_trajectories.jsonl"),
        "eval_E2": _exists_meta(paths.eval_e2 / "final_eval_trajectories.jsonl"),
        "analysis_c0_e1_e2": _exists_meta(paths.analysis_json),
        "discordant_audit": _exists_meta(paths.discordant_jsonl),
        "task_level_analysis": _exists_meta(paths.task_level_jsonl),
        "run_manifest": _exists_meta(paths.run_manifest),
    }

    provenance = {
        "run_id": paths.run_dir.name,
        "run_dir": str(paths.run_dir),
        "model_id": (manifest.get("model") or {}).get("id"),
        "adapter_hash_E1": analysis.get("adapter_hash_E1") or _adapter_hash(paths.ckpt_e1),
        "adapter_hash_E2": analysis.get("adapter_hash_E2") or _adapter_hash(paths.ckpt_e2),
        "parity_ok": analysis.get("parity_ok"),
        "paired_E2_vs_C0": (analysis.get("paired") or {}).get("E2_vs_C0"),
        "summary_win_rates": {
            arm: (analysis.get("summary") or {}).get(arm, {}).get("win_rate")
            for arm in ("C0", "E1", "E2")
        },
    }

    field_map = {
        "question": "nestful_test.jsonl: question | input | prompt",
        "offered_tools": "nestful_test.jsonl: tools (JSON list)",
        "gold_calls": "nestful_test.jsonl: gold_calls | output | gold_output",
        "expected_outcome": "nestful_test.jsonl: gold_answer | answer | final_answer",
        "predicted_calls": "eval _traj.turns[].parsed_call",
        "observations": "eval _traj.turns[].observation (when fail_reason is null)",
        "final_answer": "eval _traj.pred_answer",
        "official_win": "eval row via _traj.official_win (also scorer on test)",
        "failure_taxonomy": "derived: scripts.analysis.two_phase_root_cause_analysis.classify_failure",
        "reward_R0": "computed offline: lib.reward_v3_2_dense on saved trajectory",
        "reward_train_strict": "eval _traj.reward_train_strict (strict gold trace, NOT training R0)",
        "first_divergence": "computed: compare C0 vs E2 predicted_calls",
        "tool_call_count": "eval _traj.num_tool_calls",
    }

    missing: List[str] = []
    for key, meta in inputs.items():
        if not meta.get("exists"):
            missing.append(key)

    limitations = [
        "Training reward R0 is recomputed offline; eval reward_train_strict is a different policy.",
        "Pass-B anonymization hides checkpoint identity from the annotator model.",
        "Token estimates use chars/4 heuristic unless tiktoken is installed.",
        "Weak-model annotations are not ground truth; agreement measures annotator stability only.",
    ]
    if missing:
        limitations.insert(0, f"Missing inputs: {', '.join(missing)}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "provenance": provenance,
        "task_counts": {
            "nestful_test": _count_jsonl(paths.nestful_test),
            "eval_C0": _count_jsonl(paths.eval_c0 / "final_eval_trajectories.jsonl"),
            "eval_E1": _count_jsonl(paths.eval_e1 / "final_eval_trajectories.jsonl"),
            "eval_E2": _count_jsonl(paths.eval_e2 / "final_eval_trajectories.jsonl"),
        },
        "field_map": field_map,
        "eval_schema_sample_C0": _sample_eval_schema(paths.eval_c0),
        "missing_inputs": missing,
        "limitations": limitations,
    }


def render_discovery_md(data: dict) -> str:
    lines = [
        "# Weak-model audit — discovery",
        "",
        f"**Generated:** {data['generated_at']}",
        f"**Run ID:** {data['provenance']['run_id']}",
        "",
        "## Provenance (verified against artifacts)",
        "",
        f"- C0 win rate: {data['provenance']['summary_win_rates'].get('C0')}",
        f"- E1 win rate: {data['provenance']['summary_win_rates'].get('E1')}",
        f"- E2 win rate: {data['provenance']['summary_win_rates'].get('E2')}",
        f"- E2 vs C0 paired: {data['provenance'].get('paired_E2_vs_C0')}",
        f"- parity_ok: {data['provenance'].get('parity_ok')}",
        f"- adapter E1: `{data['provenance'].get('adapter_hash_E1')}`",
        f"- adapter E2: `{data['provenance'].get('adapter_hash_E2')}`",
        "",
        "## Input files (SHA-256)",
        "",
        "| Artifact | exists | sha256 | n |",
        "|----------|--------|--------|--:|",
    ]
    for name, meta in data["inputs"].items():
        n = data["task_counts"].get(name.replace("eval_", "").replace("_", "").upper(), "")
        cnt = data["task_counts"].get(
            "nestful_test" if name == "nestful_test" else
            name.replace("eval_", "eval_").replace("C0", "C0").replace("E1", "E1").replace("E2", "E2"),
            "",
        )
        # simpler table
        cnt_val = ""
        if name == "nestful_test":
            cnt_val = data["task_counts"].get("nestful_test", "")
        elif name == "eval_C0":
            cnt_val = data["task_counts"].get("eval_C0", "")
        elif name == "eval_E1":
            cnt_val = data["task_counts"].get("eval_E1", "")
        elif name == "eval_E2":
            cnt_val = data["task_counts"].get("eval_E2", "")
        lines.append(
            f"| {name} | {meta.get('exists')} | `{meta.get('sha256', '')[:16]}…` | {cnt_val} |"
        )
    lines += [
        "",
        "## Field mapping",
        "",
    ]
    for k, v in data["field_map"].items():
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## Limitations", ""]
    for lim in data["limitations"]:
        lines.append(f"- {lim}")
    if data.get("eval_schema_sample_C0"):
        lines += ["", "## Eval trajectory schema (sample C0)", "", "```json"]
        lines.append(json.dumps(data["eval_schema_sample_C0"], indent=2)[:2000])
        lines.append("```")
    return "\n".join(lines)
