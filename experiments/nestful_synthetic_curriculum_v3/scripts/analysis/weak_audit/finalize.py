"""Final manifest and retry finalization report."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from weak_audit.constants import SEED
from weak_audit.io_utils import read_json, read_jsonl, sha256_file, write_json


def _git_commit(repo: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def write_final_manifest(out_dir: Path, paths, merge_stats: dict, retry_stats: dict) -> dict:
    tag_files = {
        "case_packets": out_dir / "case_packets.jsonl",
        "pass_a_inputs": out_dir / "pass_a_inputs.jsonl",
        "pass_b_inputs": out_dir / "pass_b_inputs.jsonl",
        "pass_a_raw": out_dir / "pass_a_annotations_raw.jsonl",
        "pass_b_raw": out_dir / "pass_b_annotations_raw.jsonl",
        "pass_a_final": out_dir / "pass_a_annotations_final.jsonl",
        "pass_b_final": out_dir / "pass_b_annotations_final.jsonl",
        "agreement_final": out_dir / "ANNOTATION_AGREEMENT_FINAL.md",
        "high_priority_final": out_dir / "HIGH_PRIORITY_CASES_FINAL.jsonl",
    }
    sha = {k: sha256_file(v) for k, v in tag_files.items() if v.is_file()}

    manifest = {
        "status": "weak_audit_finalized",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "analyzed_run_id": paths.run_dir.name,
        "model_id": retry_stats.get("model") or "deepseek/deepseek-v3.2",
        "provider_retry": retry_stats.get("provider"),
        "seed": SEED,
        "n_case_packets": len(read_jsonl(out_dir / "case_packets.jsonl")),
        "pass_a_final_valid": merge_stats.get("pass_a_final_valid"),
        "pass_b_final_valid": merge_stats.get("pass_b_final_valid"),
        "pass_a_final_invalid": merge_stats.get("pass_a_final_invalid"),
        "pass_b_final_invalid": merge_stats.get("pass_b_final_invalid"),
        "sha256": sha,
        "git_commit": _git_commit(paths.repo),
        "retry_stats": retry_stats,
        "merge_stats": merge_stats,
    }
    write_json(out_dir / "WEAK_AUDIT_FINAL_MANIFEST.json", manifest)
    return manifest


def write_retry_finalization_report(
    out_dir: Path,
    *,
    before_agree: dict,
    after_agree: dict,
    merge_stats: dict,
    retry_stats: dict,
    provider: Optional[str],
) -> None:
    inv = read_json(out_dir / "invalid_retry_manifest.json")
    still = read_jsonl(out_dir / "invalid_annotations_final.jsonl")
    hp_before = read_jsonl(out_dir / "HIGH_PRIORITY_CASES.jsonl") if (
        out_dir / "HIGH_PRIORITY_CASES.jsonl"
    ).is_file() else []
    hp_after = read_jsonl(out_dir / "HIGH_PRIORITY_CASES_FINAL.jsonl") if (
        out_dir / "HIGH_PRIORITY_CASES_FINAL.jsonl"
    ).is_file() else []
    hp_before_ids = {r["task_id"] for r in hp_before}
    hp_after_ids = {r["task_id"] for r in hp_after}
    lines = [
        "# Retry finalization report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Invalid counts",
        "",
        f"- Originally invalid pairs: {inv.get('n_invalid_pairs')}",
        f"- Pass A originally invalid: {inv.get('n_pass_a')}",
        f"- Pass B originally invalid: {inv.get('n_pass_b')}",
        f"- Retry validated: {retry_stats.get('validated', 0)}",
        f"- Retry failed: {retry_stats.get('failed', 0)}",
        f"- Still invalid after merge: {len(still)}",
        "",
        "## Retry configuration",
        "",
        f"- Model: {retry_stats.get('model')}",
        f"- Provider: {provider or retry_stats.get('provider') or 'default routing'}",
        f"- Structured JSON Schema: {retry_stats.get('use_json_schema', True)}",
        f"- Reasoning: none",
        f"- Retry cost USD: {retry_stats.get('cost_usd')}",
        f"- Retry prompt tokens: {retry_stats.get('prompt_tokens')}",
        f"- Retry completion tokens: {retry_stats.get('completion_tokens')}",
        "",
        "## Agreement before vs after",
        "",
        f"- Before exact agreement: {before_agree.get('exact_agreement_rate')}",
        f"- After exact agreement: {after_agree.get('exact_agreement_rate')}",
        f"- Before both-valid n: {before_agree.get('n_tasks')}",
        f"- After both-valid n: {after_agree.get('n_tasks')}",
        f"- Before root κ: {before_agree.get('root_cause_kappa')}",
        f"- After root κ: {after_agree.get('root_cause_kappa')}",
        "",
        "## High-priority handoff",
        "",
        f"- Before count: {len(hp_before)}",
        f"- After count: {len(hp_after)}",
        f"- Added: {sorted(hp_after_ids - hp_before_ids) or 'none'}",
        f"- Removed: {sorted(hp_before_ids - hp_after_ids) or 'none'}",
        "",
        "## Still invalid task IDs",
        "",
    ]
    for row in still:
        lines.append(f"- {row.get('pass')} `{row.get('task_id')}`")
    lines += [
        "",
        "## Integrity",
        "",
        "- Original raw/validated files preserved (see backup manifest).",
        "- Only invalid task/pass pairs were retried.",
        "- Pass B mapping unchanged.",
        "",
        f"## Final manifest SHA-256",
        "",
        f"- See `WEAK_AUDIT_FINAL_MANIFEST.json`",
        f"- Manifest file SHA-256: {sha256_file(out_dir / 'WEAK_AUDIT_FINAL_MANIFEST.json') if (out_dir / 'WEAK_AUDIT_FINAL_MANIFEST.json').is_file() else 'pending'}",
    ]
    (out_dir / "RETRY_FINALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_finalized_md(out_dir: Path, manifest: dict) -> None:
    text = """# Weak audit finalized

## Status

`weak_audit_finalized` — artifacts frozen for strong-model handoff.

## Interpretation

- Weak annotations are **hypotheses**, not ground truth.
- `first_divergence_turn` is relatively more stable across Pass A/B.
- `root_cause` and `recommended_fix` labels are **less stable**.
- Final artifacts are ready for a subsequent strong-model review phase.
- Any further changes require a **new audit version ID**.

## Counts

"""
    text += (
        f"- Case packets: {manifest.get('n_case_packets')}\n"
        f"- Pass A final valid: {manifest.get('pass_a_final_valid')}\n"
        f"- Pass B final valid: {manifest.get('pass_b_final_valid')}\n"
        f"- Pass A final invalid: {manifest.get('pass_a_final_invalid')}\n"
        f"- Pass B final invalid: {manifest.get('pass_b_final_invalid')}\n"
    )
    (out_dir / "WEAK_AUDIT_FINALIZED.md").write_text(text, encoding="utf-8")
