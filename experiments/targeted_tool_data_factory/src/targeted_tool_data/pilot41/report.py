"""PILOT41_IMPLEMENTATION_REPORT builder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..repro import stamp, write_json, write_text
from . import SCHEMA_VERSION

REPORT_SCHEMA = "ttdf.pilot41_implementation_report.v1"


def build_report(repo_root: Path, out_dir: Path, *,
                 cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    module = repo_root / "experiments" / "targeted_tool_data_factory"
    run = module / "outputs" / "pilot4_1_profile_safe"
    lang = module / "reports" / "pilot4_language_audit" / "PILOT4_LANGUAGE_AUDIT.json"
    audit = module / "reports" / "pilot4_vs_pilot41" / "PILOT4_VS_PILOT41_AUDIT.json"
    freeze = {}
    if (run / "freeze_manifest.json").is_file():
        freeze = json.loads((run / "freeze_manifest.json").read_text(encoding="utf-8"))
    lang_j = json.loads(lang.read_text(encoding="utf-8")) if lang.is_file() else {}
    audit_j = json.loads(audit.read_text(encoding="utf-8")) if audit.is_file() else {}
    usage = {}
    if (run / "openrouter_usage_summary.json").is_file():
        usage = json.loads((run / "openrouter_usage_summary.json").read_text(
            encoding="utf-8"))

    repro = stamp(repo_root, schema_version=REPORT_SCHEMA, cli_args=cli_args)
    payload = {
        "schema_version": REPORT_SCHEMA,
        "pilot41_schema_version": SCHEMA_VERSION,
        "repro": repro,
        "freeze": freeze,
        "pilot4_language_audit": lang_j.get("train") or {},
        "comparison": audit_j.get("claim_classes") or {},
        "openrouter_usage": usage,
        "executive": {
            "IMPLEMENTED": [
                "semantic types + validate_semantic_edge",
                "workflow grammar (~40 families)",
                "non-leak deterministic query modes",
                "V9–V13 validators",
                "dense CORE cells",
                "OpenRouter writer/critic with budget + replay",
            ],
            "GENERATED": freeze.get("counts") or {},
            "DETERMINISTICALLY_VERIFIED": [
                "graph leak metrics vs Pilot4",
                "fact/unit preservation",
                "family-safe split",
                "executable oracle-before-query",
            ],
            "LLM_VALIDATED": bool(usage),
            "HUMAN_REVIEW_REQUIRED": True,
            "NOT_TESTED_BY_TRAINING": True,
            "NOT_TESTED_BY_NESTFUL": True,
        },
    }
    md = [
        "# Pilot4.1 implementation report",
        "",
        f"- commit: `{repro['git'].get('commit')}` dirty={repro['git'].get('dirty')}",
        f"- generated: {repro['generated_at_utc']}",
        "",
        "## 1. Executive summary",
        "",
        "### IMPLEMENTED",
        "",
        *[f"- {x}" for x in payload["executive"]["IMPLEMENTED"]],
        "",
        "### GENERATED",
        "",
        f"```json\n{json.dumps(payload['executive']['GENERATED'], indent=2)}\n```",
        "",
        "### DETERMINISTICALLY VERIFIED",
        "",
        *[f"- {x}" for x in payload["executive"]["DETERMINISTICALLY_VERIFIED"]],
        "",
        f"### LLM-VALIDATED: {payload['executive']['LLM_VALIDATED']}",
        "",
        "### HUMAN-REVIEW REQUIRED / NOT TESTED BY TRAINING / NOT TESTED BY NESTFUL",
        "",
        "No claim is made that Pilot4.1 improves NESTFUL official win.",
        "",
        "## 2. Existing implementation audit",
        "",
        f"Pilot4 stages_related_rate (train): "
        f"{(lang_j.get('train') or {}).get('stages_related_phrase_rate')}",
        f"Pilot4 high_or_complete graph leak: "
        f"{(lang_j.get('train') or {}).get('high_or_complete_rate')}",
        "",
        "## 9. Cost and request statistics",
        "",
        f"```json\n{json.dumps(usage, indent=2)}\n```",
        "",
        "## 11. Dataset composition",
        "",
        f"```json\n{json.dumps(freeze.get('counts'), indent=2)}\n```",
        "",
        "## 17. Reproduction commands",
        "",
        "```bash",
        "python -m targeted_tool_data.cli audit-pilot4-language",
        "python -m targeted_tool_data.cli build-workflow-registry",
        "python -m targeted_tool_data.cli generate-semantic-pilot41 --candidates 10000",
        "python -m targeted_tool_data.cli select-render-shortlist --target 2000",
        "python -m targeted_tool_data.cli render-queries-openrouter --stage smoke",
        "python -m targeted_tool_data.cli render-queries-openrouter --stage pilot",
        "python -m targeted_tool_data.cli render-queries-openrouter --stage full",
        "python -m targeted_tool_data.cli select-pilot41 --train 1000 --heldout 250 --reserve 250",
        "python -m targeted_tool_data.cli audit-pilot41",
        "python -m targeted_tool_data.cli implementation-report-pilot41",
        "# replay freeze without API:",
        "python -m targeted_tool_data.cli render-queries-openrouter --stage full --replay",
        "```",
        "",
        "## 18. Recommended training experiment, not executed",
        "",
        "- Train MT-GRPO on pilot4_1 train-1000 with history-adaptive sampler and new logs.",
        "- Matched-engine paired eval vs Pilot3/4 adapters before claiming NESTFUL gains.",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "PILOT41_IMPLEMENTATION_REPORT.json", payload)
    write_text(out_dir / "PILOT41_IMPLEMENTATION_REPORT.md", "\n".join(md))
    return payload
