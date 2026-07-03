#!/usr/bin/env python3
"""Orchestrate v3.1 curriculum build pipeline (CPU only)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V3 = REPO / "experiments/nestful_synthetic_curriculum_v3"
OUT = V3 / "outputs/curriculum_v3_1"

STEPS = [
    [sys.executable, str(V3 / "scripts/generate_full_motif_trajectories_v3_1.py")],
    [sys.executable, str(V3 / "scripts/build_prefix_curriculum_from_trajectories.py")],
    [sys.executable, str(V3 / "scripts/polish_non_scalar_samples_v3_1.py"), "--stages", "stage3_3call_composition"],
    [sys.executable, str(V3 / "scripts/process_filter_prefix_samples.py")],
    [sys.executable, str(V3 / "scripts/final_dataset_audit_v3_1.py"), "--use-filtered"],
    [sys.executable, str(V3 / "scripts/analyze_dataset_uniqueness_v3_1.py"), "--use-filtered"],
    [sys.executable, str(V3 / "scripts/validate_question_trace_alignment_v3_1.py")],
    [sys.executable, str(V3 / "scripts/validate_curriculum_integrity_v3_1.py")],
    [sys.executable, str(V3 / "scripts/replay_synthetic_gold_traces_v3_1.py")],
    [sys.executable, str(V3 / "scripts/run_tool_family_realism_v3_1.py")],
    [sys.executable, str(V3 / "scripts/run_preflight_gates.py"), "--prototype-only", "--curriculum-version", "v3_1"],
    [sys.executable, "-m", "pytest", str(V3 / "tests"), "-q"],
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for cmd in STEPS:
        print(f"[build_pipeline] running: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO))
        results.append({"command": cmd, "exit_code": rc})
        if rc != 0:
            failed = True
            break

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": not failed,
        "steps": results,
    }
    (OUT / "build_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Build Pipeline Report (v3.1)",
        "",
        f"Success: **{not failed}**",
        "",
        "## Steps",
    ]
    for r in results:
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        report.append(f"- {status}: `{' '.join(r['command'])}`")
    (OUT / "BUILD_PIPELINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
