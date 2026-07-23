"""Repository path resolution for weak audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditPaths:
    repo: Path
    v3: Path
    run_dir: Path
    out_dir: Path
    nestful_test: Path
    eval_c0: Path
    eval_e1: Path
    eval_e2: Path
    analysis_json: Path
    discordant_jsonl: Path
    task_level_jsonl: Path
    run_manifest: Path
    ckpt_e1: Path
    ckpt_e2: Path


def repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p, *p.parents]:
        if (parent / "experiments" / "nestful_synthetic_curriculum_v3").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root")


def default_paths(
    run_dir: Path | None = None,
    out_dir: Path | None = None,
) -> AuditPaths:
    repo = repo_root()
    v3 = repo / "experiments/nestful_synthetic_curriculum_v3"
    run = run_dir or (v3 / "outputs/runs/pure_stage3_2ep_20260719_221918")
    out = out_dir or (v3 / "reports/pure_stage3_weak_audit")
    eval_root = run / "eval"
    offline = v3 / "reports/pure_stage3_offline_analysis"
    return AuditPaths(
        repo=repo,
        v3=v3,
        run_dir=run,
        out_dir=out,
        nestful_test=repo / "experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl",
        eval_c0=eval_root / "C0_test",
        eval_e1=eval_root / "S3_E1_test",
        eval_e2=eval_root / "S3_E2_test",
        analysis_json=offline / "analysis_c0_e1_e2_test_overnight.json",
        discordant_jsonl=offline / "PURE_STAGE3_DISCORDANT_AUDIT.jsonl",
        task_level_jsonl=offline / "pure_stage3_task_level_analysis.jsonl",
        run_manifest=run / "run_manifest.json",
        ckpt_e1=run / "checkpoints/S3_E1",
        ckpt_e2=run / "checkpoints/S3_E2",
    )
