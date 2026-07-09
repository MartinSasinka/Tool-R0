#!/usr/bin/env python3
"""Build curriculum v3 stage files from generated synthetic tasks."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root  # noqa: E402

STAGE_FILES = {
    "stage1_linear_simple": "stage1_linear_simple.jsonl",
    "stage2_reference_reuse": "stage2_reference_reuse.jsonl",
    "stage3_structural_motifs": "stage3_structural_motifs.jsonl",
    "stage4_nestful_like_mixed": "stage4_nestful_like_mixed.jsonl",
}

REF_COMPLEXITY = {
    "low": lambda ref: ref.get("num_references", 0) <= 1,
    "medium": lambda ref: ref.get("num_references", 0) <= 3,
    "medium_high": lambda ref: ref.get("num_references", 0) <= 5,
    "high": lambda ref: True,
}


def _assign_stage(task: dict, stages_cfg: dict) -> str | None:
    motif = task.get("motif_type", "")
    ncalls = task.get("num_calls", 0)
    ref = task.get("reference_pattern") or {}
    for stage_name, rules in stages_cfg.items():
        if stage_name == "stage_minimums":
            continue
        motifs = set(rules.get("motifs") or [])
        max_calls = int(rules.get("max_calls", 99))
        rc = rules.get("reference_complexity", "high")
        if motif in motifs and ncalls <= max_calls and REF_COMPLEXITY.get(rc, lambda _r: True)(ref):
            return stage_name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/synthetic_motif_tasks.jsonl")
    ap.add_argument("--config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/curriculum_v3.yaml")
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if yaml and args.config.is_file() else {}
    stages_cfg = {k: v for k, v in (cfg.get("stages") or {}).items()}
    stage_mins = cfg.get("stage_minimums") or {}
    tasks = load_jsonl(args.input)
    buckets = {k: [] for k in STAGE_FILES}
    unassigned = []

    for t in tasks:
        stage = _assign_stage(t, stages_cfg)
        if stage and stage in buckets:
            buckets[stage].append(t)
        else:
            unassigned.append(t)

    buckets["stage4_nestful_like_mixed"].extend(unassigned)

    short_stages = {
        s: stage_mins.get(s, 0) - len(buckets.get(s, []))
        for s in STAGE_FILES
        if stage_mins.get(s, 0) > len(buckets.get(s, []))
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stages": {},
        "unassigned_count": len(unassigned),
        "stage_minimums": stage_mins,
        "stage_minimum_shortfalls": short_stages,
        "advance_gates": cfg.get("advance_gates", {}),
    }
    for stage, fname in STAGE_FILES.items():
        path = args.out_dir / fname
        with open(path, "w", encoding="utf-8") as fh:
            for t in buckets[stage]:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        motifs = Counter(t.get("motif_type") for t in buckets[stage])
        manifest["stages"][stage] = {
            "file": str(path),
            "count": len(buckets[stage]),
            "motif_coverage": dict(motifs),
        }

    with open(args.out_dir / "curriculum_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[build_curriculum_v3] wrote {args.out_dir} "
          f"({sum(len(buckets[s]) for s in STAGE_FILES)} tasks)")
    for stage in STAGE_FILES:
        print(f"  {stage}: {len(buckets[stage])}")
    if short_stages:
        print(f"WARNING: stage minimum shortfalls: {short_stages}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
