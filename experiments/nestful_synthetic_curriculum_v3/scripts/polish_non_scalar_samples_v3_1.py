#!/usr/bin/env python3
"""Safely replace scalar-heavy stage3/4 samples with non-scalar outputs (v3.1 polish)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_prefix_curriculum_from_trajectories import (  # noqa: E402
    NON_SCALAR_CLUSTERS,
    STAGE_FILES,
    _make_prefix_sample,
    _try_register,
)
from generate_full_motif_trajectories_v3_1 import generate_one  # noqa: E402
from motif_lib import load_jsonl, repo_root  # noqa: E402
from process_filter_prefix_samples import check_sample  # noqa: E402
from question_templates_v3_1 import is_non_scalar_answer, validate_question_trace_alignment  # noqa: E402
from replay_synthetic_gold_traces_v3_1 import replay_sample  # noqa: E402
from traj_utils_v3_1 import truncate_trajectory  # noqa: E402
from uniqueness_utils_v3_1 import StageDedupRegistry, compute_signatures  # noqa: E402

STAGE_TARGETS = {
    "stage3_3call_composition": {"min_share": 0.25, "ideal_share": 0.30, "prefix_len": 3},
    "stage4_4to6call_persistence": {"min_share": 0.30, "ideal_share": 0.30, "prefix_len": 4},
}

MATH_SEQ_PREFIXES = ("add->", "add->add", "add->multiply", "multiply->")


def _non_scalar_share(samples: List[dict]) -> float:
    return sum(1 for s in samples if is_non_scalar_answer(s.get("gold_answer"))) / max(len(samples), 1)


def _is_scalar_math_candidate(sample: dict) -> bool:
    if is_non_scalar_answer(sample.get("gold_answer")):
        return False
    seq = "->".join(c.get("name", "") for c in sample.get("gold_calls") or [])
    return seq.startswith(MATH_SEQ_PREFIXES) or all(
        c.get("name") in ("add", "multiply", "subtract", "divide_safe") for c in sample.get("gold_calls") or []
    )


def _generate_replacement(
    stage: str,
    prefix_len: int,
    registry: StageDedupRegistry,
    rng: random.Random,
    counter: List[int],
) -> Optional[dict]:
    for _ in range(120):
        counter[0] += 1
        cid = rng.choice(list(NON_SCALAR_CLUSTERS))
        if stage == "stage4_4to6call_persistence":
            pl = rng.randint(4, 6)
            num_calls = pl
        else:
            pl = prefix_len
            num_calls = prefix_len
        try:
            traj = generate_one(rng, 990000 + counter[0], cid, num_calls=num_calls)
        except Exception:
            continue
        if traj.get("full_num_calls", 0) < pl:
            continue
        truncated = truncate_trajectory(traj, pl)
        if len(truncated.get("gold_calls") or []) != pl:
            continue
        sample = _make_prefix_sample(
            traj,
            pl,
            stage,
            counter[0],
            rng,
            terminal=(pl >= traj.get("full_num_calls", pl)),
        )
        if sample.get("num_calls") != pl or len(sample.get("gold_calls") or []) != pl:
            continue
        if not is_non_scalar_answer(sample.get("gold_answer")):
            continue
        ok, err = replay_sample(sample)
        if not ok:
            continue
        if validate_question_trace_alignment(
            sample.get("question", ""), sample.get("gold_calls") or [], num_calls=sample.get("num_calls")
        ):
            continue
        pf_ok, _ = check_sample(sample, stage)
        if not pf_ok:
            continue
        kept = _try_register(registry, sample, rng)
        if kept:
            return kept
    return None


def polish_stage(
    samples: List[dict],
    stage: str,
    rng: random.Random,
    counter: List[int],
) -> Tuple[List[dict], dict]:
    cfg = STAGE_TARGETS[stage]
    target_share = cfg["ideal_share"]
    prefix_len = cfg["prefix_len"]
    current_share = _non_scalar_share(samples)
    stats = {
        "stage": stage,
        "before_non_scalar_share": round(current_share, 4),
        "replacements": 0,
        "after_non_scalar_share": round(current_share, 4),
        "status": "skipped",
    }
    if current_share >= target_share:
        stats["status"] = "already_met"
        return samples, stats

    need = int(len(samples) * target_share) - sum(1 for s in samples if is_non_scalar_answer(s.get("gold_answer")))
    if need <= 0:
        stats["status"] = "already_met"
        return samples, stats

    registry = StageDedupRegistry(stage, stage_target=len(samples), max_trace_count=1, max_template_count=10)
    for s in samples:
        registry.register(s, compute_signatures(s))

    out = list(samples)
    candidates = [i for i, s in enumerate(out) if _is_scalar_math_candidate(s)]
    rng.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        if replaced >= need:
            break
        repl = _generate_replacement(stage, prefix_len, registry, rng, counter)
        if repl:
            out[idx] = repl
            replaced += 1

    after_share = _non_scalar_share(out)
    stats["replacements"] = replaced
    stats["after_non_scalar_share"] = round(after_share, 4)
    stats["status"] = "polished" if replaced > 0 else "no_replacement_found"
    if after_share < cfg["min_share"]:
        stats["status"] = "partial_warn"
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1",
    )
    ap.add_argument("--seed", type=int, default=42001)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stages",
        default="stage3_3call_composition",
        help="Comma-separated stages to polish (default: stage3 only)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    counter = [900000]
    all_stats: Dict[str, dict] = {}
    modified = False

    target_stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    for stage in target_stages:
        if stage not in STAGE_TARGETS:
            print(f"[polish] unknown stage {stage}", file=sys.stderr)
            return 1
        fname = STAGE_FILES[stage]
        path = args.out_dir / fname
        if not path.is_file():
            print(f"[polish] missing {path}", file=sys.stderr)
            return 1
        samples = load_jsonl(path)
        polished, stats = polish_stage(samples, stage, rng, counter)
        all_stats[stage] = stats
        if stats["replacements"] > 0:
            modified = True
            if not args.dry_run:
                with open(path, "w", encoding="utf-8") as fh:
                    for s in polished:
                        fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    report = {
        "modified": modified,
        "dry_run": args.dry_run,
        "stages": all_stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "non_scalar_polish_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for stage, st in all_stats.items():
        print(
            f"[polish] {stage}: {st['before_non_scalar_share']:.3f} -> {st['after_non_scalar_share']:.3f} "
            f"({st['replacements']} replacements, {st['status']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
