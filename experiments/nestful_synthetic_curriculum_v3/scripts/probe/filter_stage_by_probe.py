"""Signal-positive stage filtering from a probe run (Phase 1e / RESEARCH_FIX_PLAN E2).

Takes the output directory of scripts/probe/probe_stage.py and produces a
filtered training JSONL containing ONLY the tasks whose probe groups produced
>= 2 unique rewards (i.e. tasks where GRPO would receive a gradient), plus a
dead-low task list for SFT use, plus a provenance manifest.

The ORIGINAL dataset file is never modified; the trainer receives the filtered
file as an ordinary explicit dataset path.

Determinism: no randomness here — selection is fully determined by the probe
report; output preserves source-file row order.

Usage (repo root):
    python .../filter_stage_by_probe.py \
        --probe-dir experiments/nestful_synthetic_curriculum_v3/outputs/probes/probe_2_... \
        [--output-dir .../outputs/filtered_by_probe/<name>] [--min-unique-rewards 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))

from paths import REPO_ROOT, V3_ROOT, dataset_info, sha256_file  # noqa: E402
from run_manifest import build_manifest, write_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a signal-positive filtered dataset from probe output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--probe-dir", required=True,
                    help="probe output dir containing PROBE_REPORT.json")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--min-unique-rewards", type=int, default=2,
                    help="keep tasks whose probe group had >= this many unique rewards")
    ap.add_argument("--dataset", default=None,
                    help="source dataset override (default: path recorded in probe report)")
    args = ap.parse_args()

    probe_dir = os.path.abspath(args.probe_dir)
    report_path = os.path.join(probe_dir, "PROBE_REPORT.json")
    if not os.path.isfile(report_path):
        print(f"[filter] ERROR: {report_path} not found", file=sys.stderr)
        return 1
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    if report.get("stub_warning"):
        print("[filter] WARNING: probe used the STUB backend — this filtered set is a "
              "pipeline self-test artifact, NOT a training set.", file=sys.stderr)

    dataset_path = os.path.abspath(
        args.dataset or os.path.join(REPO_ROOT, report["dataset"]["path"]))
    if not os.path.isfile(dataset_path):
        print(f"[filter] ERROR: source dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    src_sha = sha256_file(dataset_path)
    if not args.dataset and src_sha != report["dataset"]["sha256"]:
        print("[filter] ERROR: source dataset SHA changed since the probe ran "
              f"(probe={report['dataset']['sha256'][:12]}… now={src_sha[:12]}…). "
              "Re-run the probe.", file=sys.stderr)
        return 2

    groups = report.get("groups") or []
    keep_ids = {str(g["task_id"]) for g in groups
                if int(g["unique_rewards"]) >= args.min_unique_rewards}
    dead_low_ids = {str(g["task_id"]) for g in groups
                    if g["dead"] and g["mean_reward"] <= 0.35}
    probed_ids = {str(g["task_id"]) for g in groups}

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    out_dir = os.path.abspath(
        args.output_dir
        or os.path.join(V3_ROOT, "outputs", "filtered_by_probe", f"{stem}_{ts}"))
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{stem}.signal_positive.jsonl")
    out_dead_low = os.path.join(out_dir, f"{stem}.dead_low_for_sft.jsonl")

    kept = dropped_dead = not_probed = 0
    kept_motifs: Counter = Counter()
    with open(dataset_path, encoding="utf-8") as src, \
            open(out_jsonl, "w", encoding="utf-8") as f_keep, \
            open(out_dead_low, "w", encoding="utf-8") as f_dead:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("sample_id") or f"task_{idx}")
            if rid in keep_ids:
                f_keep.write(line + "\n")
                kept += 1
                kept_motifs[str(row.get("motif_type"))] += 1
            elif rid in dead_low_ids:
                f_dead.write(line + "\n")
                dropped_dead += 1
            elif rid in probed_ids:
                dropped_dead += 1  # dead but not low (e.g. saturated-high)
            else:
                not_probed += 1

    manifest = build_manifest(
        kind="probe_filtered_dataset",
        datasets=[dataset_path],
        extra={
            "probe_dir": os.path.relpath(probe_dir, REPO_ROOT).replace("\\", "/"),
            "probe_backend": report.get("backend"),
            "probe_reward": report.get("reward"),
            "probe_decoding": report.get("decoding"),
            "min_unique_rewards": args.min_unique_rewards,
            "source_sha256": src_sha,
            "n_probed": len(probed_ids),
            "kept_signal_positive": kept,
            "dropped_dead": dropped_dead,
            "not_probed_dropped": not_probed,
            "kept_motif_distribution": dict(kept_motifs),
            "dead_low_for_sft": os.path.relpath(out_dead_low, REPO_ROOT).replace("\\", "/"),
            "output": dataset_info(out_jsonl) if kept else None,
        },
    )
    write_manifest(manifest, os.path.join(out_dir, "manifest.json"))

    print(f"[filter] source        : {dataset_path}")
    print(f"[filter] probed        : {len(probed_ids)} | kept signal-positive: {kept} | "
          f"dropped dead: {dropped_dead} | not probed (dropped): {not_probed}")
    print(f"[filter] filtered file : {out_jsonl}")
    print(f"[filter] dead-low (SFT): {out_dead_low}")
    print(f"[filter] manifest      : {os.path.join(out_dir, 'manifest.json')}")
    if kept == 0:
        print("[filter] WARNING: zero signal-positive tasks — do not train on this.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
