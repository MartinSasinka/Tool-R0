#!/usr/bin/env python3
"""Build the curriculum v5 registry-driven synthetic dataset.

Every accepted row must:
  1. replay through the generator-side executor (in-generation gate);
  2. replay through the REAL trainer executor (nestful_mtgrpo_minimal
     ToolExecutor, mode="synthetic") and reproduce gold_answer;
  3. not collide with NESTFUL by question hash or trace hash (when the NESTFUL
     data file is present).

Writes:
  <out>/filtered/<stage>.jsonl
  <out>/manifests/dataset_manifest.json   (registry version+hash, file sha256s,
                                           seed, counts, generator version)

Usage:
  python build_v5_dataset.py --pilot                      # 40/stage
  python build_v5_dataset.py --examples-per-stage 800
  python build_v5_dataset.py --stages v5_stage1_2call --examples-per-stage 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
sys.path.insert(0, _V3)
sys.path.insert(0, _MINIMAL)

from lib.synthetic_gen_v5 import (  # noqa: E402
    DiversityConfig, GENERATOR_VERSION, STAGES, generate_stage,
    question_hash, trace_hash,
)
from lib.synthetic_tools import REGISTRY_VERSION, registry_hash  # noqa: E402


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nestful_hashes(repo_root: str):
    """Question/trace hashes of the NESTFUL benchmark for contamination gate."""
    candidates = [
        os.path.join(_MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"),
        os.path.join(repo_root, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"),
    ]
    qs, ts = set(), set()
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = row.get("input") or row.get("question") or ""
                if q:
                    qs.add(question_hash(q))
                out = row.get("output") or []
                if isinstance(out, str):
                    try:
                        out = json.loads(out)
                    except json.JSONDecodeError:
                        out = []
                if isinstance(out, list) and out:
                    try:
                        ts.add(trace_hash(out))
                    except (TypeError, KeyError):
                        pass
        print(f"[build_v5] contamination reference: {path} "
              f"({len(qs)} question hashes, {len(ts)} trace hashes)")
        return qs, ts
    print("[build_v5] WARNING: NESTFUL data file not found — "
          "contamination gate limited to hash-forbid sets only")
    return qs, ts


def _replay_through_trainer_executor(rows) -> int:
    """Replay every row through the REAL ToolExecutor (mode=synthetic)."""
    from executor import ToolExecutor, matches_gold
    n_fail = 0
    for r in rows:
        task = {"task_id": r["sample_id"], "question": r["question"],
                "tools": r["tools"], "gold_calls": r["gold_calls"],
                "gold_answer": r["gold_answer"], "num_calls": r["num_calls"]}
        ex = ToolExecutor(task, registry=None, mode="synthetic")
        obs, err = None, None
        for call in r["gold_calls"]:
            res = ex.execute(call)
            if res.error:
                err = res.error
                break
            obs = res.observation
        if err is not None or not matches_gold(obs, r["gold_answer"]):
            n_fail += 1
            print(f"[build_v5] TRAINER-REPLAY FAIL {r['sample_id']}: "
                  f"err={err} obs={obs!r} gold={r['gold_answer']!r}")
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="*", default=list(STAGES.keys()))
    ap.add_argument("--examples-per-stage", type=int, default=800)
    ap.add_argument("--pilot", action="store_true", help="40 examples per stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir",
                    default=os.path.join(_V3, "data", "curriculum_v5_registry"))
    ap.add_argument("--max-tool-share", type=float, default=0.08)
    args = ap.parse_args()

    n_per_stage = 40 if args.pilot else args.examples_per_stage
    cfg = DiversityConfig(max_tool_share=args.max_tool_share)

    for s in args.stages:
        if s not in STAGES:
            print(f"[build_v5] ERROR: unknown stage '{s}'. "
                  f"Known: {sorted(STAGES)}", file=sys.stderr)
            return 2

    out_dir = os.path.abspath(args.output_dir)
    filtered_dir = os.path.join(out_dir, "filtered")
    manifest_dir = os.path.join(out_dir, "manifests")
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    print(f"[build_v5] generator={GENERATOR_VERSION} registry={REGISTRY_VERSION} "
          f"hash={registry_hash()[:16]}…")
    print(f"[build_v5] stages={args.stages} n/stage={n_per_stage} seed={args.seed}")

    forb_q, forb_t = _nestful_hashes(os.path.normpath(os.path.join(_V3, "..", "..")))

    files = {}
    seen_q, seen_t = set(forb_q), set(forb_t)
    for stage in args.stages:
        rows = generate_stage(stage, n_per_stage, args.seed, cfg,
                              forbidden_question_hashes=seen_q,
                              forbidden_trace_hashes=seen_t)
        for r in rows:
            seen_q.add(question_hash(r["question"]))
            seen_t.add(trace_hash(r["gold_calls"]))

        n_fail = _replay_through_trainer_executor(rows)
        if n_fail:
            print(f"[build_v5] ABORT: {n_fail} rows failed trainer-executor replay",
                  file=sys.stderr)
            return 1

        path = os.path.join(filtered_dir, f"{stage}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[stage] = {"path": path, "rows": len(rows),
                        "sha256": _sha256_file(path)}
        print(f"[build_v5] {stage}: {len(rows)} rows -> {path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "seed": args.seed,
        "examples_per_stage": n_per_stage,
        "max_tool_share": args.max_tool_share,
        "trainer_replay": "all rows replayed through ToolExecutor(mode=synthetic)",
        "contamination": {
            "nestful_question_hashes_checked": len(forb_q),
            "nestful_trace_hashes_checked": len(forb_t),
        },
        "files": files,
    }
    mpath = os.path.join(manifest_dir, "dataset_manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[build_v5] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
