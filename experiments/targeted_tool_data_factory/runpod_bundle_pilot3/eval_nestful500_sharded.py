#!/usr/bin/env python3
"""Shard NESTFUL-500 final_eval across multiple GPUs, then merge results.

final_eval is single-engine (no train-time DP pool). To use 4 GPUs we split the
diagnostic JSONL into N contiguous shards, run one ``run.py --mode final_eval``
per GPU, and concatenate trajectories + recompute official win rate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# nestful_mtgrpo_partial/config.yaml defaults hardware.use_vllm=false.
# USE_VLLM=1 in the shell env is NOT read by final_eval — must override here.
DECODING = [
    "--override", "hardware.use_vllm=true",
    "--override", "generation.temperature=0.0",
    "--override", "generation.top_p=1.0",
    "--override", "data.num_eval_rollouts=1",
    "--override", "data.eval_paradigm=react",
]


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for cand in (run_dir / "checkpoints" / "final",
                 run_dir / "final",
                 run_dir / "checkpoints" / "FINAL"):
        if (cand / "adapter_config.json").is_file():
            return cand
    hits = sorted(run_dir.rglob("adapter_config.json"))
    return hits[-1].parent if hits else None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_shards(rows: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    if n < 1:
        raise ValueError("n_gpus must be >= 1")
    # Contiguous shards keep task order stable after merge.
    size = (len(rows) + n - 1) // n
    return [rows[i * size:(i + 1) * size] for i in range(n) if rows[i * size:(i + 1) * size]]


def official_win_rate(trajs: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = 0
    n = 0
    for row in trajs:
        # Prefer explicit official_win; fall back to nested _traj / metrics.
        w = row.get("official_win")
        if w is None:
            w = (row.get("_traj") or {}).get("official_win")
        if w is None:
            w = (row.get("metrics") or {}).get("official_win")
        if w is None:
            continue
        n += 1
        wins += int(bool(w))
    return {
        "n_scored": n,
        "n_rows": len(trajs),
        "official_win": (wins / n) if n else None,
        "n_wins": wins,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Training run directory containing the final adapter "
                         "(omit with --base-model for C0 / no-LoRA eval)")
    ap.add_argument("--base-model", action="store_true",
                    help="Eval base instruct checkpoint (no LoRA). "
                         "Sets model.lora_adapter=null; --run-dir not required.")
    ap.add_argument("--arm", default=None,
                    help="Label written into metrics_merged.json (default D1, or C0 with --base-model)")
    ap.add_argument("--diagnostic", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-py", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        print("[eval4] ABORT: empty --gpus", file=sys.stderr)
        return 2

    if not args.base_model and args.run_dir is None:
        print("[eval4] ABORT: pass --run-dir or --base-model", file=sys.stderr)
        return 2

    # Absolute paths are mandatory: nestful_mtgrpo_minimal/run.py resolves
    # relative paths from its CWD and, if the file is missing, copies the FULL
    # nestful_data.jsonl (1861) over the configured path — which silently turns
    # a 500-shard eval into a 1861-task run.
    diagnostic = args.diagnostic.resolve()
    out = args.out_dir.resolve()
    run_py = args.run_py.resolve()
    config = args.config.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None

    ck = None
    if args.base_model:
        print("[eval4] --base-model: no LoRA adapter (C0-style)")
    else:
        ck = find_checkpoint(run_dir)
        if ck is None and not args.dry_run:
            print(f"[eval4] ABORT: no final adapter under {run_dir}",
                  file=sys.stderr)
            print("[eval4] hint: for base C0 use --base-model (no --run-dir)",
                  file=sys.stderr)
            return 2
        if ck is not None:
            ck = ck.resolve()

    rows = read_jsonl(diagnostic)
    if not rows:
        print(f"[eval4] ABORT: empty diagnostic {diagnostic}", file=sys.stderr)
        return 2
    if len(rows) != 500:
        print(f"[eval4] WARN: diagnostic has {len(rows)} rows (expected 500)",
              file=sys.stderr)

    shards = split_shards(rows, len(gpus))
    # Align GPU list to non-empty shards (e.g. tiny dry datasets).
    gpus = gpus[:len(shards)]
    shards_root = out / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    arm = args.arm or ("C0" if args.base_model else "D1")
    print(f"[eval4] checkpoint: {ck if ck is not None else '(base model, no adapter)'}")
    print(f"[eval4] diagnostic: {diagnostic} ({len(rows)} rows)")
    print(f"[eval4] gpus: {gpus}  shards: {[len(s) for s in shards]}")
    print(f"[eval4] arm: {arm}")
    print(f"[eval4] out-dir (abs): {out}")

    procs: List[subprocess.Popen] = []
    shard_dirs: List[Path] = []
    for i, (gpu, shard) in enumerate(zip(gpus, shards)):
        shard_dir = (shards_root / f"gpu{gpu}").resolve()
        shard_data = (shard_dir / "data.jsonl").resolve()
        write_jsonl(shard_data, shard)
        n_written = sum(1 for _ in shard_data.open(encoding="utf-8") if _.strip())
        if n_written != len(shard):
            print(f"[eval4] ABORT: shard write mismatch {n_written}!={len(shard)}",
                  file=sys.stderr)
            return 2
        if not shard_data.is_file():
            print(f"[eval4] ABORT: missing shard data {shard_data}", file=sys.stderr)
            return 2
        shard_dirs.append(shard_dir)
        cmd = [
            sys.executable, str(run_py), "--mode", "final_eval",
            "--config", str(config),
            "--override", f"experiment.output_dir={shard_dir}",
            "--override", f"paths.full_nestful_jsonl={shard_data}",
            *DECODING,
        ]
        if ck is not None:
            cmd += ["--checkpoint", str(ck)]
        else:
            cmd += ["--override", "model.lora_adapter=null"]
        (shard_dir / "eval_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        print(f"[eval4][gpu{gpu}] {len(shard)} tasks -> {shard_data}")
        if args.dry_run:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["USE_VLLM"] = "1"
        # Blackwell / sm_120 sampler warmup workaround (same as run_final_eval_v2_parallel.sh)
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        env.pop("SYNTHETIC_TOOLS_DIR", None)
        log_f = (shard_dir / "eval.log").open("w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT))

    if args.dry_run:
        print("[eval4] DRY RUN — not launching workers")
        return 0

    t0 = time.time()
    codes = [p.wait() for p in procs]
    elapsed = time.time() - t0
    print(f"[eval4] workers finished in {elapsed/60:.1f} min; rcs={codes}")
    if any(c != 0 for c in codes):
        print("[eval4] ABORT: one or more shard workers failed", file=sys.stderr)
        return 1

    merged: List[Dict[str, Any]] = []
    for shard_dir in shard_dirs:
        traj = shard_dir / "final_eval_trajectories.jsonl"
        if not traj.is_file():
            # Some runners nest under a subdir; search once.
            hits = list(shard_dir.rglob("final_eval_trajectories.jsonl"))
            if not hits:
                print(f"[eval4] ABORT: missing trajectories under {shard_dir}",
                      file=sys.stderr)
                return 1
            traj = hits[0]
        merged.extend(read_jsonl(traj))

    if len(merged) != len(rows):
        print(f"[eval4] ABORT: merged {len(merged)} trajs vs {len(rows)} diagnostic rows "
              f"(likely relative-path overwrite with full NESTFUL-1861)",
              file=sys.stderr)
        return 1

    write_jsonl(out / "final_eval_trajectories.jsonl", merged)
    summary = official_win_rate(merged)
    summary.update({
        "checkpoint": str(ck) if ck else None,
        "diagnostic": str(diagnostic),
        "n_diagnostic": len(rows),
        "n_gpus": len(gpus),
        "shard_sizes": [len(s) for s in shards],
        "worker_returncodes": codes,
        "elapsed_sec": round(elapsed, 1),
        "arm": arm,
    })
    # Prefer per-shard metrics_official if present (sum wins / n).
    wins = 0
    scored = 0
    for shard_dir in shard_dirs:
        for mo in shard_dir.rglob("metrics_official.json"):
            try:
                m = json.loads(mo.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Common shapes: {"official_win": 0.5, "n": 125} or nested.
            if isinstance(m.get("official_win"), (int, float)) and m.get("n"):
                scored += int(m["n"])
                wins += int(round(float(m["official_win"]) * int(m["n"])))
            elif isinstance(m.get("official_win_rate"), (int, float)) and m.get("n"):
                scored += int(m["n"])
                wins += int(round(float(m["official_win_rate"]) * int(m["n"])))
    if scored:
        summary["official_win_from_shard_metrics"] = wins / scored
        summary["n_scored_from_shard_metrics"] = scored

    (out / "metrics_merged.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out / "eval_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[eval4] wrote {out / 'final_eval_trajectories.jsonl'}")
    print(f"[eval4] wrote {out / 'metrics_merged.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
