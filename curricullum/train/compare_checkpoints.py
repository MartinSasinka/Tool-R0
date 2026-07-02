"""Compare multiple training checkpoints against the NESTFUL benchmark.

Runs evaluate_nestful_stage.py for every (checkpoint, n_calls) combination
you specify and prints a side-by-side comparison table at the end.

Usage examples
--------------
# Compare the three best checkpoints on all their relevant n_calls groups:
python curricullum/train/compare_checkpoints.py \\
    --nestful_path eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \\
    --call_dist_path helper_calculations/output/nestful_call_distribution.json \\
    --output_dir curricullum/training/results/compare_20260615 \\
    --checkpoints \\
        "baseline:none" \\
        "stage_1_e2:/mnt/raid/data/llm_irafm_shared/Tool-R0/curricullum/checkpoints/qwen3_4b_curriculum_v2/stage_1_epoch2" \\
        "stage_2_e3:/mnt/raid/data/llm_irafm_shared/Tool-R0/curricullum/checkpoints/qwen3_4b_curriculum_v2/stage_2_epoch3" \\
        "stage_3_e1:/mnt/raid/data/llm_irafm_shared/Tool-R0/curricullum/checkpoints/qwen3_4b_curriculum_v2/stage_3_epoch1" \\
    --n_calls 2 3 4 \\
    --max_tasks 150 \\
    --eval_gpus 1,2

# Quick single-checkpoint evaluation (same as baseline but for a trained model):
python curricullum/train/compare_checkpoints.py \\
    --nestful_path eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \\
    --call_dist_path helper_calculations/output/nestful_call_distribution.json \\
    --output_dir curricullum/training/results/compare_quick \\
    --checkpoints "stage_3_e1:curricullum/checkpoints/qwen3_4b_curriculum_v2/stage_3_epoch1" \\
    --n_calls 2 3 4 5 6 7 \\
    --eval_gpus 1,2

Checkpoint format
-----------------
Each --checkpoints entry is  "label:adapter_path"
  - label       : short name used in output (e.g. "baseline", "stage_2_e3")
  - adapter_path: path to a LoRA adapter directory, or "none" for bare base model

The base model is always Qwen/Qwen3-4B-Instruct-2507 (read from
curricullum/train/configs/qwen3_4b_curriculum_v2.yaml if present, else default).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "evaluate_nestful_stage.py"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BATCH_SIZE = 16


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_base_model(config_path: Optional[str]) -> str:
    if config_path and Path(config_path).is_file():
        try:
            import yaml  # type: ignore
            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            return cfg.get("model", {}).get("name", DEFAULT_BASE_MODEL)
        except Exception:
            pass
    return DEFAULT_BASE_MODEL


def _run_eval(
    *,
    label: str,
    adapter_path: Optional[str],
    n_calls: int,
    nestful_path: str,
    call_dist_path: str,
    output_json: str,
    base_model: str,
    max_new_tokens: int,
    batch_size: int,
    max_tasks: Optional[int],
    gpu_ids: List[int],
    num_shards: int,
    shard_id: int,
) -> int:
    """Launch one evaluate_nestful_stage.py subprocess. Returns exit code."""
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    cmd = [
        sys.executable, "-u", str(EVAL_SCRIPT),
        "--base_model",        base_model,
        "--nestful_path",      nestful_path,
        "--call_dist_path",    call_dist_path,
        "--n_calls",           str(n_calls),
        "--max_new_tokens",    str(max_new_tokens),
        "--batch_size",        str(batch_size),
        "--save_failures",     "10",
        "--output_json",       output_json,
        "--shard_id",          str(shard_id),
        "--num_shards",        str(num_shards),
    ]
    if adapter_path and adapter_path.lower() != "none":
        cmd += ["--adapter_path", adapter_path]
    else:
        cmd += ["--no_adapter"]
    if max_tasks is not None:
        cmd += ["--max_tasks", str(max_tasks)]

    label_str = f"{label} n={n_calls} shard{shard_id}/{num_shards}"
    print(f"[eval] starting  {label_str}")
    print(f"[eval]   $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED(exit {proc.returncode})"
    print(f"[eval] done      {label_str}  {status}  {elapsed:.0f}s", flush=True)
    return proc.returncode


def _merge_shards(shard_paths: List[str]) -> Optional[Dict]:
    """Merge per-shard JSON metric files into one aggregate dict."""
    totals: Dict[str, float] = {}
    for p in shard_paths:
        if not Path(p).is_file():
            print(f"[merge] WARNING: shard file missing: {p}")
            return None
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for key in (
            "exec_pass_count", "parse_fail_count", "tool_acc_sum",
            "partial_sum", "turns_sum", "clipped_gens", "total_gens", "total_tasks",
        ):
            totals[key] = totals.get(key, 0.0) + d.get(key, 0.0)

    n = totals.get("total_tasks", 0)
    if n == 0:
        return None

    merged: Dict = {
        "total_tasks":      int(n),
        "exec_pass_rate":   totals["exec_pass_count"] / n,
        "tool_call_acc":    totals["tool_acc_sum"] / n,
        "partial_score":    totals["partial_sum"] / n,
        "parse_fail_rate":  totals["parse_fail_count"] / n,
        "avg_turns_completed": totals["turns_sum"] / n,
        "clipped_frac":     totals["clipped_gens"] / max(totals["total_gens"], 1),
        # raw counts for further merging
        "exec_pass_count":  totals["exec_pass_count"],
        "parse_fail_count": totals["parse_fail_count"],
        "tool_acc_sum":     totals["tool_acc_sum"],
        "turns_sum":        totals["turns_sum"],
        "clipped_gens":     totals["clipped_gens"],
        "total_gens":       totals["total_gens"],
    }
    return merged


# ─── main eval loop ───────────────────────────────────────────────────────────

def run_comparison(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config or str(SCRIPT_DIR / "configs" / "qwen3_4b_curriculum_v2.yaml")
    base_model = _load_base_model(config_path)
    print(f"[compare] base_model = {base_model}")

    # Parse eval GPUs
    eval_gpus = [int(g) for g in args.eval_gpus.split(",") if g.strip()] if args.eval_gpus else []
    num_shards = len(eval_gpus) if len(eval_gpus) > 1 else 1

    # Parse checkpoints:  "label:path"
    checkpoints: List[Tuple[str, Optional[str]]] = []
    for entry in args.checkpoints:
        if ":" in entry:
            label, path = entry.split(":", 1)
        else:
            label = Path(entry).name
            path = entry
        checkpoints.append((label.strip(), path.strip() if path.strip().lower() != "none" else None))

    n_calls_list: List[int] = sorted(set(args.n_calls))

    print(f"[compare] checkpoints : {[l for l,_ in checkpoints]}")
    print(f"[compare] n_calls     : {n_calls_list}")
    print(f"[compare] shards      : {num_shards}  gpus={eval_gpus}")
    print()

    # results[label][n_calls] = metrics dict
    results: Dict[str, Dict[int, Optional[Dict]]] = {}

    for label, adapter_path in checkpoints:
        results[label] = {}
        for n_calls in n_calls_list:
            slug = f"{label}_n{n_calls}"

            if num_shards > 1:
                # Launch shards in parallel
                procs: List[subprocess.Popen] = []
                shard_paths: List[str] = []
                t0 = time.time()
                for sid, gpu_id in enumerate(eval_gpus):
                    shard_json = str(output_dir / f"{slug}.shard{sid}.json")
                    shard_paths.append(shard_json)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cmd = [
                        sys.executable, "-u", str(EVAL_SCRIPT),
                        "--base_model",     base_model,
                        "--nestful_path",   args.nestful_path,
                        "--call_dist_path", args.call_dist_path,
                        "--n_calls",        str(n_calls),
                        "--max_new_tokens", str(args.max_new_tokens),
                        "--batch_size",     str(args.batch_size),
                        "--save_failures",  "10",
                        "--output_json",    shard_json,
                        "--shard_id",       str(sid),
                        "--num_shards",     str(num_shards),
                    ]
                    if adapter_path:
                        cmd += ["--adapter_path", adapter_path]
                    else:
                        cmd += ["--no_adapter"]
                    if args.max_tasks is not None:
                        cmd += ["--max_tasks", str(args.max_tasks)]
                    print(f"[compare] launching shard {sid}  {slug}  gpu={gpu_id}")
                    procs.append(subprocess.Popen(cmd, env=env))

                exit_codes = [p.wait() for p in procs]
                elapsed = time.time() - t0
                ok = all(rc == 0 for rc in exit_codes)
                print(f"[compare] {slug}  {'OK' if ok else 'SOME SHARDS FAILED'}  {elapsed:.0f}s")

                if ok:
                    merged = _merge_shards(shard_paths)
                    if merged:
                        merged_path = output_dir / f"{slug}.json"
                        merged_path.write_text(
                            json.dumps(merged, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    results[label][n_calls] = merged
                else:
                    results[label][n_calls] = None

            else:
                # Single GPU
                gpu_list = eval_gpus if eval_gpus else []
                out_json = str(output_dir / f"{slug}.json")
                rc = _run_eval(
                    label=label,
                    adapter_path=adapter_path,
                    n_calls=n_calls,
                    nestful_path=args.nestful_path,
                    call_dist_path=args.call_dist_path,
                    output_json=out_json,
                    base_model=base_model,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    max_tasks=args.max_tasks,
                    gpu_ids=gpu_list,
                    num_shards=1,
                    shard_id=0,
                )
                if rc == 0 and Path(out_json).is_file():
                    results[label][n_calls] = json.loads(
                        Path(out_json).read_text(encoding="utf-8")
                    )
                else:
                    results[label][n_calls] = None

    # ── Save full results JSON ────────────────────────────────────────────────
    summary_path = output_dir / "comparison_summary.json"
    summary: Dict = {
        "base_model": base_model,
        "n_calls": n_calls_list,
        "checkpoints": [{"label": l, "adapter": a or "none"} for l, a in checkpoints],
        "results": {
            label: {
                str(n): (m if m is not None else {"error": "eval_failed"})
                for n, m in nc_map.items()
            }
            for label, nc_map in results.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[compare] full results → {summary_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    _print_table(results, n_calls_list, checkpoints)


def _print_table(
    results: Dict[str, Dict[int, Optional[Dict]]],
    n_calls_list: List[int],
    checkpoints: List[Tuple[str, Optional[str]]],
) -> None:
    labels = [l for l, _ in checkpoints]
    col_w = max(14, max(len(l) for l in labels) + 2)

    def _fmt(m: Optional[Dict], key: str, pct: bool = True) -> str:
        if m is None:
            return "  —  "
        v = m.get(key)
        if v is None:
            return "  —  "
        return f"{100*v:5.1f}%" if pct else f"{v:.2f}"

    sep = "─" * (12 + col_w * len(labels) + 3 * len(labels))

    print()
    print("=" * len(sep))
    print("  COMPARISON TABLE — exec_pass_rate")
    print("=" * len(sep))

    # header
    hdr = f"{'n_calls':>10}  "
    for label in labels:
        hdr += f"{label:>{col_w}}  "
    print(hdr)
    print(sep)

    for n in n_calls_list:
        row = f"n_calls={n:>2}  "
        for label in labels:
            m = results.get(label, {}).get(n)
            row += f"{_fmt(m, 'exec_pass_rate'):>{col_w}}  "
        print(row)

    print(sep)
    print()
    print("  TOOL ACCURACY")
    print(sep)

    for n in n_calls_list:
        row = f"n_calls={n:>2}  "
        for label in labels:
            m = results.get(label, {}).get(n)
            row += f"{_fmt(m, 'tool_call_acc'):>{col_w}}  "
        print(row)

    print(sep)
    print()
    print("  PARSE FAIL RATE")
    print(sep)

    for n in n_calls_list:
        row = f"n_calls={n:>2}  "
        for label in labels:
            m = results.get(label, {}).get(n)
            row += f"{_fmt(m, 'parse_fail_rate'):>{col_w}}  "
        print(row)

    print(sep)
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare NESTFUL performance across training checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        metavar="LABEL:PATH",
        help=(
            'Checkpoint(s) to compare. Format: "label:adapter_path". '
            'Use "none" as path for the bare base model.'
        ),
    )
    p.add_argument(
        "--n_calls", nargs="+", type=int, default=[2, 3, 4],
        help="Which NESTFUL n_calls groups to evaluate (default: 2 3 4).",
    )
    p.add_argument(
        "--nestful_path", required=True,
        help="Path to nestful_data.jsonl.",
    )
    p.add_argument(
        "--call_dist_path", required=True,
        help="Path to nestful_call_distribution.json.",
    )
    p.add_argument(
        "--output_dir", default="curricullum/training/results/compare",
        help="Directory to write per-run JSON results and the summary.",
    )
    p.add_argument(
        "--max_tasks", type=int, default=None,
        help="Limit tasks per n_calls group (default: all). 150 is fast for quick checks.",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max generation tokens per turn (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    p.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for model.generate() (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--eval_gpus", default=None,
        help=(
            "Comma-separated physical GPU IDs for sharded evaluation. "
            "E.g. '1,2' launches 2 shards in parallel. Default: use whatever GPU is visible."
        ),
    )
    p.add_argument(
        "--config", default=None,
        help="Path to qwen3_4b_curriculum_v2.yaml (to read base_model). Optional.",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_comparison(args)
