#!/usr/bin/env python3
"""
Re-verify existing raw_toolr0 candidates, select up to N_FINAL per epoch,
then top-up generation for epochs still below target.

Usage:
  python curricullum/run_revalidate_and_topup.py
  python curricullum/run_revalidate_and_topup.py --epochs 3,4,5,6 --skip-topup
  python curricullum/run_revalidate_and_topup.py --target 400 --max-stages 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curricullum.run_generate_toolr0_curriculum import (  # noqa: E402
    DEFAULT_OPENROUTER_MODEL,
    EPOCH_CONFIGS,
    epoch_cfg,
    load_dotenv,
    resolve_nestful,
    resolve_openrouter_model,
)

RAW_DIR = ROOT / "curricullum" / "data" / "raw_toolr0"
VERIFIED_DIR = ROOT / "curricullum" / "data" / "verified_toolr0"
REJECTED_DIR = ROOT / "curricullum" / "data" / "rejected_toolr0"
FILTERED_DIR = ROOT / "curricullum" / "data" / "filtered_toolr0_synthetic"
NESTFUL = "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl"

# Fallback verify yield if no raw file yet
DEFAULT_YIELD = {1: 0.75, 2: 0.54, 3: 0.21, 4: 0.11, 5: 0.06, 6: 0.08, 7: 0.05}


def run(cmd: List[str]) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def count_json_list(path: Path) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else 0


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.open(encoding="utf-8") if ln.strip())


def verify_epoch(epoch: int, cfg: Dict[str, Any], exec_workers: int = 8, exec_timeout: float = 20.0) -> int:
    raw = RAW_DIR / f"epoch_{epoch}_candidates.json"
    if not raw.is_file():
        print(f"[skip] epoch {epoch}: no raw file", flush=True)
        return 0

    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "-u", "curricullum/data/step2_verify_candidates.py",
        "--in_json", str(raw),
        "--out_json", str(VERIFIED_DIR / f"epoch_{epoch}_verified.json"),
        "--rejected_json", str(REJECTED_DIR / f"epoch_{epoch}_rejected.json"),
        "--nestful_path", NESTFUL,
        "--epoch", str(epoch),
        "--dependency_mode", cfg["dep_mode"],
        "--max_tool_menu", str(cfg["tool_menu_max"]),
        "--use_executor",
        "--exec_workers", str(exec_workers),
        "--exec_timeout", str(exec_timeout),
    ])
    return count_json_list(VERIFIED_DIR / f"epoch_{epoch}_verified.json")


def select_epoch(epoch: int, n_final: int) -> int:
    verified = VERIFIED_DIR / f"epoch_{epoch}_verified.json"
    if not verified.is_file():
        return 0
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    out = FILTERED_DIR / f"epoch_{epoch}_{epoch}call.jsonl"
    run([
        sys.executable, "-u", "curricullum/data/step3_select_curriculum.py",
        "--in_json", str(verified),
        "--out_jsonl", str(out),
        "--n_final", str(n_final),
        "--epoch", str(epoch),
        "--seed", os.environ.get("SEED", "42"),
    ])
    return count_jsonl(out)


def merge_candidates(epoch: int, new_path: Path) -> int:
    """Append unique candidates from new_path into epoch raw json."""
    main = RAW_DIR / f"epoch_{epoch}_candidates.json"
    existing: List[Dict[str, Any]] = []
    if main.is_file():
        existing = json.loads(main.read_text(encoding="utf-8"))
    if not isinstance(existing, list):
        existing = []

    new_items = json.loads(new_path.read_text(encoding="utf-8"))
    if not isinstance(new_items, list):
        return len(existing)

    seen = set()
    for item in existing:
        meta = item.get("meta") or {}
        cid = meta.get("candidate_id") or item.get("input", "")
        seen.add(cid)

    added = 0
    for item in new_items:
        meta = item.get("meta") or {}
        cid = meta.get("candidate_id") or item.get("input", "")
        if cid in seen:
            continue
        seen.add(cid)
        existing.append(item)
        added += 1

    main.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] epoch {epoch}: +{added} candidates -> {len(existing)} total", flush=True)
    return len(existing)


def generate_topup(
    epoch: int,
    n_generate: int,
    cfg: Dict[str, Any],
    model: str,
    workers: int,
    n_seed_examples: int = 0,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_out = RAW_DIR / f"epoch_{epoch}_topup_candidates.json"
    max_gen = int(n_generate * 1.25) + 50

    # Use current raw count as part of seed so every top-up call generates
    # different candidates (avoids deduplication killing yield in later rounds).
    base_seed = int(os.environ.get("SEED", "42"))
    current_raw = count_json_list(RAW_DIR / f"epoch_{epoch}_candidates.json")
    dynamic_seed = base_seed + epoch * 17 + current_raw

    run([
        sys.executable, "-u", "curricullum/data/step1_gen_candidates.py",
        "--nestful_path", NESTFUL,
        "--out_json", str(tmp_out),
        "--epoch", str(epoch),
        "--n_generate", str(n_generate),
        "--max_generate", str(max_gen),
        "--batch_size", str(cfg["batch_size"]),
        "--parallel_workers", str(workers),
        "--max_tokens", str(cfg["max_tokens"]),
        "--tool_menu_max", str(cfg["tool_menu_max"]),
        "--model", model,
        "--seed", str(dynamic_seed),
        "--seed_mode", os.environ.get("SEED_MODE", "schema_only"),
        "--dependency_mode", cfg["dep_mode"],
        "--n_seed_examples", str(n_seed_examples),
    ])
    merge_candidates(epoch, tmp_out)
    if tmp_out.is_file():
        tmp_out.unlink()


def estimate_topup_calls(epoch: int, raw_n: int, verified_n: int, target: int) -> int:
    if verified_n >= target:
        return 0
    gap = target - verified_n
    if raw_n > 0 and verified_n > 0:
        yield_rate = max(0.05, verified_n / raw_n)
    else:
        yield_rate = DEFAULT_YIELD.get(epoch, 0.1)
    # Generate enough new raw candidates to close the gap (+30% margin)
    need = int(gap / yield_rate * 1.3) + 20
    return max(need, 50)


def rebuild_all_jsonl(stages: List[int]) -> None:
    all_path = FILTERED_DIR / "curriculum_toolr0_all.jsonl"
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", encoding="utf-8") as out:
        for e in stages:
            p = FILTERED_DIR / f"epoch_{e}_{e}call.jsonl"
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                if text.strip():
                    out.write(text)
                    if not text.endswith("\n"):
                        out.write("\n")
    run([sys.executable, "-u", "curricullum/data/inspect_dataset.py", "--path", str(all_path)])


def print_status(stages: List[int], target: int) -> None:
    print("\n--- Status ---")
    print(f"{'Ep':>3} {'raw':>6} {'verified':>9} {'filtered':>9} {'target':>7}")
    for e in stages:
        raw_n = count_json_list(RAW_DIR / f"epoch_{e}_candidates.json")
        ver_n = count_json_list(VERIFIED_DIR / f"epoch_{e}_verified.json")
        fil_n = count_jsonl(FILTERED_DIR / f"epoch_{e}_{e}call.jsonl")
        print(f"{e:>3} {raw_n:>6} {ver_n:>9} {fil_n:>9} {target:>7}")
    print()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Re-verify raw_toolr0 and top-up to target")
    ap.add_argument("--target", type=int, default=int(os.environ.get("N_FINAL", "400")))
    ap.add_argument("--max-stages", type=int, default=int(os.environ.get("MAX_STAGES", "6")))
    ap.add_argument("--epochs", default="", help="Comma list override, e.g. 3,4,5,6")
    ap.add_argument("--skip-topup", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("PARALLEL_WORKERS", "16")))
    ap.add_argument(
        "--exec-workers", type=int, default=8,
        help="Parallel IBM executor threads per epoch (default 8)",
    )
    ap.add_argument(
        "--exec-timeout", type=float, default=20.0,
        help="Hard timeout per IBM exec call in seconds (default 20)",
    )
    ap.add_argument(
        "--n-seed-examples", type=int, default=0,
        help="Number of verified seed examples to pass to generation prompt (default 0)",
    )
    args = ap.parse_args()

    if args.epochs.strip():
        stages = [int(x.strip()) for x in args.epochs.split(",") if x.strip()]
    else:
        stages = list(range(1, min(args.max_stages, 7) + 1))

    model = resolve_openrouter_model(os.environ.get("MODEL", DEFAULT_OPENROUTER_MODEL))
    target = args.target

    print("=" * 60)
    print(f"Revalidate + top-up  stages={stages}  target={target}  model={model}")
    print("=" * 60)

    # Phase 1: verify + select existing raw
    print("\n[phase 1] Re-verify existing raw_toolr0 ...", flush=True)
    for e in stages:
        cfg = epoch_cfg(e, target)
        raw_n = count_json_list(RAW_DIR / f"epoch_{e}_candidates.json")
        if raw_n == 0:
            print(f"[phase 1] epoch {e}: no raw, skip", flush=True)
            continue
        ver_n = verify_epoch(e, cfg, exec_workers=args.exec_workers, exec_timeout=args.exec_timeout)
        fil_n = select_epoch(e, target)
        print(f"[phase 1] epoch {e}: raw={raw_n} verified={ver_n} filtered={fil_n}", flush=True)

    print_status(stages, target)

    if args.skip_topup:
        rebuild_all_jsonl(stages)
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[warn] OPENROUTER_API_KEY not set — skipping top-up generation", flush=True)
        rebuild_all_jsonl(stages)
        return

    # Phase 2: top-up epochs below target
    print("\n[phase 2] Top-up generation for epochs below target ...", flush=True)
    for e in stages:
        cfg = epoch_cfg(e, target)
        raw_n = count_json_list(RAW_DIR / f"epoch_{e}_candidates.json")
        ver_n = count_json_list(VERIFIED_DIR / f"epoch_{e}_verified.json")
        if ver_n >= target:
            print(f"[phase 2] epoch {e}: already {ver_n} verified >= {target}, skip", flush=True)
            continue

        n_gen = estimate_topup_calls(e, raw_n, ver_n, target)
        print(f"[phase 2] epoch {e}: verified={ver_n}/{target}, generating ~{n_gen} more ...", flush=True)
        try:
            generate_topup(e, n_gen, cfg, model, args.workers, n_seed_examples=args.n_seed_examples)
        except subprocess.CalledProcessError as exc:
            print(f"[err] epoch {e} top-up generation failed: {exc}", flush=True)
            continue

        ver_n = verify_epoch(e, cfg, exec_workers=args.exec_workers, exec_timeout=args.exec_timeout)
        fil_n = select_epoch(e, target)
        print(f"[phase 2] epoch {e} after top-up: verified={ver_n} filtered={fil_n}", flush=True)

    rebuild_all_jsonl(stages)
    print_status(stages, target)
    print("Done.")


if __name__ == "__main__":
    main()
