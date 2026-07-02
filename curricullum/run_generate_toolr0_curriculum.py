#!/usr/bin/env python3
"""
Generate synthetic Tool-R0 curriculum via OpenRouter (native Python, Windows-friendly).

Stages 1-7 (configurable via MAX_STAGES).
Each stage targets 400 verified final tasks; generation volume scales to compensate for
the exponentially declining yield rate as call-chain depth increases.

Stages 1-5 use strict_chain verification (call i must reference call i-1).
Stages 6-7 use dag_chain verification (each call must reference ≥1 prior, depth ≥ ceil(n*0.6)).

Usage:
  # Windows:
  set OPENROUTER_API_KEY=sk-or-...
  python curricullum/run_generate_toolr0_curriculum.py

  # Linux/DGX:
  export OPENROUTER_API_KEY=sk-or-...
  python curricullum/run_generate_toolr0_curriculum.py

Env overrides (all optional):
  MODEL              OpenRouter slug (default: deepseek/deepseek-v4-flash)
                       HF ID deepseek-ai/DeepSeek-V4-Flash is auto-mapped to this slug
  MAX_STAGES         How many stages to generate, 1-7 (default: 6)
  N_FINAL            Final verified samples per stage (default: 400)
  PARALLEL_WORKERS   Concurrent OpenRouter requests (default: 16)
  USE_EXECUTOR       Set to 0 to skip IBM execution verification (not recommended)
  DATA_DIR           Output directory for filtered JSONL files
  SEED               Random seed (default: 42)
  SEED_MODE          schema_only | fewshot | fewshot_debug (default: schema_only)

Per-epoch overrides (e.g. for epoch 4):
  N_GENERATE_4       Override n_generate for epoch 4
  N_FINAL_4          Override n_final for epoch 4
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4-flash": DEFAULT_OPENROUTER_MODEL,
    "deepseek-ai/DeepSeek-V4-Flash": DEFAULT_OPENROUTER_MODEL,
}


def resolve_openrouter_model(model: str) -> str:
    key = model.strip()
    mapped = OPENROUTER_MODEL_ALIASES.get(key, key)
    if mapped != key:
        print(f"[pipeline] OpenRouter model alias: {key} -> {mapped}", flush=True)
    return mapped

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Per-epoch generation config
# ---------------------------------------------------------------------------
# Yield rates (observed/extrapolated): 1→75%, 2→54%, 3→21%, 4→11%, 5→6%, 6→8%(dag), 7→5%(dag)
# N_GENERATE is set so that N_GENERATE * yield ≥ N_FINAL * 1.25 (25% safety margin).
# max_generate = N_GENERATE + 20% headroom to avoid stopping short on unlucky runs.
# dag_chain used for epochs 6-7: more flexible DAG deps → higher yield than strict_chain.
#
# batch_size: candidates requested per OpenRouter batch (more=fewer round-trips, less=lower latency)
# max_tokens: OpenRouter completion cap; scales with output length per call (~180 tokens/call)
# tool_menu_max: max tools in prompt menu; higher→harder selection→more GRPO signal
EPOCH_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "n_generate":    900,
        "max_generate": 1100,
        "n_final":       400,
        "batch_size":     14,
        "max_tokens":   1200,
        "tool_menu_max":   6,
        "dep_mode":    "strict_chain",
    },
    2: {
        "n_generate":   2000,
        "max_generate": 2400,
        "n_final":       400,
        "batch_size":     12,
        "max_tokens":   1800,
        "tool_menu_max":   7,
        "dep_mode":    "strict_chain",
    },
    3: {
        "n_generate":   3500,
        "max_generate": 4200,
        "n_final":       400,
        "batch_size":     10,
        "max_tokens":   2400,
        "tool_menu_max":   7,
        "dep_mode":    "strict_chain",
    },
    4: {
        "n_generate":   6000,
        "max_generate": 7200,
        "n_final":       400,
        "batch_size":      8,
        "max_tokens":   2800,
        "tool_menu_max":   8,
        "dep_mode":    "strict_chain",
    },
    5: {
        "n_generate":  10000,
        "max_generate": 12000,
        "n_final":       400,
        "batch_size":      7,
        "max_tokens":   3200,
        "tool_menu_max":   8,
        "dep_mode":    "strict_chain",
    },
    6: {
        "n_generate":  14000,
        "max_generate": 17000,
        "n_final":       400,
        "batch_size":      6,
        "max_tokens":   3600,
        "tool_menu_max":   9,
        "dep_mode":    "dag_chain",
    },
    7: {
        "n_generate":  18000,
        "max_generate": 22000,
        "n_final":       400,
        "batch_size":      5,
        "max_tokens":   4096,
        "tool_menu_max":   9,
        "dep_mode":    "dag_chain",
    },
}


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file() or os.environ.get("OPENROUTER_API_KEY"):
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def epoch_cfg(epoch: int, n_final_global: int) -> Dict[str, Any]:
    """Return resolved config for this epoch, applying env overrides."""
    base = dict(EPOCH_CONFIGS[epoch])
    # Per-epoch env overrides
    if os.environ.get(f"N_GENERATE_{epoch}"):
        base["n_generate"] = int(os.environ[f"N_GENERATE_{epoch}"])
    if os.environ.get(f"N_FINAL_{epoch}"):
        base["n_final"] = int(os.environ[f"N_FINAL_{epoch}"])
    elif os.environ.get("N_FINAL"):
        base["n_final"] = n_final_global
    # Recompute max_generate whenever n_generate is overridden
    if base["max_generate"] < base["n_generate"] * 1.15:
        base["max_generate"] = int(base["n_generate"] * 1.2)
    return base


def step1(epoch: int, nestful: str, model: str, workers: int, cfg: Dict[str, Any]) -> None:
    raw_dir = ROOT / "curricullum" / "data" / "raw_toolr0"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-u", "curricullum/data/step1_gen_candidates.py",
        "--nestful_path", nestful,
        "--out_json", str(raw_dir / f"epoch_{epoch}_candidates.json"),
        "--epoch", str(epoch),
        "--n_generate",   str(cfg["n_generate"]),
        "--max_generate", str(cfg["max_generate"]),
        "--batch_size",   str(cfg["batch_size"]),
        "--parallel_workers", str(workers),
        "--max_tokens",   str(cfg["max_tokens"]),
        "--tool_menu_max", str(cfg["tool_menu_max"]),
        "--model", model,
        "--seed", os.environ.get("SEED", "42"),
        "--seed_mode", os.environ.get("SEED_MODE", "schema_only"),
        "--dependency_mode", cfg["dep_mode"],
        "--n_seed_examples", "0",
    ])


def step2_step3(
    epoch: int,
    nestful: str,
    use_executor: bool,
    data_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    raw = ROOT / "curricullum" / "data" / "raw_toolr0" / f"epoch_{epoch}_candidates.json"
    verified_dir = ROOT / "curricullum" / "data" / "verified_toolr0"
    rejected_dir = ROOT / "curricullum" / "data" / "rejected_toolr0"
    for d in (verified_dir, rejected_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    cmd2 = [
        sys.executable, "-u", "curricullum/data/step2_verify_candidates.py",
        "--in_json",      str(raw),
        "--out_json",     str(verified_dir / f"epoch_{epoch}_verified.json"),
        "--rejected_json", str(rejected_dir / f"epoch_{epoch}_rejected.json"),
        "--nestful_path", nestful,
        "--epoch",        str(epoch),
        "--dependency_mode", cfg["dep_mode"],
        "--max_tool_menu", str(cfg["tool_menu_max"]),
    ]
    if use_executor:
        cmd2.append("--use_executor")
    else:
        cmd2.append("--no_executor")
    run(cmd2)

    run([
        sys.executable, "-u", "curricullum/data/step3_select_curriculum.py",
        "--in_json",  str(verified_dir / f"epoch_{epoch}_verified.json"),
        "--out_jsonl", str(data_dir / f"epoch_{epoch}_{epoch}call.jsonl"),
        "--n_final",  str(cfg["n_final"]),
        "--epoch",    str(epoch),
        "--seed",     os.environ.get("SEED", "42"),
    ])


def resolve_nestful() -> str:
    if os.environ.get("NESTFUL_PATH"):
        return os.environ["NESTFUL_PATH"]
    for p in (
        "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl",
        "data_v2/nestful_data.jsonl",
    ):
        if (ROOT / p).is_file():
            return p
    raise FileNotFoundError("NESTFUL data not found. Set NESTFUL_PATH env var.")


def print_plan(stages: list[int], model: str, n_final_global: int, workers: int) -> None:
    print("=" * 70)
    print(f"  Tool-R0 synthetic curriculum (MAX_STAGES={max(stages)})")
    print(f"  model={model}  workers={workers}  n_final_default={n_final_global}")
    print("-" * 70)
    print(f"  {'Epoch':>5}  {'Calls':>5}  {'N_gen':>7}  {'N_final':>7}  {'Mode':<14}  {'batch':>5}  {'tok':>5}")
    for e in stages:
        cfg = epoch_cfg(e, n_final_global)
        print(
            f"  {e:>5}  {e:>5}  {cfg['n_generate']:>7}  {cfg['n_final']:>7}"
            f"  {cfg['dep_mode']:<14}  {cfg['batch_size']:>5}  {cfg['max_tokens']:>5}"
        )
    print("=" * 70)


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[err] OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    nestful = resolve_nestful()
    model = resolve_openrouter_model(os.environ.get("MODEL", DEFAULT_OPENROUTER_MODEL))
    max_stages = min(int(os.environ.get("MAX_STAGES", "6")), 7)
    n_final_global = int(os.environ.get("N_FINAL", "400"))
    workers = int(os.environ.get("PARALLEL_WORKERS", "16"))
    use_executor = os.environ.get("USE_EXECUTOR", "1") != "0"
    data_dir = ROOT / os.environ.get("DATA_DIR", "curricullum/data/filtered_toolr0_synthetic")

    stages = list(range(1, max_stages + 1))
    print_plan(stages, model, n_final_global, workers)

    # Step 1: generate candidates for all epochs in parallel
    print("\n[pipeline] Step 1: generating candidates (parallel) ...", flush=True)
    with ThreadPoolExecutor(max_workers=len(stages)) as pool:
        futs = {
            pool.submit(step1, e, nestful, model, workers, epoch_cfg(e, n_final_global)): e
            for e in stages
        }
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                fut.result()
                print(f"[pipeline] epoch {e} step1 done", flush=True)
            except Exception as exc:
                print(f"[pipeline] epoch {e} step1 FAILED: {exc}", file=sys.stderr)
                raise

    # Step 2+3: verify and select in parallel
    print("\n[pipeline] Step 2+3: verify + select (parallel) ...", flush=True)
    with ThreadPoolExecutor(max_workers=len(stages)) as pool:
        futs = {
            pool.submit(
                step2_step3, e, nestful, use_executor, data_dir, epoch_cfg(e, n_final_global)
            ): e
            for e in stages
        }
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                fut.result()
                print(f"[pipeline] epoch {e} step2+3 done", flush=True)
            except Exception as exc:
                print(f"[pipeline] epoch {e} step2+3 FAILED: {exc}", file=sys.stderr)
                raise

    # Concatenate all stages into one combined file
    all_path = data_dir / "curriculum_toolr0_all.jsonl"
    parts = sorted(data_dir.glob("epoch_*_*call.jsonl"))
    with all_path.open("w", encoding="utf-8") as out:
        for p in parts:
            text = p.read_text(encoding="utf-8")
            if text.strip():
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")

    print(f"\n[pipeline] Combined dataset -> {all_path}")
    run([sys.executable, "-u", "curricullum/data/inspect_dataset.py", "--path", str(all_path)])

    # Summary
    print("\n[pipeline] Per-stage counts:")
    for p in parts:
        n = sum(1 for ln in p.open(encoding="utf-8") if ln.strip())
        print(f"  {p.name}: {n} tasks")

    print(f"\nDone -> {data_dir}")
    print("Next: TRAINING_FORMAT=tool_r0 bash curricullum/run_train_toolr0.sh")


if __name__ == "__main__":
    main()
