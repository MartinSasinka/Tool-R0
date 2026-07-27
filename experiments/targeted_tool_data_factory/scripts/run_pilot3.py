#!/usr/bin/env python3
"""Orchestrate pilot3 — seeded from frozen pilot2 (no pilot2 mutation).

Why this is fast enough:
  Pilot2 rows are copied into the pilot3 pool and skip V4 minimal-path search.
  Only newly generated candidates (~3500) pay the expensive validate cost.

Flow:
  1. seed_pilot3_from_pilot2
  2. generate APPEND (new candidates only)
  3. validate (seed rows skip V4; new rows fully validated)
  4. select → paraphrase → reselect → probe → split → export → report
  5. preflight / docs / bundle

Usage:
  python scripts/run_pilot3.py --dry-run
  python scripts/run_pilot3.py --no-llm
  python scripts/run_pilot3.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

FACTORY = Path(__file__).resolve().parents[1]
SRC = FACTORY / "src"
REPO = FACTORY.parents[1]
CONFIG = "configs/pilot3_local.yaml"
VERSION = "pilot3"
SEED = 20260727
ADAPTATION_RATIO = 0.55
NEW_CANDIDATES = 3500   # on top of seeded pilot2 validated pool


def _load_dotenv() -> None:
    env_path = REPO / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if rc.returncode != 0:
        raise SystemExit(rc.returncode)


def _cli(stage: str, extra: list[str], dry_run: bool, no_llm: bool) -> list[str]:
    cmd = [
        sys.executable, "-X", "utf8", "-u", "-m", "targeted_tool_data.cli", stage,
        "--config", CONFIG,
        "--version", VERSION,
        "--seed", str(SEED),
        "--target", "nestful",
        "--tracks", "adaptation,generalization",
        "--adaptation-ratio", str(ADAPTATION_RATIO),
        "--engine", "v2",
        "--overwrite",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if no_llm or dry_run:
        cmd.append("--no-llm")
    cmd.extend(extra)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--skip-seed", action="store_true",
                    help="do not re-copy pilot2 (resume after seed)")
    ap.add_argument("--new-candidates", type=int, default=NEW_CANDIDATES)
    ap.add_argument("--from-stage", default=None,
                    choices=["seed", "generate", "validate", "select",
                             "paraphrase", "probe", "split", "export",
                             "report", "preflight", "docs", "bundle"])
    ap.add_argument("--skip-bundle", action="store_true")
    args = ap.parse_args()
    _load_dotenv()

    t0 = time.perf_counter()
    print(f"[pilot3] seed={SEED} G={1 - ADAPTATION_RATIO:.0%} "
          f"new_candidates={args.new_candidates} "
          f"dry_run={args.dry_run} no_llm={args.no_llm}", flush=True)

    stages = ["seed", "generate", "validate", "select", "paraphrase",
              "probe", "split", "export", "report", "preflight", "docs", "bundle"]
    if args.from_stage:
        stages = stages[stages.index(args.from_stage):]
    if args.skip_seed and "seed" in stages:
        stages = [s for s in stages if s != "seed"]

    for stage in stages:
        if stage == "seed":
            if args.dry_run:
                print("[pilot3] seed dry-run: would copy pilot2 validated -> pilot3")
            else:
                run([sys.executable, "-u",
                     str(FACTORY / "scripts" / "seed_pilot3_from_pilot2.py")],
                    FACTORY)
        elif stage == "generate":
            # Append NEW candidates on top of the seeded pool.
            n = 10 if args.dry_run else args.new_candidates
            # Use a small Python driver so we can pass append/start_index.
            driver = f"""
import argparse, sys
sys.path.insert(0, r'{SRC}')
from targeted_tool_data.cli import Ctx, step_generate
a = argparse.Namespace(
    config=r'{CONFIG}', target='nestful',
    tracks='adaptation,generalization', adaptation_ratio={ADAPTATION_RATIO},
    seed={SEED}, version='{VERSION}', candidates={n}, max_candidates=None,
    resume=False, overwrite=True, dry_run={bool(args.dry_run)},
    strict=False, no_llm=True, no_docs=False, engine='v2',
    provider=None, base_url=None, model=None)
ctx = Ctx(a)
step_generate(ctx, n_candidates={n}, start_index=10**6, append=True)
"""
            run([sys.executable, "-X", "utf8", "-u", "-c", driver], FACTORY)
        elif stage == "validate":
            run(_cli("validate", [], args.dry_run, True), SRC)
        elif stage == "select":
            run(_cli("select", [], args.dry_run, True), SRC)
        elif stage == "paraphrase":
            if args.no_llm or args.dry_run:
                print("[pilot3] paraphrase skipped")
            else:
                if not os.environ.get("OPENROUTER_API_KEY"):
                    print("[pilot3] ABORT: OPENROUTER_API_KEY missing", file=sys.stderr)
                    return 2
                run(_cli("paraphrase", [], False, False), SRC)
                run(_cli("select", [], False, True), SRC)  # reselect
        elif stage == "probe":
            run(_cli("probe", [], args.dry_run, True), SRC)
        elif stage == "split":
            run(_cli("split", [], args.dry_run, True), SRC)
        elif stage == "export":
            run(_cli("export", [], args.dry_run, True), SRC)
        elif stage == "report":
            run(_cli("report", [], args.dry_run, True), SRC)
        elif stage == "preflight":
            if args.dry_run:
                print("[pilot3] preflight skipped (dry-run)")
                continue
            export = FACTORY / "outputs" / "selected" / "export_pilot3"
            run([
                sys.executable, "-X", "utf8", "-u",
                str(FACTORY / "trainer_adapter" / "preflight_gold_replay.py"),
                "--data", str(export / "train_grpo_pilot3.jsonl"), "--expect", "600",
                "--data", str(export / "heldout_grpo_pilot3.jsonl"), "--expect", "200",
                "--data", str(export / "reserve_grpo_pilot3.jsonl"), "--expect", "200",
                "--report", str(FACTORY / "outputs" / "reports" / "preflight_pilot3.json"),
            ], FACTORY)
        elif stage == "docs":
            if args.dry_run:
                print("[pilot3] docs skipped")
                continue
            run([sys.executable, "-u",
                 str(FACTORY / "scripts" / "make_pilot3_docs.py")], FACTORY)
        elif stage == "bundle":
            if args.dry_run or args.skip_bundle:
                print("[pilot3] bundle skipped")
                continue
            run([sys.executable, "-u",
                 str(FACTORY / "runpod_bundle_pilot3" / "build_bundle.py")],
                FACTORY)

    print(f"[pilot3] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
