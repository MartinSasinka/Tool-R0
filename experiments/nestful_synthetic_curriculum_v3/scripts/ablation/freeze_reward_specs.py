#!/usr/bin/env python3
"""Freeze the reward-ablation specs (reports/reward_ablation/ABLATION_PLAN.md §6).

Runs, in order:
  1. the reward-ablation unit test suite (tests/test_reward_ablation.py);
  2. the 16x8 reward probe (reuses the existing probe tool; --backend stub
     locally, --backend vllm on RunPod);
  3. terminal-invariant + epsilon-band-safety verification (imported
     straight from lib/reward_ablation_registry.py, which asserts these at
     import time already — re-checked here explicitly and recorded);
  4. writes reports/reward_ablation/FROZEN_REWARD_SPECS.json with the exact
     formulas/scalars/epsilons/component weights, a hash of the registry
     source files, and the current git commit.

After this file is written, `terminal bands / epsilon / process weights /
gates` must NOT change without a new reward ID + new ablation version
(reports/reward_ablation/ABLATION_PLAN.md §6).

Usage (repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/freeze_reward_specs.py --backend stub
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
for p in (str(_V3),):
    if p not in sys.path:
        sys.path.insert(0, p)

REPORTS_DIR = _V3 / "reports" / "reward_ablation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO),
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN ({exc})"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(_REPO),
                              capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return True


def run_unit_tests() -> Dict[str, Any]:
    test_path = _V3 / "tests" / "test_reward_ablation.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q"],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    ok = result.returncode == 0
    if not ok:
        print(result.stdout[-4000:])
        print(result.stderr[-2000:], file=sys.stderr)
    return {"ok": ok, "returncode": result.returncode, "tail": result.stdout[-2000:]}


def run_probe(backend: str, checkpoint: str) -> Dict[str, Any]:
    probe_script = _HERE / "reward_probe_16x8.py"
    argv = [sys.executable, str(probe_script), "--backend", backend]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    result = subprocess.run(argv, cwd=str(_REPO), capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit("[freeze_reward_specs] ABORT: reward probe failed")
    # find the freshest summary file written by the probe
    probes_dir = _V3 / "outputs" / "probes"
    candidates = sorted(probes_dir.glob(f"reward_ablation_probe_{backend}_*/REWARD_PROBE_SUMMARY.json"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("[freeze_reward_specs] ABORT: no REWARD_PROBE_SUMMARY.json found")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "vllm", "hf"], default="stub")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    if args.backend == "stub":
        print("[freeze_reward_specs] WARNING: backend=stub. This freeze file will be "
              "STRUCTURALLY valid (formulas/scalars/hashes/tests) but the probe evidence "
              "is a CPU pipeline self-test, not a real on-policy GPU calibration. Re-run "
              "with --backend vllm on RunPod before trusting Round 1 GPU time to this freeze.")

    test_result = {"skipped": True} if args.skip_tests else run_unit_tests()
    if not args.skip_tests and not test_result["ok"]:
        raise SystemExit("[freeze_reward_specs] ABORT: reward-ablation unit tests failed")

    probe_result = {"skipped": True} if args.skip_probe else run_probe(args.backend, args.checkpoint)

    from lib import reward_ablation_registry as R

    invariants = {}
    for arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
        invariants[arm] = {
            "epsilon_band_safe": R.verify_epsilon_safety(arm),
            "min_adjacent_gap": R.min_adjacent_gap(R.TERMINAL_SCALARS[arm]),
            "epsilon": R.EPSILONS[arm],
        }
    all_safe = all(v["epsilon_band_safe"] for v in invariants.values())
    if not all_safe:
        raise SystemExit("[freeze_reward_specs] ABORT: epsilon-band-safety invariant violated")

    registry_files = [
        _V3 / "lib" / "reward_ablation_registry.py",
        _V3 / "lib" / "verifiable_process_reward.py",
        _V3 / "lib" / "reward_variants_offline.py",
        _V3 / "lib" / "reward_v3_2_dense.py",
        _V3 / "lib" / "reward_v3_1.py",
    ]
    file_hashes = {str(p.relative_to(_REPO)).replace("\\", "/"): _sha256_file(p) for p in registry_files}

    spec = {
        "frozen_at": _now(),
        "ablation_version": "v1",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "unit_tests": test_result,
        "probe": {
            "backend": args.backend,
            "is_real_calibration": args.backend != "stub",
            "summary": probe_result,
        },
        "terminal_taxonomy": list(R.TERMINAL_CLASSES),
        "arms": {
            arm: {
                "label": R.ARM_LABELS[arm],
                "terminal_scalars": R.TERMINAL_SCALARS.get(arm),
                "epsilon": R.EPSILONS.get(arm),
                "formula": (
                    "total_reward = execution_aware_v3_2_dense(trajectory, task) [UNCHANGED production reward]"
                    if arm == "A0_R0_CURRENT" else
                    "total_reward = terminal_scalar[unified_terminal_class] + epsilon * process_score"
                ),
                "process_score_source": {
                    "A1_OUTCOME_ONLY": "none (process_score := 0.0)",
                    "A2_R3_OUTCOME_FIRST": "lib.reward_variants_offline._process_score_no_length (gold-aware, audited R3 definition, reused verbatim)",
                    "A3_VERIFIABLE_PROCESS": "lib.verifiable_process_reward.verifiable_process_components (gold-FREE)",
                    "A4_GATED_VERIFIABLE": "lib.verifiable_process_reward.verifiable_process_components, gated to 0.0 unless gate_open(pred) (gold-FREE)",
                }.get(arm),
            }
            for arm in R.ARM_IDS
        },
        "invariants": invariants,
        "registry_file_sha256": file_hashes,
        "post_freeze_policy": (
            "terminal bands, epsilon, process weights, and hard gates MUST NOT change based on "
            "NESTFUL diagnostic-subset results after this freeze. Any change requires a new "
            "reward_id and a new ablation_version."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "FROZEN_REWARD_SPECS.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    print(f"[freeze_reward_specs] wrote {out_path}")
    print(f"[freeze_reward_specs] git_commit={spec['git_commit']} git_dirty={spec['git_dirty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
