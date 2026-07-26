#!/usr/bin/env python3
"""Tests for the post-probe Phase-1 pipeline (no GPU).

Covers offline reward pair taxonomy, variant rescoring, selection hard gate,
Phase-1 subset gates (structure), reward-patch application, and dry-runs of
the audit / verify / canary entry points against the extracted probe zip.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
PROBE_ZIP = (FACTORY / "outputs" / "runpod_pilot2" / "signal_probe_from_zip"
             / "signal_probe")

sys.path.insert(0, str(BUNDLE))
from offline_reward_audit import (  # noqa: E402
    TERMINAL_SCALARS, VARIANT_SPECS, classify_pair, evaluate_variant, rescore,
    select_safest,
)
from phase1_reward_patch import apply_phase1_reward_variant  # noqa: E402
from signal_probe_lib import extract_task_meta  # noqa: E402
from verify_phase1_subset import (  # noqa: E402
    jsd, nestful_jsd, row_features, distribution_report,
)

FAILED: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  FAIL: {msg}")


def _rec(**kw: Any) -> Dict[str, Any]:
    base = {
        "task_id": "t", "rollout_idx": 0, "phase": "P3",
        "terminal_outcome": "official_success", "success": True,
        "process_reward": 1.0, "episode_reward": 0.99,
        "n_successful_calls": 3, "executable_frac": 1.0,
        "parse_error": False, "clipped": False,
        "correct_prefix_len": 3, "arg_key_errors": 0,
        "arg_type_errors": 0, "arg_value_errors": 0,
        "track": "A", "call_count": 3, "motif": "linear",
        "answer_type": "float", "generation_cell": "A_3call_linear_00",
        "reward_policy": "reward_ablation_A4_GATED_VERIFIABLE",
        "cache_key": "k",
    }
    base.update(kw)
    return base


# ─────────────────────────────────────────────────── taxonomy ──

def test_pair_taxonomy() -> None:
    # Terminal-class inversion: success must beat parse.
    a = _rec(rollout_idx=0, terminal_outcome="official_success", success=True)
    b = _rec(rollout_idx=1, terminal_outcome="parse_or_no_call", success=False,
             process_reward=0.0, n_successful_calls=0, executable_frac=0.0,
             correct_prefix_len=0)
    # Reward inverted on purpose.
    kind = classify_pair(a, b, reward_a=0.02, reward_b=0.99)
    check(kind["kind"] == "terminal_class_inversion",
          f"success < parse reward must be terminal_class_inversion, got {kind}")

    # Success-success gold-prefix disagreement: NOT a real inversion.
    a = _rec(rollout_idx=0, correct_prefix_len=3, arg_value_errors=0,
             process_reward=0.9)
    b = _rec(rollout_idx=1, correct_prefix_len=1, arg_value_errors=2,
             process_reward=1.0)
    kind = classify_pair(a, b, reward_a=0.988, reward_b=0.99)
    check(kind["kind"] == "success_success_disagreement",
          f"gold-prefix disagreement inside success = success_success, got {kind}")

    # Clear dominance: more successful calls, same terminal failure class.
    a = _rec(rollout_idx=0, terminal_outcome="executable_wrong_result",
             success=False, n_successful_calls=4, executable_frac=1.0,
             correct_prefix_len=2)
    b = _rec(rollout_idx=1, terminal_outcome="executable_wrong_result",
             success=False, n_successful_calls=1, executable_frac=0.3,
             correct_prefix_len=0)
    kind = classify_pair(a, b, reward_a=0.20, reward_b=0.22)
    check(kind["kind"] == "clear_dominance_inversion",
          f"gold-free dominance with lower reward = clear_dominance, got {kind}")

    # Concordant clear dominance is incomparable (not an inversion).
    kind = classify_pair(a, b, reward_a=0.22, reward_b=0.20)
    check(kind["kind"] == "incomparable_pair",
          f"concordant dominance is not an inversion, got {kind}")


def test_rescore_variants() -> None:
    success = _rec(terminal_outcome="official_success", process_reward=1.0)
    specs = {v["id"]: v for v in VARIANT_SPECS}

    cur = rescore(success, specs["A4_current"])
    check(abs(cur["episode_reward"] - (0.97 + 0.02 * 1.0)) < 1e-9,
          f"A4_current success reward, got {cur}")

    flat = rescore(success, specs["A4_success_flat"])
    check(abs(flat["episode_reward"] - 0.97) < 1e-9,
          f"A4_success_flat must zero process on success, got {flat}")
    check(flat["process_score"] == 0.0, "success_flat process_score is 0")

    # Failure keeps process under success_flat.
    fail = _rec(terminal_outcome="executable_wrong_result", success=False,
                process_reward=0.8)
    flat_f = rescore(fail, specs["A4_success_flat"])
    check(abs(flat_f["episode_reward"] - (0.2 + 0.02 * 0.8)) < 1e-9,
          f"success_flat still shapes failures, got {flat_f}")

    e01 = rescore(success, specs["A4_eps_0.01"])
    check(abs(e01["episode_reward"] - (0.97 + 0.01)) < 1e-9, "eps 0.01")

    e005 = rescore(success, specs["A4_eps_0.005"])
    check(abs(e005["episode_reward"] - (0.97 + 0.005)) < 1e-9, "eps 0.005")

    a1 = rescore(success, specs["A1_outcome_only"])
    check(abs(a1["episode_reward"] - 0.96) < 1e-9, "A1 ignores process")
    check(a1["process_score"] == 0.0, "A1 process is 0")


def test_select_safest_hard_gate() -> None:
    current = {
        "variant_id": "A4_current", "family": "A4", "epsilon": 0.02,
        "success_flat": False, "hard_gate_pass": True,
        "terminal_class_inversions": 0, "clear_dominance_inversions": 0,
        "success_success_disagreements": 10,
        "dead_group_rate": 0.4, "terminal_mixed_rate": 0.3,
        "process_only_mixed_rate": 0.1, "mean_reward_range": 0.2,
        "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
        "description": "x",
    }
    bad = {**current, "hard_gate_pass": False, "terminal_class_inversions": 3}
    flat = {**current, "variant_id": "A4_success_flat", "success_flat": True,
            "dead_group_rate": 0.55}  # flatter success band → more dead
    a1 = {**current, "variant_id": "A1_outcome_only", "family": "A1",
          "terminal_mixed_rate": 0.2, "process_only_mixed_rate": 0.0,
          "dead_group_rate": 0.6,
          "train_policy": "reward_ablation_A1_OUTCOME_ONLY"}
    sel = select_safest([bad, flat, current, a1])
    check(sel["selected"] == "A4_current",
          f"prefer probed A4_current when it clears the hard gate, got {sel}")
    check(sel["hard_gate"] == "PASS", "hard gate pass")

    # If A4_current had clear-dominance inversions, fall back to success_flat.
    current_bad_dom = {**current, "clear_dominance_inversions": 5}
    sel2 = select_safest([current_bad_dom, flat, a1])
    check(sel2["selected"] == "A4_success_flat",
          f"fallback to success_flat when current has clear-dom, got {sel2}")

    only_bad = select_safest([bad])
    check(only_bad["selected"] is None and only_bad["hard_gate"] == "FAIL",
          "no passing variant = FAIL")


def test_terminal_scalars_match_registry() -> None:
    sys.path.insert(0, str(FACTORY.parent / "nestful_synthetic_curriculum_v3"))
    sys.path.insert(0, str(FACTORY.parent))
    from lib.reward_ablation_registry import (  # noqa: WPS433
        EPSILONS, TERMINAL_SCALARS as REG,
    )
    check(abs(EPSILONS["A4_GATED_VERIFIABLE"] - 0.02) < 1e-12, "A4 eps")
    for k, v in TERMINAL_SCALARS["A4"].items():
        check(abs(REG["A4_GATED_VERIFIABLE"][k] - v) < 1e-9,
              f"A4 scalar {k} matches registry")
    for k, v in TERMINAL_SCALARS["A1"].items():
        check(abs(REG["A1_OUTCOME_ONLY"][k] - v) < 1e-9,
              f"A1 scalar {k} matches registry")


def test_jsd_and_nestful_gate() -> None:
    check(jsd({"a": 1.0}, {"a": 1.0}) == 0.0, "identical JSD is 0")
    check(jsd({"a": 1.0}, {"b": 1.0}) > 0.5, "disjoint JSD is large")
    feats = [
        {"call_bucket": "2", "motif": "linear", "answer_type": "float"},
        {"call_bucket": "3", "motif": "fan_in", "answer_type": "float"},
        {"call_bucket": "4", "motif": "linear", "answer_type": "int"},
    ]
    profile = {
        "call_count_dist": {"2": 0.33, "3": 0.22, "4": 0.135, "5": 0.095, "6+": 0.22},
        "motif_dist": {"linear": 0.55, "fan_in": 0.43, "mixed": 0.02},
        "answer_type_dist": {"float": 0.77, "int": 0.05, "string": 0.07,
                            "list": 0.07, "bool": 0.02, "numeric_string": 0.02},
    }
    rep = nestful_jsd(feats, profile)
    check("jsd_call_bucket" in rep["jsd"], "jsd keys present")
    # Tiny synthetic set will fail the gate — that's fine; just check shape.
    check("max_major_jsd" in rep and "pass" in rep, "nestful_jsd shape")


def test_reward_patch_success_flat() -> None:
    sys.path.insert(0, str(FACTORY.parent / "nestful_synthetic_curriculum_v3"))
    sys.path.insert(0, str(FACTORY.parent))
    from lib import reward_ablation_registry as R  # noqa: WPS433

    # Build a minimal fake trajectory-like object via score_arm on a stub is
    # heavy; instead assert the patch mutates EPSILONS and wraps score_arm.
    before = R.score_arm
    sel = {"selected": "A4_success_flat", "family": "A4", "epsilon": 0.02,
           "success_flat": True,
           "train_policy": "reward_ablation_A4_GATED_VERIFIABLE"}
    apply_phase1_reward_variant(sel)
    check(abs(R.EPSILONS["A4_GATED_VERIFIABLE"] - 0.02) < 1e-12, "eps patched")
    check(R.score_arm is not before or getattr(R, "_PHASE1_SUCCESS_FLAT_PATCHED", False),
          "score_arm wrapped for success-flat")

    # Epsilon-only variant.
    R._PHASE1_SUCCESS_FLAT_PATCHED = False  # allow re-entry of eps path
    apply_phase1_reward_variant({
        "selected": "A4_eps_0.01", "family": "A4", "epsilon": 0.01,
        "success_flat": False,
        "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
    })
    check(abs(R.EPSILONS["A4_GATED_VERIFIABLE"] - 0.01) < 1e-12, "eps 0.01 applied")


def test_offline_audit_on_probe_zip() -> None:
    if not PROBE_ZIP.is_dir():
        print("  (skip: probe zip not extracted)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "audit"
        rc = subprocess.run(
            [sys.executable, str(BUNDLE / "offline_reward_audit.py"),
             "--probe-dir", str(PROBE_ZIP),
             "--data", str(BUNDLE / "data" / "train_grpo_pilot2.jsonl"),
             "--out-dir", str(out)],
            capture_output=True, text=True,
        )
        check(rc.returncode == 0, f"offline audit exits 0 ({rc.stderr[-800:]})")
        sel = json.loads((PROBE_ZIP / "SELECTED_REWARD_VARIANT.json").read_text(
            encoding="utf-8"))
        check(sel.get("hard_gate") == "PASS", f"hard gate PASS, got {sel}")
        check(sel.get("selected") in {v["id"] for v in VARIANT_SPECS},
              f"selected is a known variant: {sel.get('selected')}")
        report = json.loads((out / "OFFLINE_REWARD_AUDIT.json").read_text(
            encoding="utf-8"))
        a4 = next(v for v in report["variants"] if v["variant_id"] == "A4_current")
        # Under the refined taxonomy the original probe's gold-prefix noise
        # must land in success_success_disagreement, not terminal_class.
        check(a4["terminal_class_inversions"] == 0,
              f"A4_current has 0 terminal-class inversions, got {a4}")
        check(a4["success_success_disagreements"] > 0,
              "A4_current still reports success-success disagreements")
        md = (out / "OFFLINE_REWARD_AUDIT.md").read_text(encoding="utf-8")
        check("Selected variant" in md, "markdown report written")


def test_verify_phase1_on_probe_zip() -> None:
    if not PROBE_ZIP.is_dir():
        print("  (skip: probe zip not extracted)")
        return
    phase1 = PROBE_ZIP / "recommended_phase1_train.jsonl"
    deferred = PROBE_ZIP / "deferred_phase2_tasks.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "v"
        rc = subprocess.run(
            [sys.executable, str(BUNDLE / "verify_phase1_subset.py"),
             "--phase1", str(phase1), "--deferred", str(deferred),
             "--out-dir", str(out)],
            capture_output=True, text=True,
        )
        check(rc.returncode == 0, f"phase1 verify exits 0 ({rc.stderr[-1000:]})")
        report = json.loads((out / "PHASE1_SUBSET_VERIFICATION.json").read_text(
            encoding="utf-8"))
        check(report["pass"] is True, f"phase1 verification PASS, gates={report['gates']}")
        check(report["gates"]["n_tasks"]["n"] == 80, "80 tasks")
        check(report["gates"]["gold_replay"]["rate"] == 1.0, "replay 100%")
        check(report["gates"]["leakage"]["pass"] is True, "leakage 0")
        check(report["gates"]["nestful_jsd"]["max_major_jsd"] < 0.10,
              f"major JSD < 0.10, got {report['gates']['nestful_jsd']}")


def test_dry_runs() -> None:
    if not PROBE_ZIP.is_dir():
        print("  (skip: probe zip not extracted)")
        return
    rc = subprocess.run(
        [sys.executable, str(BUNDLE / "offline_reward_audit.py"),
         "--probe-dir", str(PROBE_ZIP), "--dry-run"],
        capture_output=True, text=True,
    )
    check(rc.returncode == 0 and "DRY RUN" in rc.stdout, "audit dry-run")

    rc = subprocess.run(
        [sys.executable, str(BUNDLE / "verify_phase1_subset.py"),
         "--phase1", str(PROBE_ZIP / "recommended_phase1_train.jsonl"),
         "--dry-run"],
        capture_output=True, text=True,
    )
    check(rc.returncode == 0 and "DRY RUN" in rc.stdout, "verify dry-run")


def test_train_dry_run_with_variant() -> None:
    if not PROBE_ZIP.is_dir():
        return
    # Ensure a variant file exists.
    subprocess.run(
        [sys.executable, str(BUNDLE / "offline_reward_audit.py"),
         "--probe-dir", str(PROBE_ZIP),
         "--data", str(BUNDLE / "data" / "train_grpo_pilot2.jsonl")],
        capture_output=True, text=True, check=False,
    )
    vf = PROBE_ZIP / "SELECTED_REWARD_VARIANT.json"
    check(vf.is_file(), "variant file present after audit")
    with tempfile.TemporaryDirectory() as tmp:
        rc = subprocess.run(
            [sys.executable, str(BUNDLE / "run_phase1_train.py"),
             "--train-subset", str(PROBE_ZIP / "recommended_phase1_train.jsonl"),
             "--variant-file", str(vf),
             "--output-root", tmp, "--dry-run"],
            capture_output=True, text=True,
        )
        check(rc.returncode == 0, f"train dry-run exits 0 ({rc.stderr[-500:]})")
        check("DRY RUN" in rc.stdout, "train dry-run says so")
        check("optimizer_steps~=20" in rc.stdout or "optimizer_steps≈20" in rc.stdout,
              "budget is 20 steps")


def test_shell_canary_dry_run() -> None:
    bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if not bash.is_file():
        print("  (skip: Git Bash not available)")
        return
    if not PROBE_ZIP.is_dir():
        return
    # Convert Windows path to Git Bash style.
    probe = str(PROBE_ZIP).replace("\\", "/")
    if probe[1] == ":":
        probe = f"/{probe[0].lower()}{probe[2:]}"
    repo = str(FACTORY.parent.parent).replace("\\", "/")
    if repo[1] == ":":
        repo = f"/{repo[0].lower()}{repo[2:]}"
    cmd = (
        f"cd {repo} && PYTHON=python "
        f"bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/"
        f"run_phase1_canary_4gpu.sh --dry-run --probe-dir {probe}"
    )
    rc = subprocess.run(
        [str(bash), "-c", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (rc.stdout or "") + "\n" + (rc.stderr or "")
    check(rc.returncode == 0, f"canary shell dry-run exits 0 ({out[-800:]})")
    check("offline reward audit" in out.lower() or "3/7" in out, "audit stage present")
    check("recommended_phase1" in out or "4/7" in out, "verify stage present")
    check("NOT started" in out or "deferred" in out.lower(), "deferred command printed")
    check("full NESTFUL-1661" in out or "DISABLED" in out, "1661 disabled")


def main() -> int:
    tests = [
        test_pair_taxonomy,
        test_rescore_variants,
        test_select_safest_hard_gate,
        test_terminal_scalars_match_registry,
        test_jsd_and_nestful_gate,
        test_reward_patch_success_flat,
        test_offline_audit_on_probe_zip,
        test_verify_phase1_on_probe_zip,
        test_dry_runs,
        test_train_dry_run_with_variant,
        test_shell_canary_dry_run,
    ]
    for fn in tests:
        print(f"[test] {fn.__name__}")
        fn()
    if FAILED:
        print(f"\n{len(FAILED)} CHECK(S) FAILED:")
        for msg in FAILED:
            print(f"  - {msg}")
        return 1
    print(f"\nok: all {len(tests)} phase1-next tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
