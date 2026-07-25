from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS, DEFAULT_SEED
from lib.offline_audit.paths import (
    c0_eval_dir,
    eval_dir,
    final_adapter_path,
    load_json,
    resolve_local_path,
    run_dir,
    sha256_file,
    train_log_path,
    train_summary_path,
    V3_ROOT,
)


def _reward_dispatch_check(arm: str, tl: Path) -> Dict[str, Any]:
    """Compare the DECLARED arm reward policy against what train_log.jsonl
    says actually ran (header `reward_dispatch` + per-row policy fields).

    Regression guard for the Round-1 dispatch bug (2026-07-24): every arm's
    train log showed resolved_policy=execution_aware_v3_2_dense while the run
    manifest declared reward_ablation_A*_... — i.e. the ablation never varied
    the reward. An offline audit MUST fail loudly on that mismatch instead of
    comparing downstream signals."""
    expected = ("execution_aware_v3_2_dense" if arm == "A0_R0_CURRENT"
                else f"reward_ablation_{arm}")
    out: Dict[str, Any] = {"expected_policy": expected, "logged_policies": [],
                           "dispatch_ok": None}
    if not tl.is_file():
        return out
    logged = set()
    with open(tl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rd = row.get("reward_dispatch")
            if isinstance(rd, dict):
                logged.add(str(rd.get("resolved_policy")))
            for key in ("reward_policy_resolved", "reward_train_policy"):
                if row.get(key):
                    logged.add(str(row[key]))
    out["logged_policies"] = sorted(logged)
    # None (not False) when the log carries no policy info at all — coverage
    # problem, not a proven mismatch.
    out["dispatch_ok"] = (logged == {expected}) if logged else None
    return out


def discover(runs_root: Path, seed: str, reports_dir: Path, *, strict: bool) -> Dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    arms_out: Dict[str, Any] = {}
    errors: List[str] = []

    ref_manifest = None
    for arm in ARMS:
        rd = run_dir(runs_root, arm, seed)
        manifest_p = rd / "run_manifest.json"
        if not manifest_p.is_file():
            errors.append(f"missing run_manifest for {arm}: {manifest_p}")
            continue
        manifest = load_json(manifest_p)
        if ref_manifest is None:
            ref_manifest = manifest
        ts_local = resolve_local_path(manifest.get("train_subset", ""))
        es_local = resolve_local_path(manifest.get("eval_subset", ""))
        summary_p = train_summary_path(runs_root, arm, seed)
        summary = load_json(summary_p) if summary_p.is_file() else {}
        tl = train_log_path(runs_root, arm, seed)
        ev = eval_dir(runs_root, arm, seed)
        entry = {
            "arm": arm,
            "run_dir": str(rd),
            "seed": manifest.get("seed"),
            "reward_arm": manifest.get("reward_arm"),
            "train_subset_local": str(ts_local),
            "train_subset_hash_manifest": (manifest.get("hashes") or {}).get("dataset_hash"),
            "train_subset_hash_local": sha256_file(ts_local),
            "eval_subset_local": str(es_local),
            "eval_subset_hash_manifest": (manifest.get("hashes") or {}).get("eval_subset_hash"),
            "eval_subset_hash_local": sha256_file(es_local) if es_local.suffix == ".json" else None,
            "reward_spec_hash": (manifest.get("hashes") or {}).get("reward_spec_hash"),
            "optimizer_steps": summary.get("steps"),
            "num_tasks": summary.get("num_tasks"),
            "rollouts_per_group": 8,
            "train_log": str(tl),
            "train_log_exists": tl.is_file(),
            "train_summary_exists": summary_p.is_file(),
            "final_checkpoint": str(final_adapter_path(runs_root, arm, seed)),
            "final_checkpoint_exists": final_adapter_path(runs_root, arm, seed).is_file(),
            "eval_dir": str(ev),
            "eval_trajectories_exists": (ev / "final_eval_trajectories.jsonl").is_file(),
            "reward_dispatch": _reward_dispatch_check(arm, tl),
        }
        if entry["reward_dispatch"]["dispatch_ok"] is False:
            errors.append(
                f"{arm}: REWARD DISPATCH MISMATCH — declared "
                f"{entry['reward_dispatch']['expected_policy']!r} but train log shows "
                f"{entry['reward_dispatch']['logged_policies']}")
        if manifest.get("seed") != int(seed) and str(manifest.get("seed")) != seed:
            errors.append(f"{arm}: seed mismatch {manifest.get('seed')} != {seed}")
        if entry["train_subset_hash_local"] and entry["train_subset_hash_manifest"]:
            if entry["train_subset_hash_local"] != entry["train_subset_hash_manifest"]:
                errors.append(f"{arm}: train subset hash mismatch")
        arms_out[arm] = entry

    c0 = c0_eval_dir(runs_root, seed)
    shared = {
        "c0_eval_dir": str(c0),
        "c0_trajectories_exists": (c0 / "final_eval_trajectories.jsonl").is_file(),
        "v3_root": str(V3_ROOT),
        "runs_root": str(runs_root),
    }
    payload = {
        "seed": seed,
        "arms": arms_out,
        "shared_baseline": shared,
        "reference_reward_spec_hash": (ref_manifest or {}).get("hashes", {}).get("reward_spec_hash"),
        "errors": errors,
        "n_arms_found": len(arms_out),
    }
    if strict and errors:
        payload["strict_failed"] = True

    (reports_dir / "discovery.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# Discovery — Round 1 offline audit",
        "",
        f"- Runs root: `{runs_root}`",
        f"- Seed: `{seed}`",
        f"- Arms found: **{len(arms_out)}** / {len(ARMS)}",
        "",
    ]
    if errors:
        md.append("## Validation issues")
        for e in errors:
            md.append(f"- {e}")
        md.append("")
    md.append("## Per arm")
    for arm, e in arms_out.items():
        md.append(f"### {arm}")
        md.append(f"- Run dir: `{e['run_dir']}`")
        md.append(f"- Optimizer steps: {e.get('optimizer_steps')}")
        md.append(f"- Train log: {e['train_log_exists']}")
        md.append(f"- Final checkpoint: {e['final_checkpoint_exists']}")
        md.append(f"- Eval 500: {e['eval_trajectories_exists']}")
        rdch = e.get("reward_dispatch") or {}
        md.append(f"- Reward dispatch OK: {rdch.get('dispatch_ok')} "
                  f"(expected `{rdch.get('expected_policy')}`, "
                  f"logged {rdch.get('logged_policies')})")
        md.append("")
    (reports_dir / "DISCOVERY.md").write_text("\n".join(md), encoding="utf-8")
    return payload
