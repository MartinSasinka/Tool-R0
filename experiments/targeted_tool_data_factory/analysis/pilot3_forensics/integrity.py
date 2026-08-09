"""Input integrity checks and INPUT_MANIFEST generation."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .discovery import DiscoveryResult
from .io import as_bool, count_lines, read_json, read_jsonl, rel_to, sha256_file
from .schemas import detect_schema, field_audit, schema_for_kind


STATUS_VERIFIED = "VERIFIED"
STATUS_PARTIAL = "PARTIALLY_VERIFIED"
STATUS_NOT = "NOT_VERIFIABLE"
STATUS_INCONSISTENT = "INCONSISTENT"


def _artifact_entry(
    repo_root: Path,
    key: str,
    path: Optional[Path],
    artifact_type: str,
    *,
    used: bool,
    reason_unused: str = "",
    notes: str = "",
    load_rows: bool = False,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "key": key,
        "path": rel_to(path, repo_root) if path else None,
        "abs_path": str(path) if path else None,
        "artifact_type": artifact_type,
        "exists": bool(path and path.exists()),
        "used": used,
        "reason_unused": reason_unused,
        "notes": notes,
        "size_bytes": None,
        "n_lines": None,
        "sha256": None,
        "detected_schema": None,
        "required_fields": [],
        "missing_fields": [],
        "invalid_rows": 0,
    }
    if not path or not path.exists():
        return entry
    entry["size_bytes"] = path.stat().st_size if path.is_file() else None
    if path.is_file():
        entry["n_lines"] = count_lines(path)
        entry["sha256"] = sha256_file(path)
        if load_rows and path.suffix == ".jsonl":
            rows = read_jsonl(path)
            entry["detected_schema"] = detect_schema(rows, artifact_type)
            req, opt = schema_for_kind(key if key in (
                "c0_trajectories", "d1_trajectories", "c0_hf_trajectories",
                "train_data", "full_train_data", "heldout_data", "reserve_data",
                "diagnostic_data",
            ) else entry["detected_schema"] or "")
            audit = field_audit(rows, req, opt)
            entry["required_fields"] = audit["required_fields"]
            entry["missing_fields"] = audit["missing_required_fields"]
            entry["invalid_rows"] = audit["invalid_rows"]
            entry["field_audit"] = audit
        elif path.suffix == ".json":
            try:
                obj = read_json(path)
                entry["detected_schema"] = type(obj).__name__
            except Exception as exc:  # noqa: BLE001
                entry["detected_schema"] = f"json_error:{exc}"
    return entry


def _extract_log_hparams(log_path: Path) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Prefer W&B summary / config snippets; keep patterns loose.
    patterns = {
        "model": r"(?:model[_/]?name|base_model|pretrained)[=:\s]+([A-Za-z0-9_./\-]+)",
        "kl_beta": r"(?:kl_beta|training\.kl_beta)[=:\s]+([0-9.]+)",
        "dead_group_rate": r"epoch/dead_group_rate\s+([0-9.]+)",
        "mean_reward": r"epoch/mean_reward\s+([0-9.]+)",
        "mean_unique_rewards": r"epoch/mean_unique_rewards\s+([0-9.]+)",
        "temperature": r"temperature[=:\s]+([0-9.]+)",
        "top_p": r"top_p[=:\s]+([0-9.]+)",
        "max_tokens": r"(?:max_new_tokens|max_tokens)[=:\s]+([0-9]+)",
        "tensor_parallel": r"(?:tensor_parallel(?:_size)?|tp_size)[=:\s]+([0-9]+)",
        "gpu_count": r"(?:n_gpus|num_gpus|world_size)[=:\s]+([0-9]+)",
        "use_vllm": r"use_vllm[=:\s]+([A-Za-z0-9_]+)",
        "checkpoint": r"(adapter_epoch_\d+|checkpoints/[A-Za-z0-9_./\-]+)",
        "lora": r"(lora|QLoRA|peft)[^\n]{0,80}",
        "reward_policy": r"(A4_GATED_VERIFIABLE|reward_ablation_[A-Za-z0-9_]+)",
        "run_id": r"(pilot3_D1_[A-Za-z0-9_]+)",
    }
    found: Dict[str, Any] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            found[key] = m.group(1) if m.lastindex else m.group(0)
    # W&B summary block lines
    for line in text.splitlines():
        if "epoch/dead_group_rate" in line and re.search(r"[0-9.]+", line):
            m = re.search(r"([0-9.]+)\s*$", line.strip())
            if m:
                found["dead_group_rate"] = float(m.group(1))
        if "epoch/mean_reward" in line and "dense" not in line:
            m = re.search(r"([0-9.]+)\s*$", line.strip())
            if m:
                found.setdefault("mean_reward", float(m.group(1)))
        if "epoch/mean_unique_rewards" in line:
            m = re.search(r"([0-9.]+)\s*$", line.strip())
            if m:
                found["mean_unique_rewards"] = float(m.group(1))
        if "training.kl_beta" in line or "kl_beta" in line.lower():
            m = re.search(r"([0-9.]+)", line)
            if m and "kl_beta" not in found:
                found["kl_beta"] = float(m.group(1))
    return found


def verify_traj_pairing(
    c0_rows: List[Dict[str, Any]],
    d1_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    c0_ids = [str(r.get("sample_id")) for r in c0_rows]
    d1_ids = [str(r.get("sample_id")) for r in d1_rows]
    c0_counts = Counter(c0_ids)
    d1_counts = Counter(d1_ids)
    dup_c0 = sorted([k for k, v in c0_counts.items() if v > 1])
    dup_d1 = sorted([k for k, v in d1_counts.items() if v > 1])
    set_c0, set_d1 = set(c0_ids), set(d1_ids)
    only_c0 = sorted(set_c0 - set_d1)
    only_d1 = sorted(set_d1 - set_c0)
    shared = sorted(set_c0 & set_d1)

    gold_mismatch = []
    win_c0 = win_d1 = 0
    official_invalid = 0
    for sid in shared:
        # one record assumed after dup check
        r0 = next(r for r in c0_rows if str(r.get("sample_id")) == sid)
        r1 = next(r for r in d1_rows if str(r.get("sample_id")) == sid)
        g0 = int(r0.get("num_gold_calls") or (r0.get("_traj") or {}).get("gold_num_turns") or -1)
        g1 = int(r1.get("num_gold_calls") or (r1.get("_traj") or {}).get("gold_num_turns") or -1)
        if g0 != g1:
            gold_mismatch.append({"sample_id": sid, "c0": g0, "d1": g1})
        for arm, r in (("c0", r0), ("d1", r1)):
            ow = (r.get("_traj") or {}).get("official_win")
            b = as_bool(ow)
            if b is None or (not isinstance(ow, (bool, int, float)) and str(ow) not in ("0", "1", "true", "false")):
                # still accept 0/1 floats commonly used
                if not isinstance(ow, (int, float, bool)):
                    official_invalid += 1
            if arm == "c0" and b:
                win_c0 += 1
            if arm == "d1" and b:
                win_d1 += 1

    status = STATUS_VERIFIED
    if only_c0 or only_d1 or dup_c0 or dup_d1 or gold_mismatch:
        status = STATUS_INCONSISTENT
    elif official_invalid:
        status = STATUS_PARTIAL

    return {
        "status": status,
        "n_c0": len(c0_rows),
        "n_d1": len(d1_rows),
        "n_shared": len(shared),
        "only_c0": only_c0[:20],
        "only_d1": only_d1[:20],
        "n_only_c0": len(only_c0),
        "n_only_d1": len(only_d1),
        "duplicate_c0": dup_c0,
        "duplicate_d1": dup_d1,
        "num_gold_calls_mismatches": gold_mismatch[:20],
        "n_gold_mismatches": len(gold_mismatch),
        "wins_c0_recount": win_c0,
        "wins_d1_recount": win_d1,
        "official_win_invalid_count": official_invalid,
        "pairing_ok": status in (STATUS_VERIFIED, STATUS_PARTIAL) and len(shared) > 0 and not only_c0 and not only_d1 and not dup_c0 and not dup_d1,
    }


def verify_shards_vs_merged(merged_path: Path, shards_dir: Path) -> Dict[str, Any]:
    if not shards_dir.is_dir():
        return {"status": STATUS_NOT, "reason": "shards dir missing"}
    shard_files = sorted(shards_dir.glob("**/final_eval_trajectories.jsonl"))
    if not shard_files:
        shard_files = sorted(shards_dir.glob("**/*trajectories*.jsonl"))
    if not shard_files:
        return {"status": STATUS_NOT, "reason": "no shard trajectory files"}
    merged = {str(r["sample_id"]): r for r in read_jsonl(merged_path)}
    shard_ids = []
    shard_wins = 0
    for sf in shard_files:
        for r in read_jsonl(sf):
            sid = str(r.get("sample_id"))
            shard_ids.append(sid)
            if as_bool((r.get("_traj") or {}).get("official_win")):
                shard_wins += 1
    dup = [k for k, v in Counter(shard_ids).items() if v > 1]
    set_s, set_m = set(shard_ids), set(merged)
    status = STATUS_VERIFIED
    if set_s != set_m or dup:
        status = STATUS_INCONSISTENT
    merged_wins = sum(1 for r in merged.values() if as_bool((r.get("_traj") or {}).get("official_win")))
    if shard_wins != merged_wins:
        status = STATUS_INCONSISTENT
    return {
        "status": status,
        "n_shard_files": len(shard_files),
        "n_shard_rows": len(shard_ids),
        "n_merged": len(merged),
        "duplicate_shard_ids": dup[:20],
        "only_shards": sorted(set_s - set_m)[:20],
        "only_merged": sorted(set_m - set_s)[:20],
        "wins_shards": shard_wins,
        "wins_merged": merged_wins,
    }


def build_input_manifest(
    discovery: DiscoveryResult,
    *,
    critical_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Return (manifest_json, integrity_checks, integrity_md)."""
    critical_keys = critical_keys or ["c0_trajectories", "d1_trajectories", "diagnostic_data", "train_data"]
    repo = discovery.repo_root
    entries = {}
    for key, ref in discovery.artifacts.items():
        used = key in critical_keys and bool(ref.path and ref.path.exists())
        # will refine used flags later
        entries[key] = _artifact_entry(
            repo,
            key,
            ref.path,
            ref.artifact_type,
            used=False,
            reason_unused=ref.reason_unused,
            notes=ref.notes,
            load_rows=key in {
                "c0_trajectories", "d1_trajectories", "c0_hf_trajectories",
                "train_data", "full_train_data", "heldout_data", "reserve_data",
                "diagnostic_data",
            } and bool(ref.path and ref.path.is_file()),
        )

    checks: Dict[str, Any] = {}
    c0_path = discovery.get("c0_trajectories")
    d1_path = discovery.get("d1_trajectories")
    if not c0_path or not d1_path:
        checks["pairing"] = {
            "status": STATUS_INCONSISTENT,
            "pairing_ok": False,
            "reason": "missing C0 or D1 trajectories",
        }
    else:
        c0_rows = read_jsonl(c0_path)
        d1_rows = read_jsonl(d1_path)
        checks["pairing"] = verify_traj_pairing(c0_rows, d1_rows)
        discovery.mark_used("c0_trajectories")
        discovery.mark_used("d1_trajectories")
        entries["c0_trajectories"]["used"] = True
        entries["d1_trajectories"]["used"] = True
        entries["c0_trajectories"]["reason_unused"] = ""
        entries["d1_trajectories"]["reason_unused"] = ""

        # metrics win recount
        for arm, mkey, wins in (
            ("c0", "c0_metrics", checks["pairing"]["wins_c0_recount"]),
            ("d1", "d1_metrics", checks["pairing"]["wins_d1_recount"]),
        ):
            mp = discovery.get(mkey)
            if mp:
                discovery.mark_used(mkey)
                entries[mkey]["used"] = True
                entries[mkey]["reason_unused"] = ""
                try:
                    metrics = read_json(mp)
                    declared = metrics.get("n_wins")
                    if declared is None and "official_win" in metrics:
                        declared = int(round(float(metrics["official_win"]) * int(metrics.get("n_rows") or metrics.get("n_scored") or 0)))
                    checks[f"{arm}_wins_vs_metrics"] = {
                        "status": STATUS_VERIFIED if declared == wins else STATUS_INCONSISTENT,
                        "declared_n_wins": declared,
                        "recounted_wins": wins,
                        "official_win_metric": metrics.get("official_win"),
                    }
                except Exception as exc:  # noqa: BLE001
                    checks[f"{arm}_wins_vs_metrics"] = {"status": STATUS_NOT, "error": str(exc)}

        for arm, skey, tkey in (
            ("c0", "c0_shards_dir", "c0_trajectories"),
            ("d1", "d1_shards_dir", "d1_trajectories"),
        ):
            sd = discovery.get(skey)
            tp = discovery.get(tkey)
            if sd and tp:
                checks[f"{arm}_shards"] = verify_shards_vs_merged(tp, sd)
                discovery.mark_used(skey)
                entries[skey]["used"] = True
                entries[skey]["reason_unused"] = ""
            else:
                checks[f"{arm}_shards"] = {"status": STATUS_NOT, "reason": "missing shards or merged"}

        # manifest diagnostic path comparison
        c0m = discovery.get("c0_manifest")
        d1m = discovery.get("d1_manifest")
        if c0m and d1m:
            try:
                a = read_json(c0m)
                b = read_json(d1m)
                da = Path(str(a.get("diagnostic") or "")).name
                db = Path(str(b.get("diagnostic") or "")).name
                same_n = a.get("n_diagnostic") == b.get("n_diagnostic") == 500
                checks["eval_manifest_parity"] = {
                    "status": STATUS_VERIFIED if (da == db and same_n) else STATUS_PARTIAL,
                    "c0_diagnostic_basename": da,
                    "d1_diagnostic_basename": db,
                    "c0_n_diagnostic": a.get("n_diagnostic"),
                    "d1_n_diagnostic": b.get("n_diagnostic"),
                    "c0_n_gpus": a.get("n_gpus"),
                    "d1_n_gpus": b.get("n_gpus"),
                    "c0_checkpoint": a.get("checkpoint"),
                    "d1_checkpoint": b.get("checkpoint"),
                }
                discovery.mark_used("c0_manifest")
                discovery.mark_used("d1_manifest")
                entries["c0_manifest"]["used"] = True
                entries["d1_manifest"]["used"] = True
            except Exception as exc:  # noqa: BLE001
                checks["eval_manifest_parity"] = {"status": STATUS_NOT, "error": str(exc)}

    # diagnostic hash vs arms' referenced path basename only
    diag = discovery.get("diagnostic_data")
    if diag:
        discovery.mark_used("diagnostic_data")
        entries["diagnostic_data"]["used"] = True
        entries["diagnostic_data"]["reason_unused"] = ""
        checks["diagnostic"] = {
            "status": STATUS_VERIFIED,
            "sha256": entries["diagnostic_data"]["sha256"],
            "n_lines": entries["diagnostic_data"]["n_lines"],
            "note": "diagnostic-500 is a balanced call-count slice, not a natural NESTFUL sample",
        }

    for key in ("train_data", "full_train_data", "heldout_data", "reserve_data", "target_profile",
                "generation_cells", "train_log", "preflight_gold_replay", "pilot2_train",
                "c0_hf_trajectories", "export_manifest", "selection_trace", "profile_match"):
        p = discovery.get(key)
        if p:
            discovery.mark_used(key)
            entries[key]["used"] = True
            entries[key]["reason_unused"] = ""
        else:
            discovery.mark_unused(key, "not found")
            if key in entries:
                entries[key]["used"] = False
                entries[key]["reason_unused"] = "not found"

    # D1 subset identity vs local full train freeze
    train_p = discovery.get("train_data")
    full_p = discovery.get("full_train_data")
    if train_p and full_p:
        sub_ids = {str(r.get("sample_id")) for r in read_jsonl(train_p)}
        full_ids = [str(r.get("sample_id")) for r in read_jsonl(full_p)]
        overlap = len(sub_ids & set(full_ids))
        positional = full_ids[: len(sub_ids)] == list(
            str(r.get("sample_id")) for r in read_jsonl(train_p)
        )
        status = STATUS_VERIFIED if positional else (
            STATUS_INCONSISTENT if overlap < int(0.9 * max(1, len(sub_ids))) else STATUS_PARTIAL
        )
        checks["train_subset_identity"] = {
            "status": status,
            "n_subset": len(sub_ids),
            "n_full": len(full_ids),
            "overlap": overlap,
            "positional_prefix_match": positional,
            "note": (
                "D1 subset is not the local train_grpo_pilot3 prefix; "
                "local full-train audits only partially represent D1."
                if status == STATUS_INCONSISTENT else
                "Subset matches local full-train prefix."
            ),
        }

    # rollout: only use if Pilot3; else mark unused with reason
    rl = discovery.artifacts.get("rollout_log")
    if rl and rl.path and rl.path.exists():
        if "pilot2" in str(rl.path).replace("\\", "/").lower() and "pilot3" not in str(rl.path).replace("\\", "/").lower():
            discovery.mark_unused("rollout_log", "Pilot2 rollouts are not D1 training rollouts; reward cell audit limited")
            entries["rollout_log"]["used"] = False
            entries["rollout_log"]["reason_unused"] = discovery.artifacts["rollout_log"].reason_unused
            checks["rollouts"] = {"status": STATUS_NOT, "reason": entries["rollout_log"]["reason_unused"]}
        else:
            discovery.mark_used("rollout_log")
            entries["rollout_log"]["used"] = True
            checks["rollouts"] = {"status": STATUS_PARTIAL, "path": entries["rollout_log"]["path"]}
    else:
        checks["rollouts"] = {"status": STATUS_NOT, "reason": "no per-rollout Pilot3 artifact"}

    train_log = discovery.get("train_log")
    if train_log:
        hp = _extract_log_hparams(train_log)
        checks["train_log_hparams"] = {"status": STATUS_PARTIAL if hp else STATUS_NOT, "extracted": hp}
    else:
        checks["train_log_hparams"] = {"status": STATUS_NOT, "extracted": {}}

    # Mark remaining unused
    for key, entry in entries.items():
        if not entry["used"] and entry["exists"] and not entry["reason_unused"]:
            entry["reason_unused"] = "optional / not required for this phase"
        if not entry["exists"]:
            entry["reason_unused"] = entry["reason_unused"] or "not found"

    manifest = {
        "schema_version": "pilot3_forensics.input_manifest.v1",
        "artifacts": entries,
        "critical_keys": critical_keys,
    }

    # Markdown
    lines = ["# INPUT_INTEGRITY", "", "Status legend: VERIFIED | PARTIALLY_VERIFIED | NOT_VERIFIABLE | INCONSISTENT", ""]
    for name, chk in checks.items():
        st = chk.get("status", STATUS_NOT)
        lines.append(f"## {name}")
        lines.append(f"- status: `{st}`")
        for k, v in chk.items():
            if k == "status":
                continue
            if isinstance(v, (list, dict)) and len(str(v)) > 300:
                lines.append(f"- {k}: *(see JSON)*")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.append("## Artifact usage")
    for key, entry in sorted(entries.items()):
        flag = "USED" if entry["used"] else "UNUSED"
        lines.append(f"- `{key}` [{flag}] `{entry.get('path')}` — {entry.get('reason_unused') or entry.get('notes') or ''}")
    integrity_md = "\n".join(lines) + "\n"
    return manifest, checks, integrity_md


def require_pairing_ok(checks: Dict[str, Any]) -> None:
    pairing = checks.get("pairing") or {}
    if not pairing.get("pairing_ok"):
        raise SystemExit(
            "CRITICAL: C0/D1 pairing failed integrity checks. "
            f"status={pairing.get('status')} detail={pairing}"
        )
