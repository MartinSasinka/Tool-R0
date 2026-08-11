"""Bridge: NestfulProfileSampler ↔ nestful_mtgrpo_minimal trainer.

Constructs the sampler from config, restores checkpointed state, and exposes
helpers for distribution-preserving dynamic batches. Rollouts stay in the
trainer; this module only owns sampling / classification / quota state.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_FACTORY_SRC = _HERE.parent / "targeted_tool_data_factory" / "src"
if str(_FACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(_FACTORY_SRC))

from targeted_tool_data.sampling.nestful_profile import (  # noqa: E402
    ALL_CORRECT, ALL_FAIL_NO_PROGRESS, ALL_FAIL_WITH_PROGRESS, CALL_BUCKETS,
    INVALID_GROUP, LOW_VARIANCE, MIXED_EFFECTIVE, NESTFUL_CALL_SHARES,
    NESTFUL_SAMPLER_DEFAULTS, NestfulProfileSampler, GroupObservation,
    classify_group, is_effective_nestful, load_profile_enrichment_refs,
    map_group_class, nestful_refill_batch, pool_of, register,
)

# re-exports for trainer
__all__ = [
    "ALL_CORRECT", "ALL_FAIL_NO_PROGRESS", "ALL_FAIL_WITH_PROGRESS",
    "CALL_BUCKETS", "INVALID_GROUP", "LOW_VARIANCE", "MIXED_EFFECTIVE",
    "NESTFUL_CALL_SHARES", "NestfulProfileSampler", "GroupObservation",
    "build_sampler_from_config", "classify_group", "classify_rollout_group",
    "estimate_oversample", "is_effective_nestful", "map_group_class",
    "maybe_restore_sampler", "nestful_refill_batch", "plan_epoch_candidates",
    "pool_of", "refill_same_bucket", "rewards_to_observation",
    "rolling_warn", "save_sampler_artifacts",
]

register()

SAMPLER_MODES = (
    "uniform_profile",
    "dynamic_profile",
    "dynamic_profile_plus_enrichment",
)


def _resolve(path: str, base: Optional[Path] = None) -> str:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = []
    if base is not None:
        candidates.append(base / path)
    candidates.append(_HERE / path)
    candidates.append(_HERE.parent / "targeted_tool_data_factory" / path)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str((_HERE / path).resolve())


def build_sampler_from_config(config: Dict[str, Any],
                              tasks: Optional[Sequence[Dict[str, Any]]] = None
                              ) -> Optional[NestfulProfileSampler]:
    """Return NestfulProfileSampler when sampler.mode is set; else None."""
    scfg = dict(config.get("sampler") or {})
    mode = str(scfg.get("mode") or scfg.get("sampler_mode") or "").strip()
    if not mode:
        return None
    if mode not in SAMPLER_MODES:
        raise ValueError(
            f"unsupported sampler mode {mode!r}; expected one of "
            f"{', '.join(SAMPLER_MODES)}"
        )

    paths = config.get("paths") or {}
    profile_path = scfg.get("profile_jsonl") or paths.get("train_jsonl")
    enrichment_path = scfg.get("enrichment_jsonl") or paths.get("enrichment_jsonl")
    if not profile_path:
        raise ValueError("sampler requires paths.train_jsonl or sampler.profile_jsonl")

    profile_path = _resolve(str(profile_path))
    if mode == "dynamic_profile_plus_enrichment":
        if not enrichment_path:
            raise ValueError(
                "dynamic_profile_plus_enrichment requires sampler.enrichment_jsonl "
                "or paths.enrichment_jsonl")
        enrichment_path = _resolve(str(enrichment_path))
    else:
        enrichment_path = None

    profile_refs, enrich_refs = load_profile_enrichment_refs(
        profile_path, enrichment_path)

    merged = {**NESTFUL_SAMPLER_DEFAULTS, **scfg, "sampler_mode": mode}
    seed = int((config.get("experiment") or {}).get("seed", 42))
    sampler = NestfulProfileSampler(
        profile_refs, enrich_refs if mode.endswith("enrichment") else None,
        config=merged, seed=seed,
    )

    # Index task dicts for rollout lookup (always include both pools from disk)
    by_id: Dict[str, Dict[str, Any]] = {}
    for path in ([profile_path] + ([enrichment_path] if enrichment_path else [])):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_id[str(row.get("task_id"))] = row
    if tasks:
        for t in tasks:
            by_id[str(t.get("task_id"))] = t
    sampler.task_by_id = by_id  # type: ignore[attr-defined]
    return sampler


def maybe_restore_sampler(
        sampler: NestfulProfileSampler,
        checkpoint_dir: Optional[str],
        config: Optional[Dict[str, Any]] = None,
) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir) / "sampler_state.json"
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    prev_mode = str(getattr(sampler, "sampler_mode", "") or "")
    sampler.load_state_dict(state)
    # Continuation configs may change pool mix (e.g. enable enrichment).
    # load_state_dict restores sampler_mode from the checkpoint — re-apply the
    # live config so profile/enrichment shares take effect.
    if config is not None:
        scfg = dict(config.get("sampler") or {})
        live_mode = str(scfg.get("mode") or scfg.get("sampler_mode") or "").strip()
        if live_mode:
            sampler.sampler_mode = live_mode
            sampler.cfg["sampler_mode"] = live_mode
        for k in (
            "profile_share",
            "enrichment_share",
            "allow_cross_pool_refill",
            "enrichment_schedule",
        ):
            if k in scfg:
                sampler.cfg[k] = scfg[k]
        if live_mode and live_mode != str(state.get("sampler_mode") or ""):
            print(
                f"[sampler] override restored mode "
                f"{state.get('sampler_mode')!r} -> {live_mode!r} "
                f"(profile_share={sampler.cfg.get('profile_share')} "
                f"enrichment_share={sampler.cfg.get('enrichment_share')})",
                flush=True,
            )
    print(
        f"[sampler] restored state from {path} "
        f"(bootstrap_complete={sampler.bootstrap_complete} "
        f"bootstrap_completed_at_step={sampler.bootstrap_completed_at_step} "
        f"n_prompts={len(getattr(sampler.state, 'prompt', {}) or {})} "
        f"in_bootstrap={sampler.in_bootstrap()} "
        f"mode={sampler.sampler_mode} prev_built={prev_mode})",
        flush=True,
    )


def write_online_bootstrap_report(
        sampler: NestfulProfileSampler,
        out_dir: Path,
) -> Dict[str, str]:
    """Write ONLINE_BOOTSTRAP_REPORT.{json,csv} once bootstrap completes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = sampler.bootstrap_report()
    json_path = out_dir / "ONLINE_BOOTSTRAP_REPORT.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    csv_path = out_dir / "ONLINE_BOOTSTRAP_GROUPS.csv"
    rows = sampler.bootstrap_groups
    if rows:
        keys = list(rows[0].keys())
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join(str(r.get(k, "")) for k in keys))
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        csv_path.write_text(
            "task_id,call_bucket,group_class,effective,reward_mean,reward_std,"
            "terminal_success_rate\n",
            encoding="utf-8")
    sampler._bootstrap_report_written = True
    return {"json": str(json_path), "csv": str(csv_path)}


def save_sampler_artifacts(sampler: NestfulProfileSampler,
                           out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "sampler_state.json"
    state_path.write_text(
        json.dumps(sampler.state_dict(), ensure_ascii=False, indent=2,
                   default=str),
        encoding="utf-8")
    if sampler.bootstrap_complete and not sampler._bootstrap_report_written:
        write_online_bootstrap_report(sampler, out_dir)

    bucket_path = out_dir / "sampler_bucket_stats.json"
    bucket_path.write_text(
        json.dumps({
            "profile_quota": sampler.profile_quota.as_dict(),
            "pool_actual": sampler.pool_actual,
            "nestful_target_shares": NESTFUL_CALL_SHARES,
        }, indent=2),
        encoding="utf-8")

    # prompt stats CSV (parquet optional)
    csv_path = out_dir / "sampler_prompt_stats.csv"
    rows = ["task_id,pool,call_bucket,difficulty,query_mode,times_sampled,"
            "ema_success,ema_reward_std,effective_count,all_correct_count,"
            "all_fail_progress_count,all_fail_no_progress_count,invalid_count,"
            "current_sampling_weight,last_sampled_step,last_group_class"]
    for pid, entry in sorted(sampler.state.prompt.items()):
        ref = sampler.by_id.get(pid)
        ext = sampler._ext(pid)
        rows.append(",".join([
            pid,
            pool_of(ref) if ref else "",
            (ref.call_bucket if ref else ""),
            (ref.difficulty_band if ref else ""),
            (ref.query_mode if ref else ""),
            str(entry.n_sampled),
            f"{entry.ema_terminal_success:.6f}",
            f"{math.sqrt(max(entry.ema_reward_variance, 0.0)):.6f}",
            str(ext.get("total_effective_groups", entry.effective_group_count)),
            str(entry.all_correct_count),
            str(ext.get("all_fail_progress_count", 0)),
            str(entry.consecutive_all_fail_no_progress),
            str(ext.get("invalid_count", 0)),
            f"{ext.get('current_sampling_weight', entry.selection_weight):.6f}",
            str(entry.last_sampled_step),
            str(ext.get("last_group_class") or ""),
        ]))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # try parquet
    parquet_path = out_dir / "sampler_prompt_stats.parquet"
    try:
        import pandas as pd  # type: ignore
        pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
        pq = str(parquet_path)
    except Exception:
        pq = ""

    rng_path = out_dir / "sampler_rng_state.json"
    rng_path.write_text(
        json.dumps({"rng": sampler.rng.getstate()}, default=list),
        encoding="utf-8")

    return {
        "sampler_state": str(state_path),
        "sampler_bucket_stats": str(bucket_path),
        "sampler_prompt_stats_csv": str(csv_path),
        "sampler_prompt_stats_parquet": pq,
        "sampler_rng_state": str(rng_path),
    }


def rewards_to_observation(
        prompt_id: str,
        total_rewards: Sequence[float],
        *,
        terminal_rewards: Optional[Sequence[float]] = None,
        process_rewards: Optional[Sequence[float]] = None,
        parse_flags: Optional[Sequence[bool]] = None,
        executable_flags: Optional[Sequence[bool]] = None,
        call_bucket: str = "",
        query_mode: str = "",
        difficulty_band: str = "",
        global_step: int = 0,
        invalid: bool = False,
) -> GroupObservation:
    n = len(total_rewards)
    term = list(terminal_rewards) if terminal_rewards is not None else [
        1.0 if r >= 0.99 else 0.0 for r in total_rewards
    ]
    proc = list(process_rewards) if process_rewards is not None else [
        float(r) for r in total_rewards
    ]
    obs = GroupObservation(
        global_step=global_step,
        prompt_id=prompt_id,
        group_size=n,
        terminal_rewards=term,
        process_rewards=proc,
        total_rewards=list(map(float, total_rewards)),
        parse_flags=list(parse_flags) if parse_flags is not None else [True] * n,
        executable_flags=list(executable_flags) if executable_flags is not None
        else [True] * n,
        call_bucket=call_bucket,
        query_mode=query_mode,
        difficulty_band=difficulty_band,
    )
    if invalid:
        obs.group_class = INVALID_GROUP
    return obs


def classify_rollout_group(
        rewards: Sequence[float],
        *,
        terminal_success: Optional[Sequence[bool]] = None,
        parse_ok: Optional[Sequence[bool]] = None,
        eps: float = 1e-6,
) -> Tuple[str, bool]:
    """Return (user_facing_class, is_effective)."""
    term = [1.0 if (terminal_success[i] if terminal_success else r >= 0.99) else 0.0
            for i, r in enumerate(rewards)]
    obs = rewards_to_observation(
        "tmp", rewards, terminal_rewards=term,
        parse_flags=parse_ok)
    obs.recompute()
    cls = classify_group(obs, eps_reward=eps, eps_process=eps)
    user = map_group_class(cls, obs, eps)
    eff = is_effective_nestful(obs, {"reward_variance_epsilon": eps,
                                     "epsilon_reward_std": eps,
                                     "epsilon_process_std": eps,
                                     "drop_low_variance": True})
    return user, eff


def estimate_oversample(effective_rate: float, target: int) -> Dict[str, Any]:
    """How many raw candidate groups needed for target effective groups."""
    rate = max(effective_rate, 1e-3)
    need = int(math.ceil(target / rate))
    return {
        "target_effective": target,
        "assumed_effective_rate": round(effective_rate, 4),
        "estimated_raw_candidates": need,
        "oversample_factor": round(need / max(target, 1), 3),
    }


def plan_epoch_candidates(
        sampler: NestfulProfileSampler,
        *,
        target_effective: int,
        oversample_factor: float = 1.5,
) -> List[Dict[str, Any]]:
    """Stratified candidate task list for one epoch (before online refill).

    Uniform mode returns a shuffled copy of the PROFILE pool.
    Dynamic modes draw ``ceil(target * oversample)`` candidates using the
    nestful quota / history weights (pool-aware).
    """
    mode = sampler.sampler_mode
    task_by_id = getattr(sampler, "task_by_id", {}) or {}

    if mode == "uniform_profile":
        ids = [p.prompt_id for p in sampler.profile]
        sampler.rng.shuffle(ids)
        return [task_by_id[i] for i in ids if i in task_by_id]

    n = max(1, int(math.ceil(target_effective * oversample_factor)))
    out: List[Dict[str, Any]] = []
    seen = set()
    for _ in range(n):
        if mode == "dynamic_profile_plus_enrichment":
            total = max(1, sampler.pool_actual.get("PROFILE", 0)
                        + sampler.pool_actual.get("ENRICHMENT", 0) + len(out))
            prof_share = float(sampler.cfg.get("profile_share", 0.80))
            pool = "PROFILE"
            if (sampler.pool_actual.get("PROFILE", 0) + sum(
                    1 for t in out if str(t.get("_pool")) == "PROFILE"
            )) / total > prof_share and sampler.enrichment:
                pool = "ENRICHMENT"
        else:
            pool = "PROFILE"

        if pool == "PROFILE":
            bucket = sampler.pick_profile_bucket()
        else:
            avail = sampler.available_counts("ENRICHMENT")
            bucket = "6+" if avail.get("6+", 0) else max(
                CALL_BUCKETS, key=lambda b: avail.get(b, 0))
        picks = sampler.sample_from_bucket(pool, bucket, 1)
        if not picks:
            continue
        pid = picks[0].prompt_id
        if pid in seen:
            # allow reuse only after full pool exhausted
            if len(seen) < len(task_by_id):
                continue
        seen.add(pid)
        task = task_by_id.get(pid)
        if task is None:
            continue
        row = dict(task)
        row["_pool"] = pool
        row["_call_bucket"] = picks[0].call_bucket
        sampler.note_raw_candidate(picks[0].call_bucket or bucket, pool=pool)
        out.append(row)
    return out


def refill_same_bucket(
        sampler: NestfulProfileSampler,
        *,
        pool: str,
        call_bucket: str,
        exclude_ids: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Pick one replacement from the same pool+bucket (no cross-pool by default)."""
    exclude = set(exclude_ids or [])
    allow_cross = bool(sampler.cfg.get("allow_cross_pool_refill", False))
    for attempt_pool in ((pool,) if not allow_cross else (pool, "PROFILE", "ENRICHMENT")):
        picks = sampler.sample_from_bucket(attempt_pool, call_bucket, 1)
        for p in picks:
            if p.prompt_id in exclude:
                continue
            task = (getattr(sampler, "task_by_id", {}) or {}).get(p.prompt_id)
            if task is None:
                continue
            row = dict(task)
            row["_pool"] = attempt_pool
            row["_call_bucket"] = call_bucket
            # Refill is same-bucket; still count as a RAW candidate draw.
            sampler.note_raw_candidate(call_bucket, pool=attempt_pool)
            return row
    return None


def rolling_warn(
        rates: Sequence[float],
        *,
        threshold: float,
        window: int,
) -> bool:
    if len(rates) < window:
        return False
    recent = rates[-window:]
    return (sum(recent) / len(recent)) < threshold
