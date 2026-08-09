"""Offline sampler simulation (Phase O).

Replays reward data that already exists on disk. When a real per-rollout log is
available the group outcomes come from it; otherwise the simulator falls back
to a *synthetic response model* whose success probability is a function of the
task's own difficulty signature. The fallback is clearly labelled in the output
because it says nothing about the real model.

No rollout, no model, no GPU.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..repro import stamp, write_csv, write_json, write_text
from . import (DEFAULT_CONFIG, SAMPLERS, SCHEMA_VERSION, GroupObservation,
               PromptRef, prompt_refs_from_dataset, refill_batch,
               sampling_entropy)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ReplayResponseModel:
    """Group outcomes taken from a recorded per-rollout log."""

    source = "recorded_rollout_log"

    def __init__(self, rollout_rows: Sequence[Dict[str, Any]]) -> None:
        self.by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rollout_rows:
            pid = str(row.get("prompt_id") or row.get("sample_id") or "")
            if pid:
                self.by_prompt[pid].append(row)
        self.cursor: Counter = Counter()

    def covers(self, prompt_id: str) -> bool:
        return bool(self.by_prompt.get(prompt_id))

    def group(self, prompt: PromptRef, step: int, group_size: int,
              rng: random.Random) -> Optional[GroupObservation]:
        rows = self.by_prompt.get(prompt.prompt_id)
        if not rows:
            return None
        start = self.cursor[prompt.prompt_id] % len(rows)
        self.cursor[prompt.prompt_id] += group_size
        window = [rows[(start + i) % len(rows)] for i in range(group_size)]
        return GroupObservation(
            global_step=step, prompt_id=prompt.prompt_id,
            generation_cell=prompt.generation_cell,
            semantic_program_family=prompt.semantic_program_family,
            difficulty_signature=prompt.difficulty_signature,
            group_size=group_size,
            terminal_rewards=[float(r.get("terminal_reward") or 0.0) for r in window],
            process_rewards=[float(r.get("process_reward") or 0.0) for r in window],
            total_rewards=[float(r["total_reward"]) for r in window
                           if r.get("total_reward") is not None] or [],
            parse_flags=[bool(r.get("parse_valid", True)) for r in window],
            executable_flags=[bool(r.get("executable", True)) for r in window],
            call_bucket=prompt.call_bucket, pattern_family=prompt.pattern_family,
            query_mode=prompt.query_mode,
            capability_families=prompt.capability_families,
            difficulty_band=prompt.difficulty_band)


class DifficultyResponseModel:
    """Synthetic stand-in used only when no rollout log exists.

    Success probability decays with the difficulty score, and the process
    reward is a noisy partial-credit signal. This exists to exercise the
    sampler's control flow, not to estimate anything about the real policy.
    """

    source = "synthetic_difficulty_model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def covers(self, prompt_id: str) -> bool:
        return True

    def _p_success(self, prompt: PromptRef) -> float:
        sig = prompt.difficulty_signature or {}
        s = sig.get("structural") or {}
        q = sig.get("query") or {}
        env = sig.get("environment") or {}
        z = (1.6
             - 0.28 * float(s.get("call_count", 2))
             - 0.15 * float(s.get("n_joins", 0))
             - 0.20 * float(s.get("n_late_references", 0))
             + 0.60 * float(q.get("operation_explicitness", 0.0))
             - 0.05 * float(env.get("hard_distractor_count", 0)))
        return 1.0 / (1.0 + math.exp(-z))

    def group(self, prompt: PromptRef, step: int, group_size: int,
              rng: random.Random) -> GroupObservation:
        p = self._p_success(prompt)
        terminal, process, parse, executable = [], [], [], []
        for _ in range(group_size):
            win = rng.random() < p
            terminal.append(1.0 if win else 0.0)
            partial = min(max(rng.gauss(p, 0.18), 0.0), 1.0)
            process.append(round(0.5 * (1.0 if win else partial), 4))
            parse.append(rng.random() < 0.95)
            executable.append(rng.random() < 0.9)
        return GroupObservation(
            global_step=step, prompt_id=prompt.prompt_id,
            generation_cell=prompt.generation_cell,
            semantic_program_family=prompt.semantic_program_family,
            difficulty_signature=prompt.difficulty_signature,
            group_size=group_size, terminal_rewards=terminal,
            process_rewards=process, parse_flags=parse,
            executable_flags=executable, call_bucket=prompt.call_bucket,
            pattern_family=prompt.pattern_family, query_mode=prompt.query_mode,
            capability_families=prompt.capability_families,
            difficulty_band=prompt.difficulty_band)


DEFAULT_DATASETS = [
    "outputs/pilot4_profile_safe/train.jsonl",
    "outputs/selected/export_pilot3/train_grpo_pilot3.jsonl",
]


def run_simulation(repo_root: Path, out_dir: Path, *,
                   rollout_log: Optional[Path] = None,
                   dataset: Optional[Path] = None,
                   samplers: Optional[Sequence[str]] = None,
                   steps: int = 200, group_size: int = 8, seed: int = 0,
                   cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    module_root = repo_root / "experiments" / "targeted_tool_data_factory"
    ds_path = dataset
    if ds_path is None:
        for rel in DEFAULT_DATASETS:
            if (module_root / rel).exists():
                ds_path = module_root / rel
                break
    if ds_path is None:
        raise FileNotFoundError("no dataset found for the sampler simulation")

    rows = _load_jsonl(Path(ds_path))
    prompts = prompt_refs_from_dataset(rows)

    if rollout_log and Path(rollout_log).exists():
        model: Any = ReplayResponseModel(_load_jsonl(Path(rollout_log)))
        covered = sum(1 for p in prompts if model.covers(p.prompt_id))
        if covered == 0:
            model = DifficultyResponseModel(seed)
    else:
        model = DifficultyResponseModel(seed)

    names = list(samplers or ["uniform", "dynamic_effective_group",
                              "history_adaptive", "cell_curriculum"])
    results: Dict[str, Any] = {}
    step_rows: List[Dict[str, Any]] = []

    for name in names:
        cls = SAMPLERS.get(name)
        if cls is None:
            continue
        sampler = cls(list(prompts), config=dict(DEFAULT_CONFIG), seed=seed)
        rng = random.Random(f"sim:{name}:{seed}")

        def score(prompt: PromptRef, step: int) -> GroupObservation:
            return model.group(prompt, step, group_size, rng)

        totals = Counter()
        per_step: List[Dict[str, Any]] = []
        for step in range(steps):
            summary = refill_batch(sampler, score, global_step=step)
            for key in ("candidate_prompt_count", "accepted_effective_groups",
                        "rejected_groups", "rejected_all_correct",
                        "rejected_all_fail_no_progress",
                        "retained_all_fail_with_progress", "refill_rounds"):
                totals[key] += summary[key]
            row = {
                "sampler": name, "global_step": step,
                "candidate_prompt_count": summary["candidate_prompt_count"],
                "accepted_effective_groups": summary["accepted_effective_groups"],
                "rejected_groups": summary["rejected_groups"],
                "dead_group_rate_before_filtering":
                    summary["dead_group_rate_before_filtering"],
                "effective_group_rate_after_filtering":
                    summary["effective_group_rate_after_filtering"],
                "rollout_utilization": summary["rollout_utilization"],
                "refill_rounds": summary["refill_rounds"],
                "target_reached": summary["target_reached"],
                "sampler_entropy": sampling_entropy(sampler.prompts),
                "n_cells_touched": len(sampler.state.axis["generation_cell"]),
            }
            per_step.append(row)
            step_rows.append(row)

        n_groups = totals["accepted_effective_groups"] + totals["rejected_groups"]
        curriculum = Counter(sampler.state.curriculum.values())
        results[name] = {
            "sampler": name,
            "n_steps": steps,
            "group_size": group_size,
            "n_groups": n_groups,
            "mean_dead_group_rate_before_filtering": round(
                sum(r["dead_group_rate_before_filtering"] for r in per_step)
                / max(len(per_step), 1), 4),
            "mean_effective_group_rate_after_filtering": round(
                sum(r["effective_group_rate_after_filtering"] for r in per_step)
                / max(len(per_step), 1), 4),
            "mean_rollout_utilization": round(
                sum(r["rollout_utilization"] for r in per_step)
                / max(len(per_step), 1), 4),
            "mean_refill_rounds": round(totals["refill_rounds"] / max(steps, 1), 3),
            "rollouts_spent": n_groups * group_size,
            "effective_groups": totals["accepted_effective_groups"],
            "retained_all_fail_with_progress":
                totals["retained_all_fail_with_progress"],
            "rejected_all_correct": totals["rejected_all_correct"],
            "rejected_all_fail_no_progress":
                totals["rejected_all_fail_no_progress"],
            "final_sampler_entropy": sampling_entropy(sampler.prompts),
            "n_prompts_touched": len(sampler.state.prompt),
            "curriculum_states": dict(curriculum),
        }
        write_json(out_dir / f"sampler_state_{name}.json", sampler.state_dict())

    payload = {
        "schema_version": SCHEMA_VERSION,
        "response_model": model.source,
        "response_model_caveat": (
            "synthetic difficulty model: exercises the sampler only, it is not "
            "evidence about the trained policy"
            if model.source == "synthetic_difficulty_model"
            else "replayed from a recorded rollout log"),
        "dataset": str(ds_path),
        "n_prompts": len(prompts),
        "steps": steps,
        "group_size": group_size,
        "seed": seed,
        "results": results,
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args, seeds={"simulation": seed},
                            input_paths=[Path(ds_path)]),
    }
    write_json(out_dir / "SAMPLER_SIMULATION.json", payload)
    write_csv(out_dir / "SAMPLER_SIMULATION_STEPS.csv", step_rows)
    write_text(out_dir / "SAMPLER_SIMULATION.md", _markdown(payload))
    return {"n_samplers": len(results), "n_steps": steps, "out_dir": str(out_dir),
            "payload": payload}


def _markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# SAMPLER_SIMULATION", "",
        f"- response model: `{payload['response_model']}`",
        f"- caveat: {payload['response_model_caveat']}",
        f"- dataset: `{payload['dataset']}` ({payload['n_prompts']} prompts)",
        f"- steps: {payload['steps']}, group size: {payload['group_size']}, "
        f"seed: {payload['seed']}", "",
        "No rollout, no model and no GPU were used to produce this table.", "",
        "| sampler | dead-group rate before filter | effective-group rate after "
        "filter | rollout utilisation | refill rounds | entropy |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in payload["results"].items():
        lines.append(
            f"| `{name}` | {r['mean_dead_group_rate_before_filtering']} | "
            f"{r['mean_effective_group_rate_after_filtering']} | "
            f"{r['mean_rollout_utilization']} | {r['mean_refill_rounds']} | "
            f"{r['final_sampler_entropy']} |")
    lines += ["", "## Curriculum states at the end of the simulation", ""]
    for name, r in payload["results"].items():
        if r["curriculum_states"]:
            lines.append(f"- `{name}`: {r['curriculum_states']}")
    lines.append("")
    return "\n".join(lines)
