"""Training and evaluation logging schemas (Phases P and Q).

Pilot3 left no per-group or per-rollout artifact on disk: the only surviving
evidence of reward collapse was a single W&B epoch scalar
(``epoch/mean_unique_rewards``). These writers make the next run reconstructable
offline without W&B and without the GPU box.

Nothing here imports torch, trl or vllm, so the schemas stay unit-testable on a
laptop; the trainer and the eval script attach them through thin adapters.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

TRAIN_SCHEMA_VERSION = "ttdf.train_logging.v1"
EVAL_SCHEMA_VERSION = "ttdf.eval_logging.v1"

TRAIN_MANIFEST_NAME = "TRAIN_RUN_MANIFEST.json"
EVAL_MANIFEST_NAME = "EVAL_RUN_MANIFEST.json"

ROLLOUT_FIELDS = [
    "run_id", "global_step", "epoch", "batch_id", "group_id", "prompt_id",
    "rollout_id", "generation_cell", "semantic_program_family",
    "difficulty_signature", "prompt_hash", "response_hash", "response_text",
    "parsed_calls", "terminal_reward", "process_reward", "process_components",
    "total_reward", "group_reward_mean", "group_reward_std", "raw_advantage",
    "normalized_advantage", "parse_valid", "executable", "execution_error",
    "final_answer_correct", "solution_equivalent", "stop_reason",
    "n_generated_tokens", "n_tool_calls", "kl",
]

GROUP_FIELDS = [
    "run_id", "global_step", "epoch", "batch_id", "group_id", "prompt_id",
    "generation_cell", "semantic_program_family", "difficulty_signature",
    "group_size", "terminal_rewards", "process_rewards", "total_rewards",
    "reward_mean", "reward_std", "terminal_success_rate",
    "process_std_within_terminal_class", "parse_success_rate",
    "executable_rate", "group_class", "accepted", "rejection_reason",
    "sampler_weight", "sampler_weight_components", "refill_round",
    "n_unique_total_rewards", "dead_group",
]

STEP_FIELDS = [
    "run_id", "global_step", "epoch", "candidate_prompt_count",
    "accepted_effective_groups", "rejected_all_correct",
    "rejected_all_fail_no_progress", "retained_all_fail_with_progress",
    "dead_group_rate_before_filtering", "effective_group_rate_after_filtering",
    "rollout_utilization", "reward_component_means", "kl", "loss",
    "gradient_norm", "sampler_distribution_entropy", "cell_coverage",
    "refill_rounds", "optimizer_step_executed",
]

EVAL_TASK_FIELDS = [
    "run_id", "sample_id", "shard_id", "raw_prompt_hash",
    "input_token_ids_hash", "offered_tools_hash", "generated_text_hash",
    "parsed_calls", "stop_reason", "parse_status", "execution_status",
    "official_win", "n_predicted_calls", "n_gold_calls", "answer_match",
    "solution_equivalent", "runtime_sec", "diagnostics",
]


# ── small helpers (kept dependency-free on purpose) ───────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                  default=str))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_id_hash(ids: Sequence[str]) -> str:
    """Order-sensitive: a reshuffled dataset must not hash the same."""
    return sha256_text("\n".join(str(i) for i in ids))


def _git(args: Sequence[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd or Path.cwd()),
                              capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 - logging must never break a training run
        return ""


def git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or Path.cwd()
    status = _git(["status", "--porcelain"], root)
    return {
        "commit": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
        "dirty_files": [l[3:] for l in status.splitlines()[:50]],
    }


def environment_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "docker_image": os.environ.get("DOCKER_IMAGE", ""),
    }
    for mod in ("torch", "transformers", "trl", "peft", "vllm", "datasets",
                "accelerate", "bitsandbytes"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        import torch  # noqa: PLC0415

        out["cuda"] = torch.version.cuda
        out["n_gpus"] = torch.cuda.device_count()
        out["gpus"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        out["cuda"], out["n_gpus"], out["gpus"] = None, 0, []
    return out


def chat_template_hash(tokenizer: Any) -> str:
    tpl = getattr(tokenizer, "chat_template", None) or ""
    return sha256_text(str(tpl))


class JsonlWriter:
    """Append-only writer with optional gzip. Hashes survive even if the raw
    response text is dropped, so a size-constrained run stays auditable."""

    def __init__(self, path: Path, *, fields: Sequence[str],
                 gzip_enabled: bool = False, drop_fields: Sequence[str] = ()) -> None:
        self.path = Path(str(path) + (".gz" if gzip_enabled
                                      and not str(path).endswith(".gz") else ""))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        self.drop = set(drop_fields)
        self._fh = (gzip.open(self.path, "at", encoding="utf-8")
                    if gzip_enabled else open(self.path, "a", encoding="utf-8"))
        self.n_rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        payload = {k: row.get(k) for k in self.fields if k not in self.drop}
        extra = {k: v for k, v in row.items()
                 if k not in self.fields and k not in self.drop}
        if extra:
            payload["extra"] = extra
        self._fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self.n_rows += 1

    def flush(self) -> None:
        try:
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class TrainRunLogger:
    """Phase P artifacts. Safe to construct without a GPU."""

    out_dir: Path
    run_id: str
    repo_root: Optional[Path] = None
    gzip_rollouts: bool = False
    keep_response_text: bool = True
    _rollouts: Optional[JsonlWriter] = field(default=None, init=False)
    _groups: Optional[JsonlWriter] = field(default=None, init=False)
    _steps: Optional[JsonlWriter] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rollouts = JsonlWriter(
            self.out_dir / "train_rollouts.jsonl", fields=ROLLOUT_FIELDS,
            gzip_enabled=self.gzip_rollouts,
            drop_fields=() if self.keep_response_text else ("response_text",))
        self._groups = JsonlWriter(self.out_dir / "train_groups.jsonl",
                                   fields=GROUP_FIELDS)
        self._steps = JsonlWriter(self.out_dir / "train_steps.jsonl",
                                  fields=STEP_FIELDS)

    # -- manifest
    def write_manifest(self, *, config: Dict[str, Any], dataset_path: Optional[Path],
                       sample_ids: Sequence[str],
                       subset_ids: Optional[Sequence[str]] = None,
                       subset_algorithm: str = "",
                       base_model: str = "", model_revision: str = "",
                       tokenizer_revision: str = "",
                       chat_template_sha: str = "",
                       seeds: Optional[Dict[str, int]] = None,
                       reward_version: str = "", parser_version: str = "",
                       executor_version: str = "", sampler_version: str = "",
                       sampler_config: Optional[Dict[str, Any]] = None,
                       extra: Optional[Dict[str, Any]] = None) -> Path:
        payload = {
            "schema_version": TRAIN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git_info(self.repo_root),
            "environment": environment_versions(),
            "base_model": base_model,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_hash": chat_template_sha,
            "dataset": {
                "path": str(dataset_path) if dataset_path else None,
                "sha256": (sha256_file(Path(dataset_path))
                           if dataset_path and Path(dataset_path).exists() else None),
                "n_rows": len(sample_ids),
                "ordered_sample_id_hash": ordered_id_hash(sample_ids),
                "subset_selection_algorithm": subset_algorithm,
                "n_subset": len(subset_ids or []),
                "subset_ids": list(subset_ids or []),
                "ordered_subset_id_hash": ordered_id_hash(subset_ids or []),
            },
            "seeds": dict(seeds or {}),
            "qlora_config": (config.get("model") or {}).get("qlora")
                            or config.get("qlora") or {},
            "optimizer_config": config.get("training") or {},
            "kl_config": {
                "kl_beta": (config.get("training") or {}).get("kl_beta"),
                "mt_grpo": config.get("mt_grpo") or {},
            },
            "generation_config": config.get("generation") or {},
            "reward_version": reward_version,
            "parser_version": parser_version,
            "executor_version": executor_version,
            "sampler_version": sampler_version,
            "sampler_config": sampler_config or {},
            "config_hash": sha256_obj(config),
            **(extra or {}),
        }
        path = self.out_dir / TRAIN_MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return path

    # -- rows
    def log_rollout(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._rollouts.write(row)

    def log_group(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._groups.write(row)

    def log_step(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._steps.write(row)
        self.flush()

    def save_sampler_state(self, state: Dict[str, Any],
                           checkpoint_dir: Optional[Path] = None) -> List[Path]:
        """Written next to the checkpoint so resume restores the curriculum."""
        target = Path(checkpoint_dir or self.out_dir)
        target.mkdir(parents=True, exist_ok=True)
        written = [target / "sampler_state.json"]
        written[0].write_text(json.dumps(state, indent=2, ensure_ascii=False,
                                         default=str), encoding="utf-8")

        cells = (state.get("axis") or {}).get("generation_cell") or {}
        cell_csv = target / "sampler_cell_stats.csv"
        header = ["generation_cell", "curriculum_state", "n_sampled", "n_rollouts",
                  "group_count", "effective_group_count", "all_correct_count",
                  "all_fail_count", "ema_terminal_success", "ema_total_reward",
                  "ema_reward_variance", "ema_executable_rate",
                  "last_sampled_step", "selection_weight"]
        lines = [",".join(header)]
        curriculum = state.get("curriculum") or {}
        for cell, e in sorted(cells.items()):
            lines.append(",".join(str(v) for v in [
                cell, curriculum.get(cell, ""), e.get("n_sampled", 0),
                e.get("n_rollouts", 0), e.get("group_count", 0),
                e.get("effective_group_count", 0), e.get("all_correct_count", 0),
                e.get("all_fail_count", 0), e.get("ema_terminal_success", 0.0),
                e.get("ema_total_reward", 0.0), e.get("ema_reward_variance", 0.0),
                e.get("ema_executable_rate", 0.0), e.get("last_sampled_step", -1),
                e.get("selection_weight", 0.0)]))
        cell_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(cell_csv)

        prompts = state.get("prompt") or {}
        prompt_path = _write_prompt_stats(target, prompts)
        written.append(prompt_path)
        return written

    def flush(self) -> None:
        for w in (self._rollouts, self._groups, self._steps):
            if w:
                w.flush()

    def close(self) -> None:
        for w in (self._rollouts, self._groups, self._steps):
            if w:
                w.close()


def _write_prompt_stats(target: Path, prompts: Dict[str, Any]) -> Path:
    """Parquet when pyarrow/pandas is present, CSV otherwise."""
    rows = [{"prompt_id": pid, **stats} for pid, stats in sorted(prompts.items())]
    try:
        import pandas as pd  # noqa: PLC0415

        path = target / "sampler_prompt_stats.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        path = target / "sampler_prompt_stats.csv"
        header = list(rows[0].keys()) if rows else ["prompt_id"]
        lines = [",".join(header)]
        for r in rows:
            lines.append(",".join(str(r.get(k, "")) for k in header))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


@dataclass
class EvalRunLogger:
    """Phase Q artifacts."""

    out_dir: Path
    run_id: str
    repo_root: Optional[Path] = None
    _inputs: Optional[JsonlWriter] = field(default=None, init=False)
    _traj: Optional[JsonlWriter] = field(default=None, init=False)
    _scores: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._inputs = JsonlWriter(
            self.out_dir / "eval_inputs.jsonl",
            fields=["run_id", "sample_id", "shard_id", "raw_prompt_hash",
                    "offered_tools_hash", "input_token_ids_hash",
                    "n_gold_calls", "prompt_chars"])
        self._traj = JsonlWriter(
            self.out_dir / "eval_trajectories.jsonl",
            fields=["run_id", "sample_id", "shard_id", "generated_text",
                    "generated_text_hash", "parsed_calls", "observations",
                    "stop_reason", "parse_status", "execution_status",
                    "runtime_sec"])

    def write_manifest(self, *, model_revision: str, adapter_path: Optional[str],
                       adapter_hash: Optional[str], merged_lora: bool,
                       backend: str, dataset_path: Optional[Path],
                       sample_ids: Sequence[str],
                       shard_manifest: Optional[Dict[str, Any]] = None,
                       generation: Optional[Dict[str, Any]] = None,
                       chat_template_sha: str = "",
                       tool_schema_hash: str = "",
                       parser_commit: str = "", scorer_commit: str = "",
                       scorer_config: Optional[Dict[str, Any]] = None,
                       extra: Optional[Dict[str, Any]] = None) -> Path:
        gen = generation or {}
        payload = {
            "schema_version": EVAL_SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git_info(self.repo_root),
            "environment": environment_versions(),
            "model_revision": model_revision,
            "adapter": {"path": adapter_path, "hash": adapter_hash,
                        "merged_lora": merged_lora,
                        "mode": "merged" if merged_lora else "dynamic_lora"},
            "backend": backend,
            "generation": {
                "seed": gen.get("seed"), "temperature": gen.get("temperature"),
                "top_p": gen.get("top_p"), "max_tokens": gen.get("max_tokens"),
                "stop": gen.get("stop"), "dtype": gen.get("dtype"),
                "quantization": gen.get("quantization"),
                "tensor_parallel_size": gen.get("tensor_parallel_size"),
            },
            "chat_template_hash": chat_template_sha,
            "tool_schema_serialization_hash": tool_schema_hash,
            "dataset": {
                "path": str(dataset_path) if dataset_path else None,
                "sha256": (sha256_file(Path(dataset_path))
                           if dataset_path and Path(dataset_path).exists() else None),
                "n_rows": len(sample_ids),
                "ordered_sample_id_hash": ordered_id_hash(sample_ids),
                "sample_ids": list(sample_ids),
            },
            "shard_manifest": shard_manifest or {},
            "parser_commit": parser_commit,
            "scorer_commit": scorer_commit,
            "scorer_configuration": scorer_config or {},
            **(extra or {}),
        }
        path = self.out_dir / EVAL_MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return path

    def log_input(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._inputs.write(row)

    def log_trajectory(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._traj.write(row)

    def log_task_score(self, row: Dict[str, Any]) -> None:
        row.setdefault("run_id", self.run_id)
        self._scores.append(row)

    def write_task_scores(self) -> Path:
        path = self.out_dir / "eval_task_scores.csv"
        lines = [",".join(EVAL_TASK_FIELDS)]
        for r in self._scores:
            lines.append(",".join(
                _csv_cell(r.get(k)) for k in EVAL_TASK_FIELDS))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def close(self) -> Path:
        for w in (self._inputs, self._traj):
            if w:
                w.close()
        return self.write_task_scores()


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = str(value).replace('"', '""')
    return f'"{text}"' if any(c in text for c in ',"\n') else text


# ── paired-evaluation comparability ───────────────────────────────────────
PAIRED_CHECKS = ["same_task_set", "same_task_order", "same_prompt_hashes",
                 "same_tool_schema_hashes", "same_scorer_version"]


def compare_eval_runs(manifest_a: Dict[str, Any], manifest_b: Dict[str, Any],
                      inputs_a: Sequence[Dict[str, Any]] = (),
                      inputs_b: Sequence[Dict[str, Any]] = ()
                      ) -> Dict[str, Any]:
    """Guard for paired C0/D1 comparisons.

    Pilot3's headline gap turned out to be a backend confound, so a paired
    comparison must be refused unless the two runs saw the same inputs in the
    same order under the same scorer.
    """
    ids_a = list((manifest_a.get("dataset") or {}).get("sample_ids") or [])
    ids_b = list((manifest_b.get("dataset") or {}).get("sample_ids") or [])
    by_id_a = {r.get("sample_id"): r for r in inputs_a}
    by_id_b = {r.get("sample_id"): r for r in inputs_b}
    shared = [i for i in ids_a if i in by_id_b]

    results = {
        "same_task_set": set(ids_a) == set(ids_b),
        "same_task_order": ids_a == ids_b,
        "same_prompt_hashes": all(
            by_id_a.get(i, {}).get("raw_prompt_hash")
            == by_id_b.get(i, {}).get("raw_prompt_hash") for i in shared)
            if shared else None,
        "same_tool_schema_hashes": all(
            by_id_a.get(i, {}).get("offered_tools_hash")
            == by_id_b.get(i, {}).get("offered_tools_hash") for i in shared)
            if shared else (manifest_a.get("tool_schema_serialization_hash")
                            == manifest_b.get("tool_schema_serialization_hash")),
        "same_scorer_version": (manifest_a.get("scorer_commit")
                                == manifest_b.get("scorer_commit")),
    }
    backend_a = manifest_a.get("backend")
    backend_b = manifest_b.get("backend")
    warnings: List[str] = []
    if backend_a != backend_b:
        warnings.append(
            f"backend differs ({backend_a} vs {backend_b}): the measured gap "
            "mixes the adapter effect with an inference-engine effect")
    for key, ok in results.items():
        if ok is False:
            warnings.append(f"{key} failed")
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "checks": results,
        "comparable": all(v is not False for v in results.values())
                      and backend_a == backend_b,
        "warnings": warnings,
        "n_shared_tasks": len(shared),
    }
