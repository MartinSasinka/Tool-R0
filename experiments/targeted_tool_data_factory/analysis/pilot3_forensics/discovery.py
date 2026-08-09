"""Artifact discovery for Pilot3 forensics."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


FACTORY_REL = Path("experiments/targeted_tool_data_factory")


@dataclass
class ArtifactRef:
    key: str
    path: Optional[Path]
    artifact_type: str
    used: bool = False
    reason_unused: str = ""
    notes: str = ""


@dataclass
class DiscoveryResult:
    repo_root: Path
    factory_root: Path
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Path]:
        ref = self.artifacts.get(key)
        if ref and ref.path and ref.path.exists():
            return ref.path
        return None

    def mark_used(self, key: str) -> None:
        if key in self.artifacts:
            self.artifacts[key].used = True
            self.artifacts[key].reason_unused = ""

    def mark_unused(self, key: str, reason: str) -> None:
        if key in self.artifacts:
            self.artifacts[key].used = False
            self.artifacts[key].reason_unused = reason

    def as_dict(self) -> Dict[str, Any]:
        out = {}
        for k, ref in sorted(self.artifacts.items()):
            out[k] = {
                "key": ref.key,
                "path": str(ref.path) if ref.path else None,
                "exists": bool(ref.path and ref.path.exists()),
                "artifact_type": ref.artifact_type,
                "used": ref.used,
                "reason_unused": ref.reason_unused,
                "notes": ref.notes,
            }
        return out


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.is_file() or p.is_dir():
            return p
    return None


def _largest_matching(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    rx = re.compile(pattern)
    matches = [p for p in directory.iterdir() if p.is_file() and rx.search(p.name)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size)


def discover(
    repo_root: Path,
    *,
    overrides: Optional[Dict[str, Optional[Path]]] = None,
) -> DiscoveryResult:
    """Discover Pilot3 forensic inputs under the repo.

    ``overrides`` maps artifact keys to explicit paths (None = leave auto).
    """
    repo_root = repo_root.resolve()
    factory = repo_root / FACTORY_REL
    zip2 = factory / "outputs/runpod_pilot3_from_zip2/runpod_pilot3"
    zip_alt = factory / "outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500"
    export3 = factory / "outputs/selected/export_pilot3"
    bundle3 = factory / "runpod_bundle_pilot3/data"
    bundle2 = factory / "runpod_bundle_pilot2/data"
    export2 = factory / "outputs/selected/export_pilot2"

    result = DiscoveryResult(repo_root=repo_root, factory_root=factory)

    def add(key: str, path: Optional[Path], artifact_type: str, notes: str = "") -> None:
        result.artifacts[key] = ArtifactRef(
            key=key,
            path=path.resolve() if path else None,
            artifact_type=artifact_type,
            used=False,
            reason_unused="not yet selected" if path else "not found",
            notes=notes,
        )

    # Trajectories — prefer matched vLLM C0
    c0_vllm = _first_existing([
        zip2 / "eval_C0_nestful500_vllm_matched_v2/final_eval_trajectories.jsonl",
        zip2 / "eval_C0_nestful500_vllm_matched/final_eval_trajectories.jsonl",
        factory / "outputs/runpod_pilot3/eval_C0_nestful500_vllm_matched_v2/final_eval_trajectories.jsonl",
    ])
    add("c0_trajectories", c0_vllm, "eval_trajectory", "C0 matched vLLM preferred")

    c0_vllm_dir = c0_vllm.parent if c0_vllm else None
    add("c0_metrics", (c0_vllm_dir / "metrics_merged.json") if c0_vllm_dir and (c0_vllm_dir / "metrics_merged.json").is_file() else None, "metrics")
    add("c0_manifest", (c0_vllm_dir / "eval_manifest.json") if c0_vllm_dir and (c0_vllm_dir / "eval_manifest.json").is_file() else None, "eval_manifest")
    add("c0_shards_dir", (c0_vllm_dir / "shards") if c0_vllm_dir and (c0_vllm_dir / "shards").is_dir() else None, "shards_dir")

    d1_traj = _first_existing([
        zip2 / "train_nestful500/eval/D1_nestful500/final_eval_trajectories.jsonl",
        zip_alt / "eval/D1_nestful500/final_eval_trajectories.jsonl",
    ])
    add("d1_trajectories", d1_traj, "eval_trajectory", "D1 vLLM")
    d1_dir = d1_traj.parent if d1_traj else None
    add("d1_metrics", (d1_dir / "metrics_merged.json") if d1_dir and (d1_dir / "metrics_merged.json").is_file() else None, "metrics")
    add("d1_manifest", (d1_dir / "eval_manifest.json") if d1_dir and (d1_dir / "eval_manifest.json").is_file() else None, "eval_manifest")
    add("d1_shards_dir", (d1_dir / "shards") if d1_dir and (d1_dir / "shards").is_dir() else None, "shards_dir")
    add(
        "d1_predictions",
        (d1_dir / "final_eval_predictions.partial.jsonl")
        if d1_dir and (d1_dir / "final_eval_predictions.partial.jsonl").is_file()
        else None,
        "predictions",
    )

    c0_hf = _first_existing([
        factory / "outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/final_eval_trajectories.jsonl",
    ])
    add("c0_hf_trajectories", c0_hf, "eval_trajectory", "C0 HF backend confound arm")

    # Train / splits
    train_full = _first_existing([
        export3 / "train_grpo_pilot3.jsonl",
        bundle3 / "train_grpo_pilot3.jsonl",
    ])
    add("full_train_data", train_full, "train_grpo")

    train_300 = _first_existing([
        zip2 / "train_nestful500/train_subset_300.jsonl",
        zip_alt / "train_subset_300.jsonl",
    ])
    add("train_data", train_300, "train_grpo", "D1 trained subset n=300")

    heldout = _first_existing([
        export3 / "heldout_grpo_pilot3.jsonl",
        bundle3 / "heldout_grpo_pilot3.jsonl",
        export3 / "heldout_nestful_pilot3.jsonl",
    ])
    add("heldout_data", heldout, "train_grpo")

    reserve = _first_existing([
        export3 / "reserve_grpo_pilot3.jsonl",
        bundle3 / "reserve_grpo_pilot3.jsonl",
        export3 / "reserve_nestful_pilot3.jsonl",
    ])
    add("reserve_data", reserve, "train_grpo")

    diagnostic = _first_existing([
        bundle2 / "nestful_diagnostic_500.jsonl",
        factory / "runpod_bundle_pilot3/data/nestful_diagnostic_500.jsonl",
        export3 / "nestful_diagnostic_500.jsonl",
    ])
    add("diagnostic_data", diagnostic, "nestful_diagnostic")

    # Profile / cells / manifests
    add(
        "target_profile",
        _first_existing([
            factory / "outputs/profiles/nestful_profile.json",
            bundle3 / "nestful_profile.json",
            bundle2 / "nestful_profile.json",
        ]),
        "target_profile",
    )
    add(
        "generation_cells",
        _first_existing([
            factory / "outputs/candidates/cells_pilot3.json",
            factory / "outputs/candidates/cells_pilot3.jsonl",
        ]),
        "generation_cells",
    )
    add(
        "export_manifest",
        _first_existing([export3 / "manifest_pilot3.json"]),
        "export_manifest",
    )
    add(
        "selection_trace",
        _first_existing([factory / "outputs/selected/selection_trace_pilot3.jsonl"]),
        "selection_trace",
    )
    add(
        "profile_match",
        _first_existing([factory / "outputs/selected/profile_match_pilot3.json"]),
        "profile_match",
    )
    add(
        "bundle_sha_manifest",
        _first_existing([factory / "runpod_bundle_pilot3/MANIFEST.sha256.json"]),
        "sha_manifest",
    )

    # Train logs
    train_log_dirs = [
        zip2 / "train_nestful500",
        zip_alt,
    ]
    train_log = None
    for d in train_log_dirs:
        cand = _largest_matching(d, r"run_train_nestful500_.*\.log$")
        if cand:
            train_log = cand
            break
    add("train_log", train_log, "train_log")

    # Preflight / rollouts
    add(
        "preflight_gold_replay",
        _first_existing([
            zip2 / "train_nestful500/preflight_gold_replay.json",
            zip_alt / "preflight_gold_replay.json",
        ]),
        "preflight",
    )
    add(
        "rollout_log",
        _first_existing([
            zip2 / "train_nestful500/canary_rollouts.jsonl",
            zip2 / "signal_probe/rollouts.jsonl",
            factory / "outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/rollouts.jsonl",
        ]),
        "rollouts",
        notes="Pilot3 rollouts preferred; may fall back to Pilot2 probe",
    )
    # Prefer marking pilot2 rollouts as optional/not primary
    if result.artifacts.get("rollout_log") and result.artifacts["rollout_log"].path:
        p = result.artifacts["rollout_log"].path
        if p and "pilot2" in str(p).replace("\\", "/"):
            result.artifacts["rollout_log"].notes = "Pilot2 signal_probe rollouts only (not Pilot3 D1)"

    # Pilot2 train for comparison
    add(
        "pilot2_train",
        _first_existing([
            export2 / "train_grpo_pilot2.jsonl",
            bundle2 / "train_grpo_pilot2.jsonl",
        ]),
        "train_grpo",
    )

    # Apply overrides
    overrides = overrides or {}
    for key, path in overrides.items():
        if path is None:
            continue
        path = Path(path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        else:
            path = path.resolve()
        atype = result.artifacts[key].artifact_type if key in result.artifacts else "override"
        add(key, path if path.exists() else None, atype, notes="CLI override")
        if not path.exists():
            result.artifacts[key].reason_unused = f"override path missing: {path}"

    return result
