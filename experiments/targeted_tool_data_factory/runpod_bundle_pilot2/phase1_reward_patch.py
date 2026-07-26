#!/usr/bin/env python3
"""Apply the offline-selected Phase-1 reward variant before training.

The offline audit may select an A4 flavour that the stock registry does not
expose as its own arm id (success-flat process, or a smaller epsilon). This
module mutates ``lib.reward_ablation_registry`` in place so the existing
``reward_ablation_A4_GATED_VERIFIABLE`` policy name keeps working, while the
scoring matches the selected variant.

Set ``PHASE1_REWARD_VARIANT`` to one of:
  A4_current | A4_success_flat | A4_eps_0.01 | A4_eps_0.005 | A1_outcome_only

Or pass a path to ``SELECTED_REWARD_VARIANT.json`` via
``PHASE1_REWARD_VARIANT_FILE``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_selection() -> Dict[str, Any]:
    path = os.environ.get("PHASE1_REWARD_VARIANT_FILE", "").strip()
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    vid = os.environ.get("PHASE1_REWARD_VARIANT", "").strip()
    if not vid:
        return {}
    # Minimal selection payload when only the id is provided.
    presets = {
        "A4_current": {"selected": "A4_current", "family": "A4",
                       "epsilon": 0.02, "success_flat": False,
                       "train_policy": "reward_ablation_A4_GATED_VERIFIABLE"},
        "A4_success_flat": {"selected": "A4_success_flat", "family": "A4",
                            "epsilon": 0.02, "success_flat": True,
                            "train_policy": "reward_ablation_A4_GATED_VERIFIABLE"},
        "A4_eps_0.01": {"selected": "A4_eps_0.01", "family": "A4",
                        "epsilon": 0.01, "success_flat": False,
                        "train_policy": "reward_ablation_A4_GATED_VERIFIABLE"},
        "A4_eps_0.005": {"selected": "A4_eps_0.005", "family": "A4",
                         "epsilon": 0.005, "success_flat": False,
                         "train_policy": "reward_ablation_A4_GATED_VERIFIABLE"},
        "A1_outcome_only": {"selected": "A1_outcome_only", "family": "A1",
                            "epsilon": 0.0, "success_flat": False,
                            "train_policy": "reward_ablation_A1_OUTCOME_ONLY"},
    }
    if vid not in presets:
        raise ValueError(f"unknown PHASE1_REWARD_VARIANT={vid!r}")
    return presets[vid]


def apply_phase1_reward_variant(selection: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
    """Patch the in-process reward registry. Idempotent per process."""
    sel = selection if selection is not None else _load_selection()
    if not sel or not sel.get("selected"):
        print("[phase1_reward] no variant selected — leaving registry unchanged",
              flush=True)
        return sel

    # Import AFTER sys.path has been set by the trainer entry point.
    from lib import reward_ablation_registry as R  # noqa: WPS433

    vid = str(sel["selected"])
    eps = float(sel.get("epsilon", R.EPSILONS.get("A4_GATED_VERIFIABLE", 0.02)))
    success_flat = bool(sel.get("success_flat"))

    if sel.get("family") == "A4" or vid.startswith("A4"):
        R.EPSILONS["A4_GATED_VERIFIABLE"] = eps
        print(f"[phase1_reward] A4 epsilon -> {eps}", flush=True)

        if success_flat:
            if getattr(R, "_PHASE1_SUCCESS_FLAT_PATCHED", False):
                print("[phase1_reward] success-flat already patched", flush=True)
            else:
                _orig = R.score_arm

                def _score_arm(arm_id, trajectory, task, **kwargs):
                    score = _orig(arm_id, trajectory, task, **kwargs)
                    if (arm_id == "A4_GATED_VERIFIABLE"
                            and score.terminal_class == "official_success"):
                        return R.ArmScore(
                            reward_id=score.reward_id,
                            terminal_class=score.terminal_class,
                            terminal_score=score.terminal_score,
                            process_score=0.0,
                            epsilon=score.epsilon,
                            total_reward=float(score.terminal_score),
                            components={**(score.components or {}),
                                        "success_flat": True,
                                        "raw_process_score": score.process_score},
                        )
                    return score

                R.score_arm = _score_arm  # type: ignore[assignment]
                R._PHASE1_SUCCESS_FLAT_PATCHED = True
                print("[phase1_reward] A4 success-flat process tie-break enabled",
                      flush=True)

    print(f"[phase1_reward] active variant={vid} "
          f"train_policy={sel.get('train_policy')}", flush=True)
    os.environ["PHASE1_REWARD_VARIANT_ACTIVE"] = vid
    return sel


def install_sitecustomize_hint() -> str:
    """Return the import line a launcher should exec before training."""
    return (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
            f"import phase1_reward_patch as _p; _p.apply_phase1_reward_variant()")


if __name__ == "__main__":
    sel = apply_phase1_reward_variant()
    print(json.dumps(sel, indent=2))
