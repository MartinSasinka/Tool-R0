"""Unit tests for the reward ablation layer.

Covers (per reports/reward_ablation/ABLATION_PLAN.md §17):
  - terminal ordering (explicit, tested)
  - epsilon-band safety invariant
  - valid shorter path is not auto-penalized (no gold-call-count penalty)
  - alternative valid path scores as success
  - process-score normalization into [0, 1]
  - verifiable components never compare to the gold trace
  - gated process (A4) zero unless fully executable
  - deterministic train/eval subset selection
  - manifest hashes match the frozen files on disk
  - group dead/mixed metrics integrate with the ablation's episode rewards
  - identical non-reward config across arms (once configs exist)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_SCRIPTS = _V3 / "scripts"
_REPO = _V3.parents[1]
_MIN = _V3.parents[0] / "nestful_mtgrpo_minimal"

# _V3 inserted LAST (forced to sys.path[0]) regardless of what other test
# modules already put on sys.path first — `lib.*` must resolve to the real
# package (experiments/.../lib/) and never to scripts/lib/__init__.py.
for p in (str(_MIN), str(_SCRIPTS), str(_V3)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from lib import reward_ablation_registry as R  # noqa: E402
from lib import verifiable_process_reward as VP  # noqa: E402
from rollout import Trajectory, Turn  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Pure reward-math invariants (no trajectory objects needed)
# ─────────────────────────────────────────────────────────────────────────

def test_terminal_ordering_is_explicit_and_strict():
    assert R.TERMINAL_CLASSES == (
        "official_success", "executable_wrong_result", "executable_partial",
        "execution_failure", "parse_or_no_call",
    )
    for arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
        scalars = [R.TERMINAL_SCALARS[arm][c] for c in R.TERMINAL_CLASSES]
        assert scalars == sorted(scalars, reverse=True), f"{arm} terminal scalars not strictly descending"
        assert len(set(scalars)) == len(scalars), f"{arm} has tied terminal scalars"


def test_epsilon_band_safety_invariant():
    for arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
        assert R.verify_epsilon_safety(arm), f"epsilon-band-safety violated for {arm}"
        gap = R.min_adjacent_gap(R.TERMINAL_SCALARS[arm])
        eps = R.EPSILONS[arm]
        # P_max - P_min == 1 (process_score normalized to [0,1])
        assert eps * 1.0 < gap, f"{arm}: eps={eps} not < min_gap={gap}"


def test_no_process_component_can_flip_terminal_order():
    """For every arm, the WORST possible total_reward of a higher class must
    exceed the BEST possible total_reward of the next lower class."""
    for arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
        scalars = R.TERMINAL_SCALARS[arm]
        eps = R.EPSILONS[arm]
        for i in range(len(R.TERMINAL_CLASSES) - 1):
            hi_cls, lo_cls = R.TERMINAL_CLASSES[i], R.TERMINAL_CLASSES[i + 1]
            worst_hi = scalars[hi_cls] + eps * 0.0
            best_lo = scalars[lo_cls] + eps * 1.0
            assert worst_hi > best_lo, f"{arm}: {hi_cls} can be beaten by {lo_cls}"


def test_a1_has_no_process_tie_break():
    assert R.EPSILONS["A1_OUTCOME_ONLY"] == 0.0


def test_unified_terminal_class_no_gold_call_count_penalty():
    """A trajectory that is executable and produced FEWER calls than gold
    must still be classified purely by outcome (success/executable),
    never downgraded merely for predicted_call_count < gold_call_count."""
    pred_success_short = {
        "final_pass": True, "executable_frac": 1.0, "is_executable": True,
        "refs": 1.0, "parse_err": False, "clipped": False, "no_tool": False,
        "invalid_ref": False, "n_success": 1,
    }
    # success is determined by `is_success`, not by comparing n_pred to gold_n
    # (the function signature never even receives n_pred/gold_n).
    cls = R.unified_terminal_class(pred_success_short, is_success=True)
    assert cls == "official_success"

    pred_exec_wrong_short = dict(pred_success_short)
    cls2 = R.unified_terminal_class(pred_exec_wrong_short, is_success=False)
    assert cls2 == "executable_wrong_result"
    import inspect
    src = inspect.getsource(R.unified_terminal_class)
    assert "gold_n" not in src and "num_calls" not in src and "len(gold_calls" not in src


def test_alternative_valid_path_scores_as_success():
    """Any trajectory for which `is_success` is True lands in
    official_success regardless of which/how-many calls it made."""
    pred = {
        "final_pass": True, "executable_frac": 0.5, "is_executable": False,
        "refs": None, "parse_err": False, "clipped": False, "no_tool": False,
        "invalid_ref": False, "n_success": 1,
    }
    assert R.unified_terminal_class(pred, is_success=True) == "official_success"


def test_execution_failure_vs_partial_split():
    partial = {
        "final_pass": False, "executable_frac": 0.4, "is_executable": False,
        "refs": None, "parse_err": False, "clipped": False, "no_tool": False,
        "invalid_ref": False, "n_success": 1,
    }
    failure = dict(partial, executable_frac=0.0)
    assert R.unified_terminal_class(partial, is_success=False) == "executable_partial"
    assert R.unified_terminal_class(failure, is_success=False) == "execution_failure"


def test_parse_or_no_call_dominates_execution_predicates():
    pred = {
        "final_pass": False, "executable_frac": 1.0, "is_executable": True,
        "refs": None, "parse_err": True, "clipped": False, "no_tool": False,
        "invalid_ref": False, "n_success": 3,
    }
    assert R.unified_terminal_class(pred, is_success=False) == "parse_or_no_call"


# ─────────────────────────────────────────────────────────────────────────
# Verifiable process components (A3/A4): must be gold-free
# ─────────────────────────────────────────────────────────────────────────

def _turn(parsed_call, fail_reason=None, observation=None):
    return Turn(turn_idx=0, model_text="", parsed_call=parsed_call,
                observation=observation, fail_reason=fail_reason)


def _traj(turns, final_observation=None, clipped_any=False, stop_reason=None):
    return Trajectory(task_id="t", stage=3, gold_num_turns=len(turns), turns=turns,
                       final_observation=final_observation, executor_mode="synthetic",
                       clipped_any=clipped_any, stop_reason=stop_reason)


def test_verifiable_process_normalized_to_unit_interval():
    task = {
        "tools": [{
            "name": "add_numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }],
    }
    turns = [_turn({"name": "add_numbers", "arguments": {"a": 1, "b": 2}})]
    traj = _traj(turns)
    pred = {"parse_err": False, "clipped": False, "refs": None, "executable_frac": 1.0}
    comps = VP.verifiable_process_components(traj, task, pred)
    for k, v in comps.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} not in [0,1]"
    score = VP.verifiable_process_score(comps)
    assert 0.0 <= score <= 1.0


def test_verifiable_components_reject_unknown_tool_and_bad_schema():
    task = {
        "tools": [{
            "name": "add_numbers",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]},
        }],
    }
    turns = [_turn({"name": "not_a_real_tool", "arguments": {"a": 1, "z": 9}})]
    traj = _traj(turns)
    assert VP.tool_exists_frac(traj, task) == 0.0
    assert VP.schema_keys_valid_frac(traj, task) == 0.0


def test_verifiable_components_are_gold_free():
    """The SAME trajectory must score identically for A3 regardless of what
    the task's gold_calls/gold_answer say — verifiable process must never
    reward gold-trace similarity."""
    tools = [{
        "name": "add_numbers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }]
    turns = [_turn({"name": "add_numbers", "arguments": {"a": 1, "b": 2}}, fail_reason=None)]
    traj = _traj(turns)
    pred = {"parse_err": False, "clipped": False, "refs": None, "executable_frac": 1.0}

    task_a = {"tools": tools, "gold_calls": [{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}]}
    task_b = {"tools": tools, "gold_calls": [{"name": "subtract_numbers", "arguments": {"a": 999, "b": -5}}]}

    comps_a = VP.verifiable_process_components(traj, task_a, pred)
    comps_b = VP.verifiable_process_components(traj, task_b, pred)
    assert comps_a == comps_b, "verifiable components must not depend on gold_calls content"


def test_gate_open_requires_fully_executable():
    ok = {"parse_err": False, "clipped": False, "no_tool": False, "executable_frac": 1.0}
    bad_parse = dict(ok, parse_err=True)
    bad_exec = dict(ok, executable_frac=0.0)
    assert VP.gate_open(ok) is True
    assert VP.gate_open(bad_parse) is False
    assert VP.gate_open(bad_exec) is False


def test_a4_process_score_zero_when_gate_closed():
    tools = [{"name": "add_numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}]
    turns = [_turn(None, fail_reason=None)]  # no parsed call at all -> no_tool
    traj = _traj(turns)
    task = {"tools": tools, "gold_calls": []}
    pred = {
        "final_pass": False, "executable_frac": 0.0, "is_executable": False,
        "refs": None, "parse_err": False, "clipped": False, "no_tool": True,
        "invalid_ref": False, "n_success": 0,
    }
    score = R.score_arm("A4_GATED_VERIFIABLE", traj, task, official_win=False)
    assert score.process_score == 0.0
    assert score.components.get("gate_open") is False


# ─────────────────────────────────────────────────────────────────────────
# A0 unchanged / read-only projection
# ─────────────────────────────────────────────────────────────────────────

def test_a0_matches_production_reward_unmodified():
    from lib.reward_v3_2_dense import execution_aware_v3_2_dense
    tools = [{"name": "add_numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}]
    turns = [_turn({"name": "add_numbers", "arguments": {"a": 1, "b": 2}}, fail_reason=None,
                   observation={"result": 3})]
    traj = _traj(turns, final_observation={"result": 3})
    task = {
        "tools": tools,
        "gold_calls": [{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}],
        "num_calls": 1,
        "stage": "stage3_3call_agentic_openrouter",
    }
    direct = execution_aware_v3_2_dense(traj, task, train_stage=3)
    via_registry = R.score_arm("A0_R0_CURRENT", traj, task, train_stage=3)
    assert via_registry.total_reward == pytest.approx(direct.reward)


# ─────────────────────────────────────────────────────────────────────────
# Dataset selection determinism + manifest hash integrity
# ─────────────────────────────────────────────────────────────────────────

DATA_DIR = _V3 / "reports" / "reward_ablation" / "data"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file_lf(path: Path) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.skipif(not (DATA_DIR / "train_subset_manifest.json").is_file(), reason="train subset not prepared yet")
def test_train_subset_manifest_hash_matches_file():
    manifest = json.loads((DATA_DIR / "train_subset_manifest.json").read_text(encoding="utf-8"))
    actual = _sha256_file(DATA_DIR / "train_subset_160.jsonl")
    assert actual == manifest["subset_sha256"]
    assert manifest["n_selected"] == 160
    assert len(manifest["selected_task_ids"]) == len(set(manifest["selected_task_ids"])) == 160


@pytest.mark.skipif(not (DATA_DIR / "nestful_diagnostic_500_manifest.json").is_file(), reason="eval subset not prepared yet")
def test_eval_subset_manifest_hash_matches_file():
    manifest = json.loads((DATA_DIR / "nestful_diagnostic_500_manifest.json").read_text(encoding="utf-8"))
    ids_doc = json.loads((DATA_DIR / "nestful_diagnostic_500_ids.json").read_text(encoding="utf-8"))
    actual = _sha256_file_lf(DATA_DIR / "nestful_diagnostic_500_ids.json")
    assert actual == manifest["ids_file_sha256"]
    assert len(ids_doc["task_ids"]) == len(set(ids_doc["task_ids"])) == 500


@pytest.mark.slow
def test_train_subset_selection_is_deterministic_rerun():
    script = _SCRIPTS / "ablation" / "prepare_train_subset_160.py"
    before = json.loads((DATA_DIR / "train_subset_manifest.json").read_text(encoding="utf-8"))
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(_REPO))
    after = json.loads((DATA_DIR / "train_subset_manifest.json").read_text(encoding="utf-8"))
    assert before["subset_sha256"] == after["subset_sha256"]
    assert before["selected_task_ids"] == after["selected_task_ids"]


@pytest.mark.slow
def test_eval_subset_selection_is_deterministic_rerun():
    script = _SCRIPTS / "ablation" / "prepare_nestful_diagnostic_500.py"
    before = json.loads((DATA_DIR / "nestful_diagnostic_500_ids.json").read_text(encoding="utf-8"))
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(_REPO))
    after = json.loads((DATA_DIR / "nestful_diagnostic_500_ids.json").read_text(encoding="utf-8"))
    assert before["task_ids"] == after["task_ids"]


# ─────────────────────────────────────────────────────────────────────────
# Group dead/mixed metrics integrate correctly with ablation episode rewards
# ─────────────────────────────────────────────────────────────────────────

def test_group_stats_dead_group_detection_on_ablation_rewards():
    from group_stats import compute_group_stats

    # 8 identical rollouts -> dead group regardless of which arm produced it
    identical = [[0.5] for _ in range(8)]
    stats_dead = compute_group_stats(identical, [0.5] * 8)
    assert stats_dead.dead_corrected is True

    # 8 rollouts spanning several unified terminal classes -> not dead
    mixed_rewards = [0.97, 0.02, 0.2, 0.115, 0.97, 0.02, 0.2, 0.115]
    mixed = [[r] for r in mixed_rewards]
    stats_mixed = compute_group_stats(mixed, mixed_rewards)
    assert stats_mixed.dead_corrected is False


# ─────────────────────────────────────────────────────────────────────────
# Identical non-reward config across arms
# ─────────────────────────────────────────────────────────────────────────

ARMS_DIR = _V3 / "configs" / "reward_ablation" / "arms"


@pytest.mark.skipif(not ARMS_DIR.is_dir(), reason="arm configs not created yet")
def test_arm_configs_only_differ_in_reward_and_wandb_keys():
    import yaml

    allowed_diff_top_keys = {"reward_id", "description", "reward", "wandb"}
    docs = {}
    for arm in R.ARM_IDS:
        path = ARMS_DIR / f"{arm}.yaml"
        assert path.is_file(), f"missing arm config {path}"
        docs[arm] = yaml.safe_load(path.read_text(encoding="utf-8"))

    for arm, doc in docs.items():
        extra_keys = set(doc.keys()) - allowed_diff_top_keys
        assert not extra_keys, f"{arm}.yaml has unexpected top-level keys: {extra_keys}"
        assert doc["reward_id"] == arm

    # every arm's train_policy must be unique (the one true experimental variable)
    policies = {arm: docs[arm]["reward"]["train_policy"] for arm in R.ARM_IDS}
    assert len(set(policies.values())) == len(R.ARM_IDS), f"duplicate train_policy: {policies}"


def test_make_episode_turn_reward_seq_sparse_and_labeled():
    tools = [{"name": "add_numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}]
    turns = [_turn({"name": "add_numbers", "arguments": {"a": 1}}, fail_reason=None),
             _turn(None, fail_reason=None)]
    traj = _traj(turns)
    task = {"tools": tools, "gold_calls": [{"name": "add_numbers", "arguments": {"a": 1}}], "num_calls": 1}
    for arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
        fn = R.make_episode_turn_reward_seq(arm)
        assert fn.reward_policy == f"reward_ablation_{arm}"
        out = fn(traj, task)
        assert len(out["r_seq"]) == len(turns)
        assert all(x == 0.0 for x in out["r_seq"])
        assert out["diagnostics"]["reward_id"] == arm
