"""Unit tests for the reward-ablation CLI/summary/decision layer added on top
of the reward registry (run_reward_ablation.py, summarize_reward_ablation.py,
select_reward_arms.py). All tests are CPU-only / deterministic — they build
synthetic `final_eval_v5.py run` output fixtures instead of running a real
GPU eval, and synthetic training-diagnostics instead of a real GRPO run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_SCRIPTS_ABLATION = _V3 / "scripts" / "ablation"
# Order matters: _SCRIPTS_ABLATION must be importable (`import run_reward_ablation`)
# but _V3 must win sys.path[0] so `lib.*` resolves to the real package, not
# scripts/lib/__init__.py — see tests/test_reward_ablation.py for the bug this
# guards against. Force-reinsert both, _V3 last, regardless of prior state.
for p in (str(_SCRIPTS_ABLATION), str(_V3)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import run_reward_ablation as RRA  # noqa: E402
import select_reward_arms as SEL  # noqa: E402
import summarize_reward_ablation as SUM  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# run_reward_ablation.py: config building, hashing, resume-safety
# ─────────────────────────────────────────────────────────────────────────

def test_effective_config_only_differs_by_reward_train_policy():
    configs = {arm: RRA.load_effective_config(arm) for arm in RRA.ARM_IDS}
    for arm, cfg in configs.items():
        assert cfg["reward"]["train_policy"] == RRA.TRAIN_POLICY[arm]

    def _strip_reward_and_wandb(cfg):
        c = json.loads(json.dumps(cfg))
        c.pop("reward", None)
        c.pop("wandb", None)
        c.pop("reward_id", None)
        c.pop("description", None)
        return c

    baseline = _strip_reward_and_wandb(configs["A0_R0_CURRENT"])
    for arm in RRA.ARM_IDS:
        assert _strip_reward_and_wandb(configs[arm]) == baseline, f"{arm} config differs beyond reward.train_policy"


def test_build_experiment_id_deterministic_when_run_id_given():
    eid = RRA.build_experiment_id("A2_R3_OUTCOME_FIRST", 1, 20260724, "my-fixed-id")
    assert eid == "my-fixed-id"


def test_build_experiment_id_includes_arm_round_seed_when_auto():
    eid = RRA.build_experiment_id("A3_VERIFIABLE_PROCESS", 1, 20260724, None)
    assert "A3_VERIFIABLE_PROCESS" in eid and "r1" in eid and "seed20260724" in eid


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world", encoding="utf-8")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert RRA._sha256_file(p) == expected  # noqa: SLF001


def test_materialize_eval_subset_passthrough_for_jsonl(tmp_path):
    p = tmp_path / "already_materialized.jsonl"
    p.write_text('{"sample_id": "x"}\n', encoding="utf-8")
    assert RRA.materialize_eval_subset(p, tmp_path / "out") == p


def test_materialize_eval_subset_filters_by_id(tmp_path, monkeypatch):
    src = tmp_path / "nestful_test.jsonl"
    src.write_text(
        '{"sample_id": "a"}\n{"sample_id": "b"}\n{"sample_id": "c"}\n', encoding="utf-8")
    monkeypatch.setattr(RRA, "NESTFUL_TEST", src)
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"task_ids": ["a", "c"]}), encoding="utf-8")
    out = RRA.materialize_eval_subset(ids_path, tmp_path / "out")
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["sample_id"] for r in lines} == {"a", "c"}


def test_materialize_eval_subset_aborts_on_missing_ids(tmp_path, monkeypatch):
    src = tmp_path / "nestful_test.jsonl"
    src.write_text('{"sample_id": "a"}\n', encoding="utf-8")
    monkeypatch.setattr(RRA, "NESTFUL_TEST", src)
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"task_ids": ["a", "does-not-exist"]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        RRA.materialize_eval_subset(ids_path, tmp_path / "out")


def test_state_step_done_and_mark_step_roundtrip(tmp_path):
    state = RRA.load_state(tmp_path)
    assert not RRA.step_done(state, "train")
    RRA.mark_step(state, "train", checkpoint="/x")
    RRA.save_state(tmp_path, state)
    reloaded = RRA.load_state(tmp_path)
    assert RRA.step_done(reloaded, "train")
    assert reloaded["steps"]["train"]["checkpoint"] == "/x"


def test_assert_resume_compatible_blocks_arm_switch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"reward_arm": "A0_R0_CURRENT", "seed": 1}), encoding="utf-8")

    class _Args:
        reward_arm = "A1_OUTCOME_ONLY"
        seed = 1

    with pytest.raises(SystemExit):
        RRA.assert_resume_compatible(run_dir, _Args(), "eid")


def test_assert_resume_compatible_blocks_seed_switch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"reward_arm": "A0_R0_CURRENT", "seed": 1}), encoding="utf-8")

    class _Args:
        reward_arm = "A0_R0_CURRENT"
        seed = 2

    with pytest.raises(SystemExit):
        RRA.assert_resume_compatible(run_dir, _Args(), "eid")


def test_assert_resume_compatible_allows_same_arm_seed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"reward_arm": "A0_R0_CURRENT", "seed": 1}), encoding="utf-8")

    class _Args:
        reward_arm = "A0_R0_CURRENT"
        seed = 1

    RRA.assert_resume_compatible(run_dir, _Args(), "eid")  # must not raise


def test_smoke_config_caps_tasks_and_generations():
    class _Args:
        reward_arm = "A0_R0_CURRENT"
        seed = 1
        train_subset = RRA.DEFAULT_TRAIN_SUBSET
        eval_subset = RRA.DEFAULT_EVAL_SUBSET_IDS
        round = 1
        smoke = True

    cfg = RRA.build_run_config(_Args())
    assert cfg["smoke"]["enabled"] is True
    assert cfg["smoke"]["max_train_tasks"] == 8
    assert cfg["training"]["num_generations"] == 8


# ─────────────────────────────────────────────────────────────────────────
# summarize_reward_ablation.py: synthetic final_eval_v5-shaped fixtures
# ─────────────────────────────────────────────────────────────────────────

def _write_eval_fixture(out_dir: Path, wins: dict, extra_official: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_eval_trajectories.jsonl", "w", encoding="utf-8") as fh:
        for tid, win in wins.items():
            row = {
                "sample_id": tid,
                "num_gold_calls": 3,
                "internal_f1_func": 1.0 if win else 0.3,
                "internal_f1_param": 1.0 if win else 0.2,
                "_traj": {"official_win": bool(win), "official_full_match": bool(win),
                          "num_tool_calls": 3 if win else 1, "executable": bool(win)},
            }
            fh.write(json.dumps(row) + "\n")
    n = len(wins)
    win_rate = sum(1 for w in wins.values() if w) / n if n else 0.0
    official = {"win_rate": win_rate, "f1_func": win_rate, "f1_param": win_rate,
                "full_sequence_accuracy": win_rate}
    official.update(extra_official or {})
    (out_dir / "metrics_official.json").write_text(json.dumps(official), encoding="utf-8")


def test_summarize_arm_writes_all_required_deliverables(tmp_path):
    c0_dir = tmp_path / "c0"
    _write_eval_fixture(c0_dir, {"t1": True, "t2": False, "t3": False, "t4": True})
    arm_dir = tmp_path / "arm"
    _write_eval_fixture(arm_dir, {"t1": True, "t2": True, "t3": False, "t4": True})

    SUM.summarize_arm("A2_R3_OUTCOME_FIRST", arm_dir, None, c0_dir, None)

    for fn in ("task_results.jsonl", "metrics.json", "metrics.md", "paired_vs_c0.json",
               "failure_taxonomy.csv", "bucket_metrics.csv"):
        assert (arm_dir / fn).is_file(), f"missing {fn}"

    paired = json.loads((arm_dir / "paired_vs_c0.json").read_text(encoding="utf-8"))
    assert paired["n_gained"] == 1  # t2: 0 -> 1
    assert paired["n_regressed"] == 0
    assert "mcnemar" in paired


def test_summarize_arm_skips_r0_pairing_for_a0_itself(tmp_path):
    c0_dir = tmp_path / "c0"
    _write_eval_fixture(c0_dir, {"t1": True, "t2": False})
    a0_dir = tmp_path / "a0"
    _write_eval_fixture(a0_dir, {"t1": True, "t2": False})
    SUM.summarize_arm("A0_R0_CURRENT", a0_dir, None, c0_dir, a0_dir)
    assert not (a0_dir / "paired_vs_r0.json").is_file()


def test_mcnemar_symmetric_case_yields_high_p_value():
    """b == c (as many regressions as gains) -> the continuity-corrected
    chi2 approximation is conservative for tiny n but must still land in
    "not significant" territory (p > 0.3), and get closer to 1.0 as the
    discordant count grows while staying symmetric."""
    base_small = {"t1": {"_traj": {"official_win": True}}, "t2": {"_traj": {"official_win": False}}}
    cand_small = {"t1": {"_traj": {"official_win": False}}, "t2": {"_traj": {"official_win": True}}}
    small = SUM._mcnemar(base_small, cand_small)  # noqa: SLF001
    assert small["b_base_win_cand_loss"] == 1
    assert small["c_base_loss_cand_win"] == 1
    assert small["p_value"] > 0.3

    base_large = {f"t{i}": {"_traj": {"official_win": i % 2 == 0}} for i in range(40)}
    cand_large = {f"t{i}": {"_traj": {"official_win": i % 2 == 1}} for i in range(40)}
    large = SUM._mcnemar(base_large, cand_large)  # noqa: SLF001
    assert large["b_base_win_cand_loss"] == large["c_base_loss_cand_win"] == 20
    assert large["p_value"] > 0.85
    assert large["p_value"] > small["p_value"]


def test_round_summary_aggregates_all_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(SUM, "REPORTS_DIR", tmp_path / "reports")
    arm_dirs = {}
    for arm, wins in (("A0_R0_CURRENT", {"t1": True, "t2": False}),
                       ("A2_R3_OUTCOME_FIRST", {"t1": True, "t2": True})):
        d = tmp_path / arm
        _write_eval_fixture(d, wins)
        SUM.summarize_arm(arm, d, None, tmp_path / "A0_R0_CURRENT", None)
        arm_dirs[arm] = d
    summary = SUM.round_summary(1, arm_dirs)
    assert set(summary["arms"].keys()) == {"A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST"}
    assert (tmp_path / "reports" / "round1" / "ROUND1_SUMMARY.json").is_file()
    assert (tmp_path / "reports" / "round1" / "ROUND1_SUMMARY.md").is_file()


# ─────────────────────────────────────────────────────────────────────────
# select_reward_arms.py: hard gates + lexicographic ranking
# ─────────────────────────────────────────────────────────────────────────

def _clean_training_diag(**overrides):
    base = {
        "terminal_inversions": 0,
        "nan_or_inf_detected": False,
        "official_loss_beats_success_in_group": False,
        "dead_group_rate": 0.05,
        "control_dead_group_rate": 0.05,
        "parse_rate": 0.98,
        "control_parse_rate": 0.98,
        "synthetic_terminal_success_rate": 0.5,
        "reward_up_but_terminal_success_down": False,
        "reward_hacking_suspected": False,
    }
    base.update(overrides)
    return base


def test_evaluate_gates_pass_when_clean():
    entry = {"metrics": {"diagnostics": {"executable_rate": 0.9}}}
    control_entry = {"metrics": {"diagnostics": {"executable_rate": 0.9}}}
    g = SEL.evaluate_gates("A2_R3_OUTCOME_FIRST", entry, control_entry, _clean_training_diag())
    assert g["verdict"] == "PASS"
    assert g["reasons"] == []


def test_evaluate_gates_fail_on_terminal_inversion():
    entry = {"metrics": {"diagnostics": {}}}
    g = SEL.evaluate_gates("A3_VERIFIABLE_PROCESS", entry, {},
                            _clean_training_diag(terminal_inversions=2))
    assert g["verdict"] == "FAIL"
    assert any("terminal_inversion" in r for r in g["reasons"])


def test_evaluate_gates_fail_on_nan_inf():
    entry = {"metrics": {"diagnostics": {}}}
    g = SEL.evaluate_gates("A1_OUTCOME_ONLY", entry, {}, _clean_training_diag(nan_or_inf_detected=True))
    assert g["verdict"] == "FAIL"


def test_evaluate_gates_a1_dead_group_becomes_conditional_not_fail():
    entry = {"metrics": {"diagnostics": {}}}
    g = SEL.evaluate_gates("A1_OUTCOME_ONLY", entry, {},
                            _clean_training_diag(dead_group_rate=0.5, control_dead_group_rate=0.05))
    assert g["verdict"] == "CONDITIONAL"


def test_evaluate_gates_non_control_dead_group_fails():
    entry = {"metrics": {"diagnostics": {}}}
    g = SEL.evaluate_gates("A3_VERIFIABLE_PROCESS", entry, {},
                            _clean_training_diag(dead_group_rate=0.5, control_dead_group_rate=0.05))
    assert g["verdict"] == "FAIL"


def test_rank_arms_prioritizes_synthetic_success_first():
    summary = {"arms": {
        "A2_R3_OUTCOME_FIRST": {"metrics": {"official": {"win_rate": 0.1}}, "paired_vs_c0": {}},
        "A3_VERIFIABLE_PROCESS": {"metrics": {"official": {"win_rate": 0.9}}, "paired_vs_c0": {}},
    }}
    gate_results = {
        "A2_R3_OUTCOME_FIRST": {"verdict": "PASS"},
        "A3_VERIFIABLE_PROCESS": {"verdict": "PASS"},
    }
    training_diag = {
        "A2_R3_OUTCOME_FIRST": {"synthetic_terminal_success_rate": 0.9},
        "A3_VERIFIABLE_PROCESS": {"synthetic_terminal_success_rate": 0.5},
    }
    ranked = SEL.rank_arms(gate_results, summary, training_diag)
    # A2 wins despite lower NESTFUL win_rate because synthetic success is criterion #1.
    assert ranked[0] == "A2_R3_OUTCOME_FIRST"


def test_rank_arms_excludes_failed_and_a1():
    summary = {"arms": {
        "A1_OUTCOME_ONLY": {"metrics": {"official": {"win_rate": 0.9}}, "paired_vs_c0": {}},
        "A2_R3_OUTCOME_FIRST": {"metrics": {"official": {"win_rate": 0.5}}, "paired_vs_c0": {}},
        "A4_GATED_VERIFIABLE": {"metrics": {"official": {"win_rate": 0.5}}, "paired_vs_c0": {}},
    }}
    gate_results = {
        "A1_OUTCOME_ONLY": {"verdict": "CONDITIONAL"},
        "A2_R3_OUTCOME_FIRST": {"verdict": "PASS"},
        "A4_GATED_VERIFIABLE": {"verdict": "FAIL"},
    }
    ranked = SEL.rank_arms(gate_results, summary, {})
    assert ranked == ["A2_R3_OUTCOME_FIRST"]


def test_build_round2_plan_includes_control_plus_top2_and_is_not_auto_launched():
    ranked = ["A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"]
    gate_results = {a: {"verdict": "PASS"} for a in ranked}
    plan = SEL.build_round2_plan(ranked, gate_results)
    assert plan["not_auto_launched"] is True
    assert plan["arms"] == ["A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS"]
    assert plan["seed"] == 20260725
    assert len(plan["commands"]) == 3
    assert all("--round 2" in c and "--seed 20260725" in c for c in plan["commands"])


def test_build_round2_plan_only_takes_passing_arms():
    ranked = ["A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS"]
    gate_results = {"A2_R3_OUTCOME_FIRST": {"verdict": "PASS"}, "A3_VERIFIABLE_PROCESS": {"verdict": "CONDITIONAL"}}
    plan = SEL.build_round2_plan(ranked, gate_results)
    assert plan["arms"] == ["A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST"]
