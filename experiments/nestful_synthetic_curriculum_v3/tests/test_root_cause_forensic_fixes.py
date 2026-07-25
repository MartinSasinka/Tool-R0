#!/usr/bin/env python3
"""Regression tests for the bugs found by the 2026-07-25 root-cause forensic
audit (reports/root_cause_forensic/). No GPU required.

Protected bugs:
  1. Reward-dispatch override: a defaulted REWARD_POLICY env var must NEVER
     overwrite an explicitly configured reward_ablation_* train policy
     (Round-1 reward ablation 2026-07-24 trained all 5 arms with A0's
     execution_aware_v3_2_dense because of this).
  2. run_reward_ablation.assert_dispatched_policy hard guard.
  3. Offline-audit discovery must flag declared-vs-logged reward mismatch.
  4. Offline-audit verdict must short-circuit to REWARD_DISPATCH_BUG.
  5. Adapter similarity must be judged on delta-to-init (B@A), not on raw
     flattened weights (shared seeded lora_A init makes raw cosine ~1.0).
  6. SYNTHETIC_SUCCESS_REWARD must equal the v3_2_dense fully_correct band
     lower bound (0.90), and no other class band may reach it.
  7. Forensic-audit copy of _turn_returns stays in parity with the trainer.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_V3 = Path(__file__).resolve().parents[1]
_MINIMAL = _V3.parent / "nestful_mtgrpo_minimal"
_PARTIAL = _V3.parent / "nestful_mtgrpo_partial"
for p in (str(_V3), str(_PARTIAL), str(_MINIMAL)):
    if p in sys.path:
        sys.path.remove(p)
for p in (str(_MINIMAL), str(_PARTIAL), str(_V3)):
    sys.path.insert(0, p)
sys.path.insert(0, str(_V3 / "scripts" / "audit" / "root_cause_forensic"))


def _import_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class TestRewardDispatchFix(unittest.TestCase):
    """Bug 1: config reward_ablation_* policy must survive REWARD_POLICY env."""

    @classmethod
    def setUpClass(cls):
        cls._env_backup = {k: os.environ.get(k) for k in ("REWARD_POLICY", "REWARD_NAME")}
        cls.v3_run = _import_by_path(_V3 / "run.py", "v3_run_forensic_test")
        cls.v3_run._hook_select_train_reward()

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_ablation_policy_survives_env_default(self):
        # exact poison from two_phase_train_session.py module import
        os.environ["REWARD_POLICY"] = "execution_aware_v3_2_dense"
        os.environ.pop("REWARD_NAME", None)
        config = {"reward": {"train_policy": "reward_ablation_A1_OUTCOME_ONLY"}}
        self.v3_run._partial._select_train_reward(config)
        self.assertEqual(config["reward"]["train_policy"],
                         "reward_ablation_A1_OUTCOME_ONLY")
        import grpo_train
        fn = grpo_train.episode_turn_reward_seq
        self.assertEqual(getattr(fn, "reward_policy", None),
                         "reward_ablation_A1_OUTCOME_ONLY")

    def test_every_arm_dispatches_itself(self):
        os.environ["REWARD_POLICY"] = "execution_aware_v3_2_dense"
        from lib import reward_ablation_registry
        import grpo_train
        for arm in reward_ablation_registry.ARM_IDS:
            if arm == "A0_R0_CURRENT":
                continue  # A0 IS execution_aware_v3_2_dense by design
            config = {"reward": {"train_policy": f"reward_ablation_{arm}"}}
            self.v3_run._partial._select_train_reward(config)
            self.assertEqual(config["reward"]["train_policy"], f"reward_ablation_{arm}")
            self.assertEqual(
                getattr(grpo_train.episode_turn_reward_seq, "reward_policy", None),
                f"reward_ablation_{arm}")

    def test_v3_2_dense_branch_unchanged(self):
        os.environ.pop("REWARD_POLICY", None)
        os.environ.pop("REWARD_NAME", None)
        config = {"reward": {"train_policy": "execution_aware_v3_2_dense"}}
        self.v3_run._partial._select_train_reward(config)
        self.assertEqual(config["reward"]["train_policy"], "execution_aware_v3_2_dense")
        import grpo_train
        self.assertEqual(
            getattr(grpo_train.episode_turn_reward_seq, "reward_policy", None),
            "execution_aware_v3_2_dense")

    def test_unknown_ablation_arm_raises(self):
        config = {"reward": {"train_policy": "reward_ablation_BOGUS_ARM"}}
        with self.assertRaises(ValueError):
            self.v3_run._partial._select_train_reward(config)


class TestRunnerGuard(unittest.TestCase):
    """Bug 2: hard post-init assertion in the ablation runner."""

    @classmethod
    def setUpClass(cls):
        cls.runner = _import_by_path(
            _V3 / "scripts" / "ablation" / "run_reward_ablation.py",
            "run_reward_ablation_forensic_test")

    def test_matching_policy_passes(self):
        cfg = {"reward": {"train_policy": "reward_ablation_A1_OUTCOME_ONLY"}}
        self.runner.assert_dispatched_policy(cfg, "reward_ablation_A1_OUTCOME_ONLY")

    def test_overridden_policy_aborts(self):
        cfg = {"reward": {"train_policy": "execution_aware_v3_2_dense"}}
        with self.assertRaises(SystemExit):
            self.runner.assert_dispatched_policy(cfg, "reward_ablation_A1_OUTCOME_ONLY")


class TestDiscoveryDispatchCheck(unittest.TestCase):
    """Bug 3: offline audit discovery flags declared-vs-logged mismatch."""

    def _write_log(self, tmp: Path, resolved: str) -> Path:
        p = tmp / "train_log.jsonl"
        rows = [
            {"reward_dispatch": {"configured_policy": resolved,
                                 "resolved_policy": resolved}},
            {"task_id": "t0", "reward_policy_resolved": resolved,
             "reward_train_policy": resolved,
             "episode_rewards": [0.1], "turn_rewards": [[0.1]]},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def test_mismatch_flagged(self):
        from lib.offline_audit.discovery import _reward_dispatch_check
        with tempfile.TemporaryDirectory() as td:
            p = self._write_log(Path(td), "execution_aware_v3_2_dense")
            out = _reward_dispatch_check("A1_OUTCOME_ONLY", p)
        self.assertIs(out["dispatch_ok"], False)
        self.assertIn("execution_aware_v3_2_dense", out["logged_policies"])

    def test_correct_dispatch_ok(self):
        from lib.offline_audit.discovery import _reward_dispatch_check
        with tempfile.TemporaryDirectory() as td:
            p = self._write_log(Path(td), "reward_ablation_A1_OUTCOME_ONLY")
            out = _reward_dispatch_check("A1_OUTCOME_ONLY", p)
        self.assertIs(out["dispatch_ok"], True)

    def test_a0_control_expects_v3_2(self):
        from lib.offline_audit.discovery import _reward_dispatch_check
        with tempfile.TemporaryDirectory() as td:
            p = self._write_log(Path(td), "execution_aware_v3_2_dense")
            out = _reward_dispatch_check("A0_R0_CURRENT", p)
        self.assertIs(out["dispatch_ok"], True)

    def test_no_policy_info_is_none(self):
        from lib.offline_audit.discovery import _reward_dispatch_check
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train_log.jsonl"
            p.write_text(json.dumps({"task_id": "t0"}) + "\n", encoding="utf-8")
            out = _reward_dispatch_check("A1_OUTCOME_ONLY", p)
        self.assertIsNone(out["dispatch_ok"])


class TestVerdictShortCircuit(unittest.TestCase):
    """Bug 4: dispatch mismatch dominates every other verdict."""

    def _ctx(self):
        return {
            "discovery": {"arms": {"A1_OUTCOME_ONLY": {"reward_dispatch": {
                "dispatch_ok": False,
                "expected_policy": "reward_ablation_A1_OUTCOME_ONLY",
                "logged_policies": ["execution_aware_v3_2_dense"],
            }}}},
            "pairwise": {"status": {"mode": "PARTIAL"}, "pairs": [{
                "arm_a": "A0_R0_CURRENT", "arm_b": "A4_GATED_VERIFIABLE",
                "n_hash_matched_rollouts": 35,
                "reward_pearson_hash_matched": 0.9999999999999998,
                "reward_spearman_hash_matched": 0.9999999999999999,
                "advantage_cosine_hash_matched": 0.49,
            }]},
        }

    def test_dispatch_bug_verdict(self):
        from lib.offline_audit.verdict import decide_verdict
        v = decide_verdict(self._ctx())
        self.assertEqual(v["verdict"], "REWARD_DISPATCH_BUG")
        self.assertTrue(v["rewards_identical_on_hash_matched"])

    def test_identical_rewards_detected(self):
        from lib.offline_audit.verdict import _rewards_identical_on_hash_matched
        self.assertTrue(_rewards_identical_on_hash_matched(self._ctx()))
        ctx2 = self._ctx()
        ctx2["pairwise"]["pairs"][0]["reward_pearson_hash_matched"] = 0.82
        self.assertFalse(_rewards_identical_on_hash_matched(ctx2))


class TestAdapterDeltaMetric(unittest.TestCase):
    """Bug 5: raw flat cosine is init-dominated; delta (B@A) must be primary."""

    def test_shared_A_init_masks_opposite_updates(self):
        import torch
        from lib.offline_audit.adapters import _delta_dot, _delta_norm2
        torch.manual_seed(0)
        A = torch.randn(4, 64, dtype=torch.float64)          # shared seeded init
        B1 = 0.001 * torch.randn(32, 4, dtype=torch.float64)  # tiny update arm 1
        B2 = -B1                                              # OPPOSITE update arm 2
        mods1 = {"m": (A, B1)}
        mods2 = {"m": (A, B2)}
        n1 = _delta_norm2(mods1) ** 0.5
        n2 = _delta_norm2(mods2) ** 0.5
        cos_delta = _delta_dot(mods1, mods2) / (n1 * n2)
        self.assertLess(cos_delta, -0.99)  # updates are opposite
        # ...yet raw flattened-weights cosine says "nearly identical":
        f1 = torch.cat([A.reshape(-1), B1.reshape(-1)])
        f2 = torch.cat([A.reshape(-1), B2.reshape(-1)])
        cos_raw = float(torch.dot(f1, f2) / (f1.norm() * f2.norm()))
        self.assertGreater(cos_raw, 0.99)

    def test_delta_norm_matches_materialized(self):
        import torch
        from lib.offline_audit.adapters import _delta_norm2
        torch.manual_seed(1)
        A = torch.randn(4, 16, dtype=torch.float64)
        B = torch.randn(8, 4, dtype=torch.float64)
        self.assertAlmostEqual(_delta_norm2({"m": (A, B)}),
                               float(((B @ A) ** 2).sum()), places=8)


class TestSuccessThresholdDefinition(unittest.TestCase):
    """Bug 6: the reward-threshold success proxy must match v3_2_dense bands."""

    def test_threshold_equals_fully_correct_lower_bound(self):
        from lib.offline_audit import SYNTHETIC_SUCCESS_REWARD
        from lib.reward_v3_2_dense import BANDS
        self.assertEqual(SYNTHETIC_SUCCESS_REWARD, BANDS["fully_correct"][0])

    def test_no_other_class_reaches_threshold(self):
        from lib.offline_audit import SYNTHETIC_SUCCESS_REWARD
        from lib.reward_v3_2_dense import BANDS
        for cls, (_lo, hi) in BANDS.items():
            if cls == "fully_correct":
                continue
            self.assertLess(hi, SYNTHETIC_SUCCESS_REWARD, cls)


class TestTurnReturnsParity(unittest.TestCase):
    """Bug 7 guard: the forensic audit's self-contained copy of _turn_returns
    must stay byte-equivalent in behavior with the trainer's."""

    def test_parity_random_cases(self):
        import random
        from grpo_train import _turn_returns
        common = _import_by_path(
            _V3 / "scripts" / "audit" / "root_cause_forensic" / "common.py",
            "forensic_common_parity_test")
        rng = random.Random(7)
        for _ in range(50):
            n = rng.randint(1, 6)
            r_seq = [rng.uniform(0, 1) for _ in range(n)]
            R = rng.uniform(0, 1)
            gamma = rng.choice([0.0, 0.5, 1.0])
            lam = rng.choice([0.0, 0.5, 1.0])
            a = _turn_returns(r_seq, R, gamma, lam)
            b = common.turn_returns(r_seq, R, gamma, lam)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=12)


if __name__ == "__main__":
    unittest.main()
