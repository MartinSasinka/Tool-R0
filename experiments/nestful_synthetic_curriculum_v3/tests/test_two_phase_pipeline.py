"""Unit tests for two-phase GRPO pipeline helpers (no GPU required)."""
from __future__ import annotations

import json
import os
import sys
import unittest

V3 = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if V3 not in sys.path:
    sys.path.insert(0, V3)

from scripts.training.two_phase_utils import (  # noqa: E402
    REQUIRED_REGISTRY_VERSION,
    adapter_dir_hash,
    atomic_publish_checkpoint,
    audit_dataset_ids,
    assert_canonical_training,
    rollout_seed,
    task_seed,
    verify_adapter_dir,
    verify_dev_test_disjoint,
)


class TestTwoPhaseUtils(unittest.TestCase):
    def test_rollout_seed_deterministic(self):
        a = rollout_seed(base=42, task_idx=3, rollout_idx=1)
        b = rollout_seed(base=42, task_idx=3, rollout_idx=1)
        c = rollout_seed(base=42, task_idx=3, rollout_idx=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_task_seed_varies_by_phase(self):
        s1 = task_seed(data_seed=42, task_idx=0, phase="phase1")
        s2 = task_seed(data_seed=42, task_idx=0, phase="phase2")
        self.assertNotEqual(s1, s2)

    def test_canonical_assertions_pass(self):
        assert_canonical_training(
            executor_mode="synthetic",
            reward_policy="execution_aware_v3_2_dense",
            registry_version=REQUIRED_REGISTRY_VERSION,
        )

    def test_canonical_assertions_fail_gold_replay(self):
        with self.assertRaises(SystemExit):
            assert_canonical_training(
                executor_mode="gold_replay",
                reward_policy="execution_aware_v3_2_dense",
                registry_version=REQUIRED_REGISTRY_VERSION,
            )

    def test_phase1_dataset_unique_ids(self):
        p = os.path.join(
            V3, "data", "training_ready_v5", "filtered", "phase1_stage2_train.jsonl")
        if not os.path.isfile(p):
            self.skipTest("training dataset not present")
        aud = audit_dataset_ids(p)
        self.assertTrue(aud["ok"])
        self.assertEqual(aud["rows"], 429)

    def test_dev_test_disjoint(self):
        dev = os.path.join(
            V3, "..", "nestful_mtgrpo_minimal", "data", "splits", "nestful_dev.jsonl")
        dev = os.path.normpath(dev)
        if not os.path.isfile(dev):
            self.skipTest("nestful_dev.jsonl missing")
        rep = verify_dev_test_disjoint(dev)
        self.assertTrue(rep["ok"], rep)

    def test_atomic_publish_checkpoint(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src")
            os.makedirs(src)
            with open(os.path.join(src, "adapter_config.json"), "w", encoding="utf-8") as fh:
                json.dump({"peft_type": "LORA"}, fh)
            with open(os.path.join(src, "adapter_model.safetensors"), "wb") as fh:
                fh.write(b"fake-weights")

            dest = os.path.join(td, "checkpoints", "C1")
            manifest = atomic_publish_checkpoint(src, dest, label="C1")
            self.assertTrue(os.path.isdir(dest))
            self.assertFalse(os.path.isdir(dest + ".tmp"))
            self.assertEqual(manifest["label"], "C1")
            self.assertTrue(os.path.isfile(os.path.join(dest, "checkpoint_manifest.json")))
            verified = verify_adapter_dir(dest)
            self.assertTrue(verified["ok"])
            self.assertEqual(manifest["adapter_hash"], adapter_dir_hash(dest))


if __name__ == "__main__":
    unittest.main()
