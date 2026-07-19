#!/usr/bin/env python3
"""Unit tests for pure Stage 3 overnight pipeline pieces (no GPU)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_V3 = Path(__file__).resolve().parents[1]
_REPO = _V3.parents[1]
sys.path.insert(0, str(_V3))
sys.path.insert(0, str(_V3 / "scripts" / "data"))

MATERIALIZE = _V3 / "scripts" / "data" / "materialize_pure_stage3.py"
AUDIT = _V3 / "scripts" / "data" / "audit_stage3_nestful_syntax.py"
SHELL = _V3 / "scripts" / "v5" / "run_pure_stage3_two_epoch_overnight.sh"
PHASE2 = _V3 / "data" / "training_ready_v5" / "filtered" / "phase2_stage3_plus_stage2_replay.jsonl"


class TestShellSyntax(unittest.TestCase):
    def test_bash_n(self):
        if os.name == "nt":
            self.skipTest("bash -n on Unix/WSL")
        r = subprocess.run(["bash", "-n", str(SHELL)], capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode())


@unittest.skipUnless(PHASE2.is_file(), "phase2 dataset missing")
class TestMaterializeAndAudit(unittest.TestCase):
    def test_materialize_326_no_stage2(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "stage3.jsonl"
            r = subprocess.run(
                [sys.executable, str(MATERIALIZE),
                 "--source", str(PHASE2), "--out", str(out), "--force"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(rows), 326)
            self.assertTrue(all("stage3" in str(r.get("stage")) for r in rows))
            self.assertFalse(any("stage2" in str(r.get("stage")) for r in rows))
            self.assertTrue(all(r.get("num_calls") == 3 for r in rows))
            ids = [r["sample_id"] for r in rows]
            self.assertEqual(len(ids), len(set(ids)))

    def test_audit_noop_preserves_hash(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "stage3.jsonl"
            report_dir = Path(td) / "audit"
            subprocess.run(
                [sys.executable, str(MATERIALIZE),
                 "--source", str(PHASE2), "--out", str(out), "--force"],
                check=True, capture_output=True,
            )
            h1 = hashlib.sha256(out.read_bytes()).hexdigest()
            r = subprocess.run(
                [sys.executable, str(AUDIT),
                 "--input", str(out), "--report-dir", str(report_dir)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            report = json.loads((report_dir / "stage3_nestful_syntax_audit.json").read_text(
                encoding="utf-8"))
            self.assertEqual(report["verdict"], "NO_MISMATCH")
            self.assertIsNone(report.get("output_path"))
            h2 = hashlib.sha256(out.read_bytes()).hexdigest()
            self.assertEqual(h1, h2)
            self.assertEqual(report["input_sha256"], h1)

    def test_normalize_underscore_preserves_execution(self):
        """If we force-normalize a synthetic underscore rewrite, semantics hold."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit_s3", AUDIT)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        maybe_normalize_row = mod.maybe_normalize_row
        row = {
            "sample_id": "toy",
            "stage": "stage3_3call_agentic_openrouter",
            "num_calls": 3,
            "gold_calls": [
                {"name": "add_numbers", "arguments": {"a": 1, "b": 2},
                 "label": "$var_1"},
                {"name": "add_numbers",
                 "arguments": {"a": "$var_1.result$", "b": 3},
                 "label": "$var_2"},
                {"name": "add_numbers",
                 "arguments": {"a": "$var_2.result$", "b": 4},
                 "label": "$var_3"},
            ],
            "question": "add numbers",
            "tools": [],
            "observations": None,
            "gold_answer": None,
        }
        new, changes = maybe_normalize_row(row)
        self.assertTrue(changes)
        self.assertEqual(new["gold_calls"][0]["label"], "$var1")
        self.assertEqual(new["gold_calls"][1]["arguments"]["a"], "$var1.result$")


class TestCreditProbeFormula(unittest.TestCase):
    def test_turn_returns_lambda0_still_sums_future(self):
        sys.path.insert(0, str(_V3.parent / "nestful_mtgrpo_minimal"))
        from grpo_train import _turn_returns
        G = _turn_returns([0.9, 0.2], episode_reward=0.35, gamma=1.0, lambda_episode=0.0)
        self.assertAlmostEqual(G[0], 1.1)  # 0.9+0.2, no R_episode
        self.assertAlmostEqual(G[1], 0.2)

    def test_gamma0_is_local(self):
        sys.path.insert(0, str(_V3.parent / "nestful_mtgrpo_minimal"))
        from grpo_train import _turn_returns
        G = _turn_returns([0.9, 0.2], episode_reward=0.35, gamma=0.0, lambda_episode=1.0)
        # gamma=0 zeros future turn rewards AND the episode term
        # (gamma^(T-t+1) with T-t+1 >= 1). So G_t == r_t for every t.
        self.assertAlmostEqual(G[0], 0.9)
        self.assertAlmostEqual(G[1], 0.2)


if __name__ == "__main__":
    unittest.main()
