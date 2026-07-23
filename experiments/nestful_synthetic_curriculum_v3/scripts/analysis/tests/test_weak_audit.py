"""Unit tests for weak-model audit pipeline."""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments/nestful_mtgrpo_minimal"
for p in (_MINIMAL, _V3, _HERE.parent):
    sys.path.insert(0, str(p))

from weak_audit.agreement import cohen_kappa, compare_passes  # noqa: E402
from weak_audit.backup import backup_before_retry  # noqa: E402
from weak_audit.compression import compress_packet, estimate_tokens  # noqa: E402
from weak_audit.constants import SEED  # noqa: E402
from weak_audit.io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from weak_audit.invalid_discovery import discover_invalid, write_invalid_discovery  # noqa: E402
from weak_audit.merge import merge_final_annotations  # noqa: E402
from weak_audit.pass_builder import build_pass_b  # noqa: E402
from weak_audit.provider_audit import compute_provider_stats  # noqa: E402
from weak_audit.retry import build_retry_inputs, run_invalid_retry  # noqa: E402
from weak_audit.schema import validate_annotation  # noqa: E402
from weak_audit.selection import select_tasks  # noqa: E402
from weak_audit.summarize_outputs import write_summarize_outputs  # noqa: E402


def _sample_meta(n: int = 120):
    meta = {}
    r0 = {}
    for i in range(n):
        tid = f"t{i}"
        w0 = i % 5 == 0
        w2 = i % 7 == 0
        meta[tid] = {
            "w0": w0, "w1": w0, "w2": w2,
            "gold_call_bucket": str(i % 3 + 1),
            "motif": f"m{i % 4}",
            "first_divergence_turn": i % 3,
            "c0_failure": "wrong tool",
            "e2_failure": "wrong args",
            "reward_mismatch": w0 and not w2,
            "e2_executable_wrong": i % 11 == 0,
        }
        r0[tid] = {
            "class_C0": "too_few_calls" if w0 and i % 13 == 0 else "ok",
            "class_E2": "ok",
        }
    return meta, r0


class TestSelection(unittest.TestCase):
    def test_dedup_and_limits(self):
        meta, r0 = _sample_meta(200)
        cohort_tasks, assigned = select_tasks(meta, r0, seed=SEED)
        all_ids = []
        for ids in cohort_tasks.values():
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertLessEqual(len(all_ids), 250)
        self.assertEqual(len(cohort_tasks["c0_win_e2_loss"]),
                         len([t for t, m in meta.items() if m["w0"] and not m["w2"]]))


class TestPassB(unittest.TestCase):
    def test_permutation_and_mapping(self):
        packet = {
            "task_id": "abc123",
            "question": "q",
            "expected_outcome": "x",
            "gold_metadata": {"gold_call_count": 2, "motif": "m", "gold_calls": []},
            "relevant_tools": [],
            "deterministic_flags": {"first_divergence_turn": 1},
            "C0": {"calls": [{"name": "a"}]},
            "E1": {"calls": [{"name": "b"}]},
            "E2": {"calls": [{"name": "c"}]},
        }
        inp, mapping = build_pass_b(packet, seed=SEED)
        inv = {v: k for k, v in mapping.items()}
        for arm in ("C0", "E1", "E2"):
            label = inv[arm]
            self.assertEqual(
                packet[arm]["calls"],
                inp["case"]["trajectories"][label]["calls"],
            )

    def test_deterministic_shuffle(self):
        packet = {"task_id": "fixed", "question": "", "expected_outcome": "",
                  "gold_metadata": {"gold_call_count": 1, "motif": ""},
                  "relevant_tools": [], "deterministic_flags": {},
                  "C0": {}, "E1": {}, "E2": {}}
        m1 = build_pass_b(packet, seed=SEED)[1]
        m2 = build_pass_b(packet, seed=SEED)[1]
        self.assertEqual(m1, m2)


class TestSchema(unittest.TestCase):
    def test_valid_annotation(self):
        obj = {
            "task_id": "t1",
            "first_divergence_turn": 1,
            "root_cause": "reward_mismatch",
            "shorter_path_verdict": "valid",
            "observation_used_correctly": True,
            "reward_ordering_correct": False,
            "responsible_reward_component": "call_count",
            "recommended_fix": "outcome_reward",
            "confidence": 0.8,
            "evidence": "turn 2 call differs",
        }
        self.assertEqual(validate_annotation(obj, expected_task_id="t1"), [])


class TestCompression(unittest.TestCase):
    def test_compress_reduces_tokens(self):
        pkt = {
            "task_id": "t",
            "question": "x" * 2000,
            "relevant_tools": [
                {"name": "used", "description": "d" * 500},
                {"name": "unused", "description": "u" * 500},
            ],
            "gold_metadata": {"gold_calls": [{"name": "used"}]},
            "C0": {"calls": [{"name": "used"}]},
            "E1": {"calls": []},
            "E2": {"calls": []},
        }
        out, log = compress_packet(pkt, target=100, hard=6000)
        self.assertTrue(log["removed_parts"])
        self.assertLessEqual(log["tokens_after"], log["tokens_before"])
        names = [t["name"] for t in out["relevant_tools"]]
        self.assertIn("used", names)
        self.assertNotIn("unused", names)


class TestAgreement(unittest.TestCase):
    def test_cohen_kappa_perfect(self):
        self.assertAlmostEqual(cohen_kappa(["a", "a"], ["a", "a"]), 1.0)

    def test_compare_passes(self):
        a = {"t1": {"root_cause": "x", "reward_ordering_correct": True,
                    "first_divergence_turn": 1, "shorter_path_verdict": "valid",
                    "observation_used_correctly": True,
                    "responsible_reward_component": "none", "recommended_fix": "no_change",
                    "confidence": 0.9}}
        b = {"t1": dict(a["t1"])}
        stats = compare_passes(a, b)
        self.assertEqual(stats["exact_agreement_rate"], 1.0)


def _valid_ann(task_id: str, pass_label: str = "A") -> dict:
    return {
        "task_id": task_id,
        "pass": pass_label,
        "first_divergence_turn": 1,
        "root_cause": "reward_mismatch",
        "shorter_path_verdict": "valid",
        "observation_used_correctly": True,
        "reward_ordering_correct": False,
        "responsible_reward_component": "call_count",
        "recommended_fix": "outcome_reward",
        "confidence": 0.8,
        "evidence": "turn 2 differs",
    }


class TestRetryFinalize(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.t_a = "task_a_valid"
        self.t_b = "task_b_invalid"
        self.t_c = "task_c_invalid"
        write_jsonl(self.tmp / "pass_a_inputs.jsonl", [
            {"task_id": self.t_a, "pass": "A", "case": {}},
            {"task_id": self.t_b, "pass": "A", "case": {}},
        ])
        write_jsonl(self.tmp / "pass_b_inputs.jsonl", [
            {"task_id": self.t_c, "pass": "B", "case": {}},
        ])
        write_jsonl(self.tmp / "pass_a_annotations.jsonl", [_valid_ann(self.t_a, "A")])
        write_jsonl(self.tmp / "pass_b_annotations.jsonl", [])
        write_jsonl(self.tmp / "invalid_annotations.jsonl", [
            {"task_id": self.t_b, "pass": "A", "errors": ["invalid enum"]},
            {"task_id": self.t_c, "pass": "B", "errors": ["truncation"]},
        ])
        write_jsonl(self.tmp / "pass_a_annotations_raw.jsonl", [
            {"task_id": self.t_a, "pass": "A", "provider": "p1", "response": "{}"},
            {"task_id": self.t_b, "pass": "A", "provider": "p1", "response": "{"},
        ])
        write_jsonl(self.tmp / "pass_b_annotations_raw.jsonl", [
            {"task_id": self.t_c, "pass": "B", "provider": "p2", "response": "{"},
        ])
        write_json(self.tmp / "pass_b_mapping.json", {"A": "C0"})
        write_jsonl(self.tmp / "case_packets.jsonl", [
            {"task_id": self.t_a, "cohorts": ["c0_win_e2_loss"]},
            {"task_id": self.t_b, "cohorts": ["reward_mismatch"]},
            {"task_id": self.t_c, "cohorts": ["c0_win_e2_loss"]},
        ])

    def test_discover_only_invalid_pairs(self):
        _, manifest = discover_invalid(self.tmp)
        self.assertEqual(manifest["n_invalid_pairs"], 2)
        pairs = {(p["task_id"], p["pass_label"]) for p in manifest["pairs"]}
        self.assertEqual(pairs, {(self.t_b, "A"), (self.t_c, "B")})
        self.assertNotIn((self.t_a, "A"), pairs)

    def test_build_retry_inputs_excludes_valid(self):
        write_json(self.tmp / "invalid_retry_manifest.json", {
            "pairs": [
                {"task_id": self.t_b, "pass_label": "A"},
                {"task_id": self.t_c, "pass_label": "B"},
            ]
        })
        inputs = build_retry_inputs(self.tmp)
        self.assertEqual(len(inputs), 2)
        ids = {(i["task_id"], i["pass"]) for i in inputs}
        self.assertNotIn((self.t_a, "A"), ids)

    def test_dry_run_retry(self):
        write_invalid_discovery(self.tmp)
        stats = run_invalid_retry(self.tmp, model="m", dry_run=True)
        self.assertTrue(stats["dry_run"])
        self.assertEqual(stats["n_pairs"], 2)

    def test_merge_and_final_counts(self):
        write_json(self.tmp / "invalid_retry_manifest.json", {
            "pairs": [
                {"task_id": self.t_b, "pass_label": "A"},
                {"task_id": self.t_c, "pass_label": "B"},
            ]
        })
        write_jsonl(self.tmp / "retry_invalid_validated.jsonl", [
            _valid_ann(self.t_b, "A"),
            _valid_ann(self.t_c, "B"),
        ])
        write_jsonl(self.tmp / "retry_invalid_failed.jsonl", [])
        stats = merge_final_annotations(self.tmp, expected_per_pass=2)
        self.assertEqual(stats["pass_a_replaced"], 1)
        self.assertEqual(stats["pass_b_replaced"], 1)
        self.assertEqual(len(read_jsonl(self.tmp / "pass_a_annotations_final.jsonl")), 2)
        self.assertEqual(len(read_jsonl(self.tmp / "pass_b_annotations_final.jsonl")), 1)
        self.assertEqual(len(read_jsonl(self.tmp / "invalid_annotations_final.jsonl")), 0)

    def test_merge_keeps_still_invalid(self):
        write_json(self.tmp / "invalid_retry_manifest.json", {
            "pairs": [{"task_id": self.t_b, "pass_label": "A"}],
        })
        write_jsonl(self.tmp / "retry_invalid_validated.jsonl", [])
        write_jsonl(self.tmp / "retry_invalid_failed.jsonl", [
            {"task_id": self.t_b, "pass": "A", "errors": ["still bad"]},
        ])
        stats = merge_final_annotations(self.tmp, expected_per_pass=2)
        self.assertEqual(stats["pass_a_still_invalid"], 1)
        self.assertEqual(len(read_jsonl(self.tmp / "invalid_annotations_final.jsonl")), 1)

    def test_no_duplicate_final_rows(self):
        write_json(self.tmp / "invalid_retry_manifest.json", {
            "pairs": [{"task_id": self.t_b, "pass_label": "A"}],
        })
        write_jsonl(self.tmp / "retry_invalid_validated.jsonl", [_valid_ann(self.t_b, "A")])
        write_jsonl(self.tmp / "retry_invalid_failed.jsonl", [])
        merge_final_annotations(self.tmp, expected_per_pass=2)
        finals = read_jsonl(self.tmp / "pass_a_annotations_final.jsonl")
        self.assertEqual(len(finals), len({r["task_id"] for r in finals}))

    def test_provider_stats(self):
        report, recommended = compute_provider_stats(self.tmp)
        self.assertTrue(report["providers"])
        if recommended is not None:
            self.assertIn(recommended, {r["provider"] for r in report["providers"]})
        else:
            self.assertTrue(all(r["requests"] < 10 for r in report["providers"]))

    def test_summarize_final_suffix(self):
        write_jsonl(self.tmp / "pass_a_annotations_final.jsonl", [_valid_ann(self.t_a, "A")])
        write_jsonl(self.tmp / "pass_b_annotations_final.jsonl", [_valid_ann(self.t_a, "B")])
        out = write_summarize_outputs(self.tmp, suffix="final")
        self.assertEqual(out["n_both"], 1)
        self.assertTrue((self.tmp / "ANNOTATION_AGREEMENT_FINAL.md").is_file())

    def test_backup_sha256_manifest(self):
        import tempfile
        backup = Path(tempfile.mkdtemp())
        manifest = backup_before_retry(self.tmp, backup)
        self.assertIn("sha256", manifest)
        self.assertTrue((backup / "MANIFEST_SHA256.json").is_file())
        for name in manifest["files"]:
            self.assertEqual(manifest["sha256"][name], sha256_file(backup / name))


if __name__ == "__main__":
    unittest.main()
