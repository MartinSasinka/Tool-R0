"""Regression tests for the executor.mode=gold_replay guard on agentic
(synthetic-tool) datasets.

Background: agentic tool names (e.g. "units_per_box") are NOT entries in the
real NESTFUL IBM function registry, which IS present in this repo (used for
genuine NESTFUL data). Any code path that resolves executor.mode="auto"
against that registry picks "full" execution and either hard-fails every
episode on `unknown_function`, or worse, silently executes a DIFFERENT real
IBM function on a name collision — corrupting the reward either way.
`probe_stage.py` now auto-detects agentic datasets (by path or by a `source`
field containing "agentic") and forces `executor.mode=gold_replay` unless the
caller explicitly overrides it. These tests exercise that guard end-to-end
with the CPU-only stub backend (no GPU / model download required).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "experiments/nestful_synthetic_curriculum_v3"
SCRIPTS_LIB = V3 / "scripts/lib"
PROBE_SCRIPT = V3 / "scripts/probe/probe_stage.py"

sys.path.insert(0, str(SCRIPTS_LIB))
from paths import (  # noqa: E402
    is_agentic_synthetic_dataset, is_agentic_synthetic_dataset_path,
    peek_dataset_source,
)

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail=None) -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}: {detail}")
        _FAILURES.append(name)


# A single synthetic tool call whose name is guaranteed to NOT exist in the
# real IBM NESTFUL function registry.
_FIXTURE_ROW = {
    "sample_id": "fixture_probe_guard_000001",
    "question": "Compute a fake quantity.",
    "tools": [{
        "name": "definitely_not_a_real_ibm_function_xyz",
        "description": "fixture-only synthetic tool.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]},
        "output_parameters": {"output_0": {"type": "number"}},
    }],
    "gold_calls": [{"name": "definitely_not_a_real_ibm_function_xyz",
                    "arguments": {"a": 3, "b": 4}, "label": "$var1"}],
    "observations": [7],
    "gold_answer": 7,
    "num_calls": 1,
    "stage": "stage1_1call_atomic",
}


def _write_fixture(dir_path: Path, filename: str, *, source: str | None) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / filename
    row = dict(_FIXTURE_ROW)
    if source is not None:
        row["source"] = source
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return path


def _run_probe(dataset_path: Path, out_dir: Path, *, extra_args=None) -> dict:
    # 15 generations at seed=42 deterministically covers the stub "perfect"
    # behavior (gen_idx=1) for this fixture's task_id — see the hash-derived
    # `_stub_rng_choice` in probe_stage.py. A "perfect" gold-matching call
    # under a BROKEN executor.mode (full, against the real IBM registry)
    # still fails with `unknown_function` / a colliding real function; only
    # under the correct gold_replay mode does it hit the reward's
    # fully_correct floor (>= 0.90).
    args = [sys.executable, str(PROBE_SCRIPT),
            "--dataset", str(dataset_path),
            "--num-tasks", "1", "--num-generations", "15", "--seed", "42",
            "--backend", "stub", "--output-dir", str(out_dir)]
    args.extend(extra_args or [])
    result = subprocess.run(args, cwd=str(REPO), capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, (
        f"probe_stage.py failed rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}")
    report_path = out_dir / "PROBE_REPORT.json"
    with open(report_path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    # ---- unit tests: dataset-type detection helpers -----------------------
    check("path heuristic flags a curriculum_v4 agentic path",
          is_agentic_synthetic_dataset_path(
              "data/curriculum_v4_nestful_like_agentic_openrouter/filtered/x.jsonl"),
          "expected True")
    check("path heuristic does NOT flag the canonical v3.1 corpus",
          not is_agentic_synthetic_dataset_path(
              "outputs/curriculum_v3_1/filtered/stage2_2call_dependency.jsonl"),
          "expected False")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # content-based detection: neutral path, but source field says agentic
        neutral = _write_fixture(tmp_path / "neutral_dir", "custom_export.jsonl",
                                 source="curriculum_v4_nestful_like_agentic_openrouter")
        check("peek_dataset_source reads the first row's source field",
              peek_dataset_source(str(neutral)) ==
              "curriculum_v4_nestful_like_agentic_openrouter",
              peek_dataset_source(str(neutral)))
        check("is_agentic_synthetic_dataset detects via `source` even with a "
              "neutral path",
              is_agentic_synthetic_dataset(str(neutral)), "expected True")

        # no agentic markers anywhere -> not flagged
        clean = _write_fixture(tmp_path / "neutral_dir", "clean.jsonl", source=None)
        check("is_agentic_synthetic_dataset is False with no markers at all",
              not is_agentic_synthetic_dataset(str(clean)), "expected False")

        # ---- end-to-end: probe_stage.py auto-forces gold_replay ------------
        agentic_dir = tmp_path / "agentic_openrouter_fixture"
        agentic_file = _write_fixture(agentic_dir, "stage1_fixture.jsonl", source=None)
        out_dir = tmp_path / "probe_out_default"
        report = _run_probe(agentic_file, out_dir)
        check("probe_stage.py auto-forces executor.mode=gold_replay on an "
              "agentic-path dataset",
              report.get("executor_mode") == "gold_replay", report.get("executor_mode"))
        max_reward = max(report["groups"][0]["rewards"])
        check("a gold-matching stub completion scores a high (fully_correct) "
              "reward once gold_replay is forced — under the broken `full` "
              "mode this would be capped low by unknown_function/collision",
              max_reward >= 0.85, report["groups"][0])

        # ---- explicit --override takes precedence over the auto-guard -----
        out_dir_full = tmp_path / "probe_out_full_override"
        report_full = _run_probe(agentic_file, out_dir_full,
                                 extra_args=["--override", "executor.mode=full"])
        check("--override executor.mode=full takes precedence over the "
              "agentic auto-guard",
              report_full.get("executor_mode") == "full", report_full.get("executor_mode"))
        max_reward_full = max(report_full["groups"][0]["rewards"])
        check("forcing executor.mode=full on a synthetic-tool fixture "
              "corrupts even the gold-matching completion's reward — this "
              "is the exact bug the auto-guard exists to prevent",
              max_reward_full < 0.85, report_full["groups"][0])

    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        return 1
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
