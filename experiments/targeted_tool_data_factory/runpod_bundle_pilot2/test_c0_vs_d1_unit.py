#!/usr/bin/env python3
"""Unit checks for --c0-vs-d1 support in eval/report (no GPU, no network)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def _write_traj(path: Path, wins: dict[str, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sid, w in wins.items():
            fh.write(json.dumps({
                "sample_id": sid,
                "internal_win_rate": 1.0 if w else 0.0,
                "internal_function_f1": 1.0 if w else 0.0,
                "internal_parameter_f1": 1.0 if w else 0.0,
                "internal_executability": 1.0,
                "_traj": {"official_win": w},
            }) + "\n")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # Windows console encodings differ; decode as bytes then replace.
    return subprocess.run(cmd, capture_output=True)


def _out(r: subprocess.CompletedProcess) -> str:
    return ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", errors="replace")


def test_report_c0_vs_d1_without_d0() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_traj(root / "eval" / "C0_heldout80" / "final_eval_trajectories.jsonl",
                    {"a": False, "b": True, "c": False})
        _write_traj(root / "eval" / "D1_heldout80" / "final_eval_trajectories.jsonl",
                    {"a": True, "b": True, "c": False})
        _write_traj(root / "eval" / "C0_nestful500" / "final_eval_trajectories.jsonl",
                    {"x": False, "y": False})
        _write_traj(root / "eval" / "D1_nestful500" / "final_eval_trajectories.jsonl",
                    {"x": True, "y": False})
        md = root / "C0_VS_D1_REPORT.md"
        js = root / "C0_VS_D1_REPORT.json"
        r = _run([sys.executable, str(BUNDLE / "make_paired_report.py"),
                  "--results", str(root), "--out", str(md), "--json-out", str(js),
                  "--baseline", "C0", "--treatment", "D1", "--title", "C0 vs D1"])
        assert r.returncode == 0, _out(r)
        text = md.read_text(encoding="utf-8")
        assert "C0 vs D1" in text
        assert "missing D0" not in text
        assert "gained" in text.lower()
        report = json.loads(js.read_text(encoding="utf-8"))
        assert report["baseline"] == "C0" and report["treatment"] == "D1"
        held = next(s for s in report["sections"] if s["eval_set"] == "heldout80")
        assert held["complete"] is True
        assert held["n_gained"] == 1  # a: C0 lose -> D1 win
        assert held["n_lost"] == 0


def test_eval_cli_accepts_c0_d1_without_d0_run() -> None:
    r = _run([sys.executable, str(BUNDLE / "run_eval_all.py"), "--help"])
    assert r.returncode == 0
    assert "--arms" in _out(r)
    # dry-run with C0,D1 and no --d0-run must not abort on missing D0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        held = root / "held.jsonl"
        diag = root / "diag.jsonl"
        held.write_text("{}\n", encoding="utf-8")
        diag.write_text("{}\n", encoding="utf-8")
        r = _run([sys.executable, str(BUNDLE / "run_eval_all.py"),
                  "--output-root", str(root / "runs"),
                  "--d1-run", "pilot2_D1_seed20260726",
                  "--heldout", str(held),
                  "--diagnostic", str(diag),
                  "--results", str(root / "results"),
                  "--arms", "C0,D1",
                  "--dry-run"])
        out = _out(r)
        assert r.returncode == 0, out
        assert "C0_heldout80" in out
        assert "D1_heldout80" in out
        assert "D0_heldout80" not in out


def main() -> int:
    test_report_c0_vs_d1_without_d0()
    print("ok: report C0 vs D1 without D0")
    test_eval_cli_accepts_c0_d1_without_d0_run()
    print("ok: eval CLI C0,D1 without --d0-run")
    print("[test_c0_vs_d1_unit] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
