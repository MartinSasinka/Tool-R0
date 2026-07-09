"""Offline test of run_eval_batch.run_cells_parallel (mock commands, no GPU)."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(_HERE, "..", "scripts", "eval", "run_eval_batch.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("reb", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parallel_scheduler_reports_failures():
    reb = _load_runner()
    bd = tempfile.mkdtemp()
    ok_cmd = [sys.executable, "-c", "print('ok')"]
    bad_cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
    cmds = [
        ({"name": "a"}, os.path.join(bd, "a"), ok_cmd),
        ({"name": "b"}, os.path.join(bd, "b"), bad_cmd),
        ({"name": "c"}, os.path.join(bd, "c"), ok_cmd),
    ]
    failures = reb.run_cells_parallel(cmds, bd, ["0", "1"], 2)
    assert [(n, rc) for n, rc, _ in failures] == [("b", 3)], failures
    # per-cell logs exist
    for name in ("a", "b", "c"):
        assert os.path.isfile(os.path.join(bd, f"{name}.log")), name


def test_parallel_scheduler_all_ok():
    reb = _load_runner()
    bd = tempfile.mkdtemp()
    ok_cmd = [sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"]
    cmds = [({"name": f"c{i}"}, os.path.join(bd, f"c{i}"), ok_cmd) for i in range(3)]
    failures = reb.run_cells_parallel(cmds, bd, ["0", "1"], 2)
    assert failures == []
    # each log recorded the pinned GPU id
    seen = set()
    for i in range(3):
        with open(os.path.join(bd, f"c{i}.log"), encoding="utf-8") as fh:
            seen.add(fh.read().strip())
    assert seen <= {"0", "1"}, seen


if __name__ == "__main__":
    test_parallel_scheduler_reports_failures()
    test_parallel_scheduler_all_ok()
    print("[test_parallel_eval_scheduler] ALL TESTS PASSED")
