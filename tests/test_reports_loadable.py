"""Every comparison CSV must be loadable by pandas (CSV-hygiene regression).

The audit found report CSVs that broke ``pandas.read_csv`` because free-text
fields held unescaped commas / newlines. This test enforces that all current
and future ``experiments/comparison/*.csv`` files round-trip, and that the
``nestful_core.logging_utils.write_csv`` writer produces loadable output even
for adversarial values.
"""
from __future__ import annotations

import glob
import os

import pytest

pd = pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPARISON = os.path.join(_REPO, "experiments", "comparison")


def _csv_files():
    return sorted(glob.glob(os.path.join(_COMPARISON, "*.csv")))


@pytest.mark.parametrize("path", _csv_files())
def test_comparison_csv_loads(path):
    df = pd.read_csv(path)
    assert df.shape[1] >= 1, f"{os.path.basename(path)} parsed to 0 columns"


def test_write_csv_handles_commas_and_newlines(tmp_path):
    from nestful_core.logging_utils import write_csv

    rows = [
        {"a": "x, y, z", "b": "line1\nline2", "c": {"k": [1, 2, 3]}, "d": True},
        {"a": "plain", "b": "ok", "c": [1, "two", 3], "d": False},
    ]
    out = str(tmp_path / "hygiene.csv")
    write_csv(out, rows)
    df = pd.read_csv(out)
    assert list(df.columns) == ["a", "b", "c", "d"]
    assert len(df) == 2
    assert df.iloc[0]["a"] == "x, y, z"
