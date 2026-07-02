"""Tests for the official NESTFUL scoring adapter + the internal replica.

Per the audit policy:
  * official_* (from nestful_official_score.py / the real scorer) is CANONICAL.
  * internal_* (from metrics.py) is a DIAGNOSTIC replica.

We assert agreement between official_* and internal_* on partial/full/win for the
control fixtures, and we test the corpus-level macro-F1 aggregation SEPARATELY
(official F1 is corpus-level macro, not a per-sample number).

Win Rate requires executing the fixture functions, which the official scorer does
via signal.SIGALRM (Unix only); those assertions are skipped on Windows.
"""
import json
import os

import pytest

from metrics import compute_nestful_official_metrics, internal_corpus_macro_f1
from nestful_official_score import (
    build_item,
    score_items,
    score_items_per_sample,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
FUNC_DIR = os.path.join(_HERE, "fixtures", "exec_funcs")
WANT_WIN = os.name != "nt"  # official Win re-exec uses SIGALRM (Unix only)
win_only = pytest.mark.skipif(not WANT_WIN, reason="Win Rate needs SIGALRM (Unix)")

TOOLS = [
    {"name": "add", "output_parameters": {"result": {}},
     "parameters": {"properties": {"a": {}, "b": {}}}},
    {"name": "multiply", "output_parameters": {"result": {}},
     "parameters": {"properties": {"a": {}, "b": {}}}},
    {"name": "divide", "output_parameters": {"result": {}},
     "parameters": {"properties": {"a": {}, "b": {}}}},
]

# Nested gold: add(1,2)=3 -> multiply(3,10)=30
GOLD = [
    {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
    {"name": "multiply", "label": "$var2", "arguments": {"a": "$var1.result$", "b": 10}},
]
GOLD_ANS = 30


def _make_item(pred_calls, gold=GOLD, gold_ans=GOLD_ANS, tools=TOOLS):
    return build_item(pred_calls, {"output": gold, "tools": tools, "gold_answer": gold_ans})


def _official_one(pred_calls, gold=GOLD, gold_ans=GOLD_ANS):
    item = _make_item(pred_calls, gold=gold, gold_ans=gold_ans)
    return score_items_per_sample([item], executable_func_dir=FUNC_DIR, win_rate=WANT_WIN)[0]


def _internal_one(pred_calls, gold=GOLD):
    return compute_nestful_official_metrics(pred_calls, gold)


# ---------------------------------------------------------------------------
# Scenario 1: perfect prediction -> partial=full=1, win=1
# ---------------------------------------------------------------------------

def test_scenario1_perfect():
    pred = [dict(c) for c in GOLD]
    off = _official_one(pred)
    int_ = _internal_one(pred)
    assert off["official_partial_match"] == 1.0
    assert off["official_full_match"] == 1.0
    assert off["parse_valid"] is True
    # internal agrees on partial/full
    assert int_["partial_sequence_accuracy"] == 1.0
    assert int_["full_sequence_accuracy"] == 1.0


@win_only
def test_scenario1_perfect_win():
    off = _official_one([dict(c) for c in GOLD])
    assert off["official_win"] == 1.0
    assert off["executable"] is True
    assert off["pred_answer"] == 30


# ---------------------------------------------------------------------------
# Scenario 2: right functions, wrong order -> partial=0, full=0
# (F1 Func would still be high; that's the corpus-vs-sequence distinction.)
# ---------------------------------------------------------------------------

_FLAT_GOLD = [
    {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
    {"name": "multiply", "label": "$var2", "arguments": {"a": 3, "b": 10}},
]


def test_scenario2_wrong_order():
    pred = [
        {"name": "multiply", "label": "$var1", "arguments": {"a": 3, "b": 10}},
        {"name": "add", "label": "$var2", "arguments": {"a": 1, "b": 2}},
    ]
    off = _official_one(pred, gold=_FLAT_GOLD)
    int_ = _internal_one(pred, gold=_FLAT_GOLD)
    assert off["official_partial_match"] == 0.0
    assert off["official_full_match"] == 0.0
    assert int_["partial_sequence_accuracy"] == pytest.approx(off["official_partial_match"])
    assert int_["full_sequence_accuracy"] == off["official_full_match"]


# ---------------------------------------------------------------------------
# Scenario 3: right sequence, wrong argument -> partial=0.5, full=0
# ---------------------------------------------------------------------------

def test_scenario3_wrong_argument():
    pred = [
        {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
        {"name": "multiply", "label": "$var2", "arguments": {"a": "$var1.result$", "b": 99}},
    ]
    off = _official_one(pred)
    int_ = _internal_one(pred)
    assert off["official_partial_match"] == pytest.approx(0.5)
    assert off["official_full_match"] == 0.0
    # internal replica agrees on partial/full
    assert int_["partial_sequence_accuracy"] == pytest.approx(0.5)
    assert int_["full_sequence_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 4: alternative trajectory, same answer -> Win=1 while Full=0
# ---------------------------------------------------------------------------

def test_scenario4_alternative_trajectory_full0():
    pred = [{"name": "multiply", "label": "$var1", "arguments": {"a": 5, "b": 6}}]  # 30
    off = _official_one(pred)
    assert off["official_full_match"] == 0.0  # different sequence than gold


@win_only
def test_scenario4_alternative_trajectory_win1():
    pred = [{"name": "multiply", "label": "$var1", "arguments": {"a": 5, "b": 6}}]
    off = _official_one(pred)
    assert off["official_win"] == 1.0  # reaches gold answer 30 via a valid path
    assert off["official_full_match"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 5: nested-reference execution -> Win=1
# ---------------------------------------------------------------------------

_NESTED_GOLD = [
    {"name": "add", "label": "$var1", "arguments": {"a": 2, "b": 3}},      # 5
    {"name": "multiply", "label": "$var2", "arguments": {"a": "$var1.result$", "b": 4}},  # 20
]


def test_scenario5_nested_partial_full():
    pred = [dict(c) for c in _NESTED_GOLD]
    off = _official_one(pred, gold=_NESTED_GOLD, gold_ans=20)
    int_ = _internal_one(pred, gold=_NESTED_GOLD)
    assert off["official_partial_match"] == 1.0
    assert off["official_full_match"] == 1.0
    assert int_["full_sequence_accuracy"] == 1.0


@win_only
def test_scenario5_nested_win():
    pred = [dict(c) for c in _NESTED_GOLD]
    off = _official_one(pred, gold=_NESTED_GOLD, gold_ans=20)
    assert off["official_win"] == 1.0
    assert off["pred_answer"] == 20
    assert off["executable"] is True


# ---------------------------------------------------------------------------
# Scenario 6: missing variable reference -> safe Win=0, no crash
# ---------------------------------------------------------------------------

def test_scenario6_missing_ref_does_not_crash():
    pred = [
        {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
        {"name": "multiply", "label": "$var2", "arguments": {"a": "$var9.result$", "b": 10}},
    ]
    off = _official_one(pred)  # must not raise
    assert off["parse_valid"] is True


@win_only
def test_scenario6_missing_ref_win0():
    pred = [
        {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
        {"name": "multiply", "label": "$var2", "arguments": {"a": "$var9.result$", "b": 10}},
    ]
    off = _official_one(pred)
    assert off["official_win"] == 0.0
    assert off["executable"] is False
    assert off["execution_error"] == "execution_failed"


# ---------------------------------------------------------------------------
# Scenario 7: invalid JSON -> failure mode, eval continues
# ---------------------------------------------------------------------------

def test_scenario7_invalid_json_failure_and_continues():
    bad_item = {
        "generated_text": "this is not valid json {",
        "output": json.dumps(GOLD),
        "tools": json.dumps(TOOLS),
        "gold_answer": json.dumps(GOLD_ANS),
    }
    good_item = _make_item([dict(c) for c in GOLD])
    # A bad sample must not abort scoring of the rest.
    results = score_items_per_sample([bad_item, good_item],
                                     executable_func_dir=FUNC_DIR, win_rate=WANT_WIN)
    assert len(results) == 2
    bad, good = results
    assert bad["parse_valid"] is False
    assert bad["official_partial_match"] == 0.0
    assert bad["official_full_match"] == 0.0
    assert good["official_full_match"] == 1.0  # the good one is still scored

    # internal replica: empty/invalid prediction -> failure mode, never throws
    int_bad = compute_nestful_official_metrics([], GOLD)
    assert int_bad["partial_sequence_accuracy"] == 0.0
    assert int_bad["full_sequence_accuracy"] == 0.0
    assert int_bad["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# Corpus-level F1 aggregation (tested separately: official F1 is corpus macro)
# ---------------------------------------------------------------------------

def test_corpus_f1_official_all_correct():
    items = [
        _make_item([dict(c) for c in GOLD]),
        _make_item([dict(c) for c in _NESTED_GOLD], gold=_NESTED_GOLD, gold_ans=20),
    ]
    m = score_items(items, executable_func_dir=FUNC_DIR, win_rate=False)
    assert m["f1_func"] == pytest.approx(1.0)
    assert m["f1_param"] == pytest.approx(1.0)
    assert m["partial_sequence_accuracy"] == pytest.approx(1.0)
    assert m["full_sequence_accuracy"] == pytest.approx(1.0)


def test_corpus_f1_official_wrong_function_drops_f1():
    # Replace multiply with divide in the prediction -> function-name F1 < 1.
    pred = [
        {"name": "add", "label": "$var1", "arguments": {"a": 1, "b": 2}},
        {"name": "divide", "label": "$var2", "arguments": {"a": "$var1.result$", "b": 10}},
    ]
    items = [_make_item(pred)]
    m = score_items(items, executable_func_dir=FUNC_DIR, win_rate=False)
    assert m["f1_func"] < 1.0


def test_internal_corpus_macro_f1_matches_expectations():
    gold_lists = [["add", "multiply"], ["add"]]
    assert internal_corpus_macro_f1(gold_lists, gold_lists) == pytest.approx(1.0)
    # A wrong prediction set drops the macro-F1 below 1.
    pred_lists = [["add", "divide"], ["add"]]
    assert internal_corpus_macro_f1(gold_lists, pred_lists) < 1.0


# ---------------------------------------------------------------------------
# executor.py safe-failure behaviour (never raises; returns ExecResult.error)
# ---------------------------------------------------------------------------

def test_build_item_serializes_set_and_tuple_arguments():
    from nestful_official_score import build_item

    item = build_item(
        [{"name": "add", "arguments": {"a": {1, 2}, "b": (3, 4)}, "label": "$var1"}],
        {
            "output": [{"name": "add", "arguments": {"a": 1}, "label": "$var1"}],
            "tools": [{"name": "add", "output_parameters": {"result": {}}}],
            "gold_answer": 5,
        },
    )
    parsed = json.loads(item["generated_text"])
    assert parsed[0]["arguments"]["a"] == [1, 2] or parsed[0]["arguments"]["a"] == [2, 1]
    assert parsed[0]["arguments"]["b"] == [3, 4]


def test_json_field_str_quotes_bare_datetime_strings():
    """Bare string gold_answer values must survive json.loads in the official scorer."""
    from nestful_official_score import _json_field_str, build_item

    s = _json_field_str("022-01-01T00:00:00")
    assert json.loads(s) == "022-01-01T00:00:00"

    item = build_item([], {
        "output": [{"name": "add", "arguments": {"a": 1}, "label": "$var1"}],
        "tools": [{"name": "add", "output_parameters": {"result": {}}}],
        "gold_answer": "022-01-01T00:00:00",
    })
    assert json.loads(item["gold_answer"]) == "022-01-01T00:00:00"


def test_full_dataset_build_items_no_json_decode_error():
    """Every NESTFUL row must serialize for the official scorer without crashing."""
    from nestful_official_score import build_item, load_raw_dataset

    path = os.path.join(_HERE, "..", "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
    if not os.path.isfile(path):
        pytest.skip("full nestful dataset not present")
    raw = load_raw_dataset(path)
    for sid, row in raw.items():
        item = build_item([], row)
        json.loads(item["gold_answer"])
        json.loads(item["tools"])
        json.loads(item["output"])
    from executor import ToolExecutor

    task = {"tools": TOOLS, "gold_calls": GOLD, "gold_answer": GOLD_ANS}
    ex = ToolExecutor(task, registry=None, mode="gold_replay")

    r_unknown = ex.execute({"name": "nope", "arguments": {}})
    assert r_unknown.error and "unknown_tool" in r_unknown.error

    r_badargs = ex.execute({"name": "add", "arguments": "not-a-dict"})
    assert r_badargs.error == "invalid_arguments_type"

    r_unresolved = ex.execute({"name": "add", "arguments": {"a": "$missing$", "b": 2}})
    assert r_unresolved.error and "unresolved_variable" in r_unresolved.error
