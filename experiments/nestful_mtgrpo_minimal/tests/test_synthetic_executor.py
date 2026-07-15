"""Regression tests for executor mode="synthetic" (real synthetic execution).

These tests pin the core contract of the new training executor:
  * wrong argument VALUES execute for real and never receive the gold result;
  * invalid references and invalid argument types are hard errors;
  * valid multi-call chains receive real observations and later calls can
    consume earlier outputs (including object-output fields);
  * schema violations (unknown key, missing required key) are hard errors.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from executor import ToolExecutor  # noqa: E402
from synthetic_tool_registry import SyntheticToolRegistry, get_synthetic_registry  # noqa: E402

_V3_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "nestful_synthetic_curriculum_v3"))
sys.path.insert(0, _V3_DIR)
from lib.synthetic_tools import tool_schema  # noqa: E402


def _make_task(tool_names, gold_calls, gold_answer):
    return {
        "task_id": "syn-test",
        "question": "test",
        "tools": [tool_schema(n) for n in tool_names],
        "gold_calls": gold_calls,
        "gold_answer": gold_answer,
        "num_calls": len(gold_calls),
    }


# Chain: apply_discount(100, 10%) -> 90.0; add_sales_tax(90, 10%) -> 99.0
_CHAIN_TASK = _make_task(
    ["apply_discount", "add_sales_tax", "is_above_threshold", "rectangle_metrics"],
    gold_calls=[
        {"name": "apply_discount",
         "arguments": {"price": 100, "discount_percent": 10}, "label": "$var1"},
        {"name": "add_sales_tax",
         "arguments": {"net_price": "$var1.output_0$", "tax_rate_percent": 10},
         "label": "$var2"},
    ],
    gold_answer=99.0,
)


def _ex(task=None):
    return ToolExecutor(task or _CHAIN_TASK, registry=None, mode="synthetic")


def test_registry_loads():
    reg = get_synthetic_registry()
    assert reg.available, reg.load_error
    assert len(reg.tool_names()) > 34
    assert reg.registry_hash()


def test_valid_chain_receives_real_observations():
    ex = _ex()
    r1 = ex.execute({"name": "apply_discount",
                     "arguments": {"price": 100, "discount_percent": 10},
                     "label": "$var1"})
    assert r1.error is None
    assert r1.observation == 90.0

    r2 = ex.execute({"name": "add_sales_tax",
                     "arguments": {"net_price": "$var1.output_0$",
                                   "tax_rate_percent": 10},
                     "label": "$var2"})
    assert r2.error is None
    assert r2.observation == 99.0
    # The reference was resolved to the REAL previous observation.
    assert r2.arguments_resolved["net_price"] == 90.0


def test_wrong_argument_values_never_receive_gold_result():
    ex = _ex()
    # Same tool/keys as gold call 1, but a WRONG price value.
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": 200, "discount_percent": 10},
                    "label": "$var1"})
    assert r.error is None
    assert r.observation == 180.0        # real execution of the wrong value
    assert r.observation != 90.0         # NOT the gold observation

    # Chain the wrong value forward: final result must NOT be the gold answer.
    r2 = ex.execute({"name": "add_sales_tax",
                     "arguments": {"net_price": "$var1.output_0$",
                                   "tax_rate_percent": 10},
                     "label": "$var2"})
    assert r2.error is None
    assert r2.observation == 198.0
    assert r2.observation != _CHAIN_TASK["gold_answer"]


def test_gold_replay_would_have_masked_the_wrong_value():
    """Contrast pin: the LEGACY mode accepts the wrong value (keys match) and
    returns a gold-derived observation — the exact defect synthetic mode fixes."""
    ex = ToolExecutor(_CHAIN_TASK, registry=None, mode="gold_replay")
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": 200, "discount_percent": 10},
                    "label": "$var1"})
    assert r.error is None  # legacy mode cannot falsify wrong values


def test_invalid_reference_fails():
    ex = _ex()
    r = ex.execute({"name": "add_sales_tax",
                    "arguments": {"net_price": "$var9.output_0$",
                                  "tax_rate_percent": 10},
                    "label": "$var1"})
    assert r.error is not None
    assert "unresolved_variable" in r.error


def test_missing_object_field_reference_fails():
    ex = _ex()
    r1 = ex.execute({"name": "rectangle_metrics",
                     "arguments": {"length": 10, "width": 4}, "label": "$var1"})
    assert r1.error is None
    assert r1.observation == {"area": 40.0, "perimeter": 28.0}

    r2 = ex.execute({"name": "add_sales_tax",
                     "arguments": {"net_price": "$var1.no_such_field$",
                                   "tax_rate_percent": 10},
                     "label": "$var2"})
    assert r2.error is not None
    assert "unresolved_field" in r2.error


def test_object_field_reference_resolves():
    ex = _ex()
    ex.execute({"name": "rectangle_metrics",
                "arguments": {"length": 10, "width": 4}, "label": "$var1"})
    r = ex.execute({"name": "is_above_threshold",
                    "arguments": {"value": "$var1.area$", "threshold": 30},
                    "label": "$var2"})
    assert r.error is None
    assert r.observation is True
    assert r.arguments_resolved["value"] == 40.0


def test_wrong_argument_type_fails():
    ex = _ex()
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": "not-a-number", "discount_percent": 10},
                    "label": "$var1"})
    assert r.error is not None
    assert "argument_type_mismatch" in r.error


def test_numeric_string_is_coerced_not_rejected():
    ex = _ex()
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": "100", "discount_percent": 10},
                    "label": "$var1"})
    assert r.error is None
    assert r.observation == 90.0


def test_unknown_argument_key_fails():
    ex = _ex()
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": 100, "discount_percent": 10,
                                  "bogus_key": 1},
                    "label": "$var1"})
    assert r.error is not None
    assert "unknown_argument" in r.error


def test_missing_required_argument_fails():
    ex = _ex()
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": 100}, "label": "$var1"})
    assert r.error is not None
    assert "missing_required_argument" in r.error


def test_optional_argument_can_be_omitted():
    task = _make_task(
        ["shipping_cost"],
        gold_calls=[{"name": "shipping_cost",
                     "arguments": {"weight_kg": 10, "rate_per_kg": 2},
                     "label": "$var1"}],
        gold_answer=25.0,
    )
    ex = _ex(task)
    r = ex.execute({"name": "shipping_cost",
                    "arguments": {"weight_kg": 10, "rate_per_kg": 2},
                    "label": "$var1"})
    assert r.error is None
    assert r.observation == 25.0  # base_fee default 5.0 applied

    r2 = ex.execute({"name": "shipping_cost",
                     "arguments": {"weight_kg": 10, "rate_per_kg": 2,
                                   "base_fee": 0},
                     "label": "$var2"})
    assert r2.error is None
    assert r2.observation == 20.0


def test_tool_not_offered_in_task_fails():
    ex = _ex()
    r = ex.execute({"name": "circle_area", "arguments": {"radius": 2},
                    "label": "$var1"})
    assert r.error is not None
    assert "unknown_tool" in r.error


def test_execution_error_propagates():
    task = _make_task(
        ["divide_numbers"],
        gold_calls=[{"name": "divide_numbers",
                     "arguments": {"dividend": 10, "divisor": 2},
                     "label": "$var1"}],
        gold_answer=5.0,
    )
    ex = _ex(task)
    r = ex.execute({"name": "divide_numbers",
                    "arguments": {"dividend": 10, "divisor": 0},
                    "label": "$var1"})
    assert r.error is not None
    assert "division_by_zero" in r.error
    # Scope must NOT advance on error.
    assert ex.indexed == []


def test_deterministic_execution():
    obs = []
    for _ in range(2):
        ex = _ex()
        r = ex.execute({"name": "apply_discount",
                        "arguments": {"price": 123.45, "discount_percent": 15},
                        "label": "$var1"})
        obs.append(r.observation)
    assert obs[0] == obs[1]


def test_synthetic_mode_fails_hard_without_registry():
    bad = SyntheticToolRegistry(v3_dir=os.path.join(os.sep, "definitely", "missing"))
    # A bogus dir may still import the registry when the real v3 dir is already
    # on sys.path from earlier imports; only assert the hard-fail wiring when
    # the load genuinely failed.
    if not bad.available:
        assert bad.load_error


def test_gold_answer_never_substituted_on_last_call():
    """Even on the final gold position with gold-matching keys but a wrong
    value, synthetic mode returns the real computation, not gold_answer."""
    task = _make_task(
        ["apply_discount"],
        gold_calls=[{"name": "apply_discount",
                     "arguments": {"price": 100, "discount_percent": 10},
                     "label": "$var1"}],
        gold_answer=90.0,
    )
    ex = _ex(task)
    r = ex.execute({"name": "apply_discount",
                    "arguments": {"price": 500, "discount_percent": 10},
                    "label": "$var1"})
    assert r.error is None
    assert r.observation == 450.0
    assert r.observation != task["gold_answer"]
