"""Invariants of the Pilot4.3 semantic-type lattice and the op registry.

Nothing here mutates production state: ``build_ops()`` is cached by the module
under test, so every test that needs to perturb the registry works on a copy.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List

import pytest

from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43 import values as V
from targeted_tool_data.pilot43.ops import (PRESERVE, Op, build_ops, registry_hash,
                                            validate_ops)

OPS: Dict[str, Op] = build_ops()
TRACKS = ("A_NATIVE", "G_GENERAL_1", "G_GENERAL_2")
PIDS = sorted(OPS)

FULL_SURFACE_PIDS = [pid for pid in PIDS if len(OPS[pid].surfaces) == 3]


# ---------------------------------------------------------------------------
# value sampling per semantic type (shared by the property test)
# ---------------------------------------------------------------------------
def _semantic_samplers() -> Dict[str, List[Callable[[random.Random], Any]]]:
    """One or more admissible value generators per semantic type."""
    out: Dict[str, List[Callable[[random.Random], Any]]] = {}
    for _hint, (sem, fn) in V.HINTS.items():
        out.setdefault(sem, []).append(fn)
    out[st.IDENTIFIER] = [lambda r: f"{r.choice(V.PROJECTS)}-{r.randint(10, 99)}"]
    out[st.CATEGORY] = [lambda r: r.choice(("low", "medium", "high"))]
    out[st.UNIT_NAME] = [lambda r: r.choice(V.UNIT_WORDS)]
    out[st.FLAG] = [lambda r: r.random() < 0.5]
    out[st.AREA] = [lambda r: round(r.uniform(0.5, 400.0), 2)]
    out[st.NUMERIC_TEXT] = [lambda r: f"{r.uniform(1.0, 900.0):.2f}"]
    return out


SAMPLERS = _semantic_samplers()


def test_every_semantic_type_has_a_sampler():
    """The property test below is only meaningful if it can cover every type."""
    assert sorted(st.ALL) == sorted(SAMPLERS)


# ---------------------------------------------------------------------------
# registry shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pid", PIDS)
def test_capability_is_family_dot_name(pid: str):
    cap = OPS[pid].capability
    assert cap, f"{pid} has no capability"
    family, _, name = cap.partition(".")
    assert family and name and "." not in name, cap
    assert OPS[pid].family == family


@pytest.mark.parametrize("pid", PIDS)
def test_arity_matches_parameter_count(pid: str):
    op = OPS[pid]
    assert op.arity == len(op.params) >= 1
    assert len({p.name for p in op.params}) == op.arity


@pytest.mark.parametrize("pid", PIDS)
def test_surface_arity_matches_op_arity(pid: str):
    op = OPS[pid]
    for surf in op.surfaces:
        assert len(surf.param_names) == op.arity, f"{pid}/{surf.name}"
        assert surf.name and surf.description


@pytest.mark.parametrize("pid", FULL_SURFACE_PIDS)
def test_declared_surfaces_carry_the_three_tracks(pid: str):
    assert OPS[pid].tracks() == TRACKS


def test_every_op_has_exactly_three_surfaces():
    missing = [pid for pid in PIDS if OPS[pid].tracks() != TRACKS]
    assert missing == []


def test_surface_lookup_never_falls_back_to_another_track():
    wrong = [(pid, track) for pid in PIDS for track in TRACKS
             if OPS[pid].surface(track).track != track]
    assert wrong == []


def test_an_unknown_track_raises_instead_of_returning_a_native_surface():
    with pytest.raises(KeyError):
        OPS[PIDS[0]].surface("G_GENERAL_9")


def test_surface_names_are_unique_across_the_registry():
    seen: Dict[str, str] = {}
    clashes = []
    for pid in PIDS:
        for surf in OPS[pid].surfaces:
            if surf.name in seen and seen[surf.name] != pid:
                clashes.append((surf.name, seen[surf.name], pid))
            seen[surf.name] = pid
    assert clashes == []


@pytest.mark.parametrize("pid", PIDS)
def test_fn_is_callable(pid: str):
    assert callable(OPS[pid].fn)


@pytest.mark.parametrize("pid", PIDS)
def test_declared_semantic_types_exist(pid: str):
    op = OPS[pid]
    for p in op.params:
        assert p.sem in st.ALL, f"{pid}.{p.name}: {p.sem}"
    assert op.out_sem == PRESERVE or op.out_sem in st.ALL


def test_validate_ops_reports_no_error():
    assert validate_ops(OPS) == []


def test_validate_ops_catches_a_surface_collision():
    """The gate must be fail-closed, not merely quiet on a clean registry."""
    ops = dict(OPS)
    victim, other = "add", "subtract"
    clashing = tuple(
        s.__class__(track=s.track, name=OPS[victim].surfaces[0].name,
                    param_names=s.param_names, description=s.description,
                    output_field=s.output_field)
        for s in OPS[other].surfaces)
    ops[other] = OPS[other].__class__(
        pid=other, capability=OPS[other].capability, params=OPS[other].params,
        out_sem=OPS[other].out_sem, fn=OPS[other].fn, surfaces=clashing,
        source=OPS[other].source)
    assert validate_ops(ops)


def test_registry_hash_is_stable_and_content_sensitive():
    first, second = registry_hash(OPS), registry_hash(OPS)
    assert first == second
    reduced = dict(OPS)
    reduced.pop("add")
    assert registry_hash(reduced) != first


# ---------------------------------------------------------------------------
# semantic-type compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sem", sorted(st.ALL))
def test_compatible_is_reflexive(sem: str):
    assert st.compatible(sem, sem)


@pytest.mark.parametrize("sem", sorted(st.NUMERIC))
def test_generic_parameter_accepts_every_numeric_quantity(sem: str):
    assert st.compatible(st.GENERIC, sem)


@pytest.mark.parametrize("sem", sorted(st.TEXTUAL | st.COLLECTIONS | {st.FLAG}))
def test_generic_parameter_rejects_non_numeric_values(sem: str):
    """GenericScalar is the arithmetic escape hatch, not an ``Any``."""
    assert not st.compatible(st.GENERIC, sem)


#: pairs that must never be interchangeable in either direction
INCOMPATIBLE_PAIRS = [
    (st.MONEY, st.DUR_H), (st.MONEY, st.LEN_KM), (st.MONEY, st.PERCENTAGE),
    (st.PERCENTAGE, st.RATIO), (st.DUR_H, st.DUR_MIN), (st.DUR_S, st.DUR_D),
    (st.LEN_M, st.LEN_KM), (st.MASS_KG, st.MASS_G), (st.VOL_L, st.VOL_ML),
    (st.TEMP_C, st.TEMP_F), (st.BYTES, st.COUNT), (st.NUMBER_LIST, st.TEXT_LIST),
    (st.RECORD, st.MAPPING),
]


@pytest.mark.parametrize("left,right", INCOMPATIBLE_PAIRS)
def test_unit_bearing_types_are_not_mutually_compatible(left: str, right: str):
    assert not st.compatible(left, right)
    assert not st.compatible(right, left)


@pytest.mark.parametrize("sem", sorted(st.PHYSICAL))
def test_physical_parameters_never_accept_a_generic_scalar(sem: str):
    assert not st.compatible(sem, st.GENERIC)


def test_compatible_rejects_unknown_types():
    with pytest.raises(ValueError):
        st.compatible("NotAType", st.GENERIC)


# ---------------------------------------------------------------------------
# property test over the whole registry
# ---------------------------------------------------------------------------
def _degenerate(value: Any) -> str | None:
    if value is None:
        return "returned None"
    if isinstance(value, (set, tuple)):
        return f"returned a {type(value).__name__}"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "returned NaN/Inf"
    return None


#: see :func:`test_floor_divide_returns_the_declared_integer`
KNOWN_TYPE_DEFECTS = {"floor_divide"}


def _op_cases(n: int, seed: int) -> List[Any]:
    """``n`` op ids, every op represented at least once."""
    rng = random.Random(seed)
    cases = list(PIDS)
    while len(cases) < n:
        cases.append(rng.choice(PIDS))
    rng.shuffle(cases)
    return [pytest.param(pid, marks=pytest.mark.xfail(
                reason="floor_divide declares Count and returns a float",
                strict=True))
            if pid in KNOWN_TYPE_DEFECTS else pid
            for pid in cases[:max(n, len(PIDS))]]


@pytest.mark.parametrize("pid", _op_cases(300, seed=20260731))
def test_op_output_matches_its_declared_semantic_type_or_raises(pid: str):
    op = OPS[pid]
    rng = random.Random(f"{pid}:20260731")
    for _ in range(3):
        sems = [p.sem for p in op.params]
        kwargs = {p.name: rng.choice(SAMPLERS[p.sem])(rng) for p in op.params}
        try:
            out = op.fn(**kwargs)
        except Exception:      # noqa: BLE001 - raising is an accepted outcome
            continue
        problem = _degenerate(out)
        assert problem is None, f"{pid}({kwargs}) {problem}"
        out_sem = op.resolve_out_sem(sems)
        assert st.matches_value(out_sem, out), (
            f"{pid}({kwargs}) -> {out!r} is {st.value_kind(out)}, "
            f"declared {out_sem}")


@pytest.mark.xfail(reason="arithmetic.floor_divide declares out_sem Count "
                          "(runtime integer) but its base primitive always "
                          "returns a float, so check_value_types rejects any "
                          "program using it and the V4 frontier silently drops "
                          "every floor_divide result",
                   strict=True)
def test_floor_divide_returns_the_declared_integer():
    op = OPS["floor_divide"]
    out = op.fn(a=130.02, b=132.48)
    assert st.matches_value(op.resolve_out_sem([st.GENERIC, st.GENERIC]), out)


def test_resolve_out_sem_keeps_the_quantity_and_skips_rate_inputs():
    """``percent_of(20 %, 500 EUR)`` is money, never a percentage."""
    op = OPS["percent_of"]
    assert op.out_sem == PRESERVE
    assert op.resolve_out_sem([st.PERCENTAGE, st.MONEY]) == st.MONEY
    assert op.resolve_out_sem([st.PERCENTAGE, st.GENERIC]) == st.GENERIC
