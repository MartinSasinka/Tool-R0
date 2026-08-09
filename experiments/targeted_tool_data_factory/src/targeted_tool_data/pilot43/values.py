"""Workflow-specific value generation.

Pilot4.2 sampled integers in 1..2000 for every role in every domain, which is
why the export contains 844 % discounts and inventory picks larger than stock.
Here every role declares a *hint*; the hint owns the realistic range, the unit,
the boundary behaviour and its own rejection rules. A threshold is never sampled
independently of the oracle: :func:`calibrate_predicate` reads the observed value
first and only then chooses the comparison constant, which is what makes boolean
balancing possible at all.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from . import semtypes as st

BOOL_BANDS = ("clear_true", "near_true", "near_false", "clear_false")


class ValueError_(ValueError):
    """Raised when a hint cannot produce an admissible value."""


# ── realistic entity vocabulary (no benchmark strings are copied) ─────────
PRODUCTS = ("desk lamp", "office chair", "laptop stand", "water filter",
            "cable tray", "monitor arm", "storage bin", "label printer",
            "floor mat", "air purifier", "keyboard tray", "hand sanitiser")
MATERIALS = ("aluminium sheet", "oak panel", "steel bracket", "acrylic pane",
             "copper wire", "rubber gasket", "glass insert", "pine board")
DEPARTMENTS = ("logistics", "facilities", "procurement", "support",
               "field service", "quality", "fabrication", "dispatch")
CITIES = ("Brno", "Leeds", "Aarhus", "Turin", "Gdansk", "Utrecht", "Lyon",
          "Porto", "Tampere", "Graz", "Malmo", "Nantes")
WAREHOUSES = ("north depot", "south depot", "east hub", "west hub",
              "central store", "overflow yard")
PEOPLE = ("Kovac", "Nielsen", "Marchetti", "Okafor", "Lindqvist", "Baranowski",
          "Haddad", "Fernandes", "Novak", "Ibrahim", "Sorensen", "Almeida")
PROJECTS = ("atlas", "beacon", "cedar", "delta", "ember", "fjord", "granite",
            "harbour", "ivory", "juniper")
FILE_STEMS = ("shipment_log", "inventory_snapshot", "route_plan", "audit_trail",
              "meter_reading", "batch_report", "service_ticket", "cost_sheet")
EXTENSIONS = ("csv", "json", "txt", "xml", "log", "yaml", "tsv")
DIRS = ("data", "exports", "archive", "reports", "staging", "inbound",
        "outbound", "warehouse", "finance", "ops")
HOSTS = ("depot.example.org", "reports.internal.example.com",
         "api.logistics.example.net", "files.example.co", "hub.example.io",
         "metrics.plant.example.org")
CURRENCIES = ("EUR", "GBP", "USD", "PLN", "CZK", "DKK")
UNIT_WORDS = ("kg", "units", "boxes", "pallets", "litres", "metres", "hours")
STATUSES = ("open", "closed", "pending", "escalated", "resolved")


def _round2(x: float) -> float:
    return round(float(x) + 0.0, 2)


# ── hint samplers ────────────────────────────────────────────────────────
def _money_price(rng: random.Random) -> float:
    return _round2(rng.choice([
        rng.uniform(3.5, 49.9), rng.uniform(50, 480), rng.uniform(480, 2400)]))


def _money_total(rng: random.Random) -> float:
    return _round2(rng.uniform(120, 18500))


def _money_budget(rng: random.Random) -> float:
    return _round2(rng.uniform(500, 42000))


def _money_fee(rng: random.Random) -> float:
    return _round2(rng.uniform(1.5, 240))


def _quantity(rng: random.Random) -> int:
    return rng.choice([rng.randint(1, 12), rng.randint(12, 96),
                       rng.randint(96, 640)])


def _count_small(rng: random.Random) -> int:
    return rng.randint(2, 5)


def _count_items(rng: random.Random) -> int:
    return rng.randint(3, 240)


def _count_people(rng: random.Random) -> int:
    return rng.randint(2, 48)


def _percent_discount(rng: random.Random) -> float:
    return _round2(rng.choice([rng.uniform(2, 12), rng.uniform(12, 30),
                               rng.uniform(30, 50)]))


def _percent_tax(rng: random.Random) -> float:
    return float(rng.choice([5.0, 7.0, 9.0, 12.0, 15.0, 19.0, 20.0, 21.0, 23.0]))


def _percent_growth(rng: random.Random) -> float:
    return _round2(rng.uniform(0.8, 24.0))


def _percent_margin(rng: random.Random) -> float:
    return _round2(rng.uniform(4.0, 38.0))


def _percent_share(rng: random.Random) -> float:
    return _round2(rng.uniform(5.0, 85.0))


def _ratio_target(rng: random.Random) -> float:
    return round(rng.uniform(0.15, 0.95), 3)


def _duration_hours(rng: random.Random) -> float:
    return _round2(rng.choice([rng.uniform(0.5, 8), rng.uniform(8, 72)]))


def _duration_days(rng: random.Random) -> int:
    return rng.choice([rng.randint(1, 14), rng.randint(14, 120)])


def _duration_minutes(rng: random.Random) -> int:
    return rng.choice([rng.randint(5, 90), rng.randint(90, 720)])


def _duration_seconds(rng: random.Random) -> int:
    return rng.randint(30, 5400)


def _length_m(rng: random.Random) -> float:
    return _round2(rng.uniform(0.4, 48.0))


def _length_km(rng: random.Random) -> float:
    return _round2(rng.uniform(1.5, 780.0))


def _mass_kg(rng: random.Random) -> float:
    return _round2(rng.uniform(0.3, 940.0))


def _mass_g(rng: random.Random) -> int:
    return rng.randint(25, 9800)


def _volume_l(rng: random.Random) -> float:
    return _round2(rng.uniform(0.5, 260.0))


def _volume_ml(rng: random.Random) -> int:
    return rng.randint(50, 4800)


def _temp_c(rng: random.Random) -> float:
    return _round2(rng.uniform(-14.0, 41.0))


def _temp_f(rng: random.Random) -> float:
    return _round2(rng.uniform(18.0, 104.0))


def _bytes_size(rng: random.Random) -> int:
    return rng.choice([rng.randint(512, 65536), rng.randint(65536, 8_400_000)])


def _score_points(rng: random.Random) -> float:
    return _round2(rng.uniform(12.0, 98.0))


def _index_position(rng: random.Random) -> int:
    return rng.randint(1, 3)


def _places(rng: random.Random) -> int:
    return rng.choice([0, 1, 2])


def _number_list(lo: float, hi: float, n_lo: int = 4, n_hi: int = 7,
                 integral: bool = False) -> Callable[[random.Random], List[Any]]:
    def gen(rng: random.Random) -> List[Any]:
        n = rng.randint(n_lo, n_hi)
        vals = [rng.uniform(lo, hi) for _ in range(n)]
        out = [int(round(v)) for v in vals] if integral else [_round2(v) for v in vals]
        if len(set(out)) < max(2, n - 1):
            out = sorted(set(out)) or out
        return out
    return gen


def _text_label(rng: random.Random) -> str:
    return rng.choice(PRODUCTS + MATERIALS)


def _text_note(rng: random.Random) -> str:
    subject = rng.choice(PRODUCTS + MATERIALS)
    place = rng.choice(WAREHOUSES + CITIES)
    return f"  {subject} logged at {place}   by {rng.choice(PEOPLE)} "


def _identifier_code(rng: random.Random) -> str:
    return (f"{rng.choice(PROJECTS)}-{rng.choice(DEPARTMENTS).replace(' ', '')}"
            f"-{rng.randint(1000, 9999)}")


def _delimited_record(rng: random.Random) -> str:
    return (f"{rng.choice(PROJECTS)}|{rng.choice(DEPARTMENTS)}|"
            f"{rng.randint(10, 999)}|{rng.choice(STATUSES)}")


def _separator(rng: random.Random) -> str:
    return rng.choice(["|", ";", ",", "/", "::"])


def _path_file(rng: random.Random) -> str:
    depth = rng.randint(2, 4)
    parts = [rng.choice(DIRS) for _ in range(depth)]
    stem = rng.choice(FILE_STEMS)
    ext = rng.choice(EXTENSIONS)
    lead = "/" if rng.random() < 0.75 else ""
    noisy = rng.random() < 0.35
    body = "/".join(parts)
    if noisy:
        body = body.replace("/", "/./", 1)
    return f"{lead}{body}/{stem}_{rng.randint(2, 48)}.{ext}"


def _path_dir(rng: random.Random) -> str:
    depth = rng.randint(2, 4)
    return "/" + "/".join(rng.choice(DIRS) for _ in range(depth))


def _url_link(rng: random.Random) -> str:
    scheme = rng.choice(["https", "https", "http"])
    host = rng.choice(HOSTS)
    port = rng.choice(["", "", "", ":8443", ":8080"])
    path = "/" + "/".join(rng.choice(DIRS) for _ in range(rng.randint(1, 3)))
    query = rng.choice([
        "", f"?region={rng.choice(CITIES).lower()}&page={rng.randint(1, 9)}",
        f"?depot={rng.choice(WAREHOUSES).replace(' ', '-')}&limit={rng.randint(10, 90)}"])
    return f"{scheme}://{host}{port}{path}{query}"


def _date_iso(rng: random.Random) -> str:
    year = rng.randint(2023, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _mapping_rates(rng: random.Random) -> Dict[str, float]:
    keys = rng.sample(list(DEPARTMENTS), rng.randint(3, 5))
    return {k: _round2(rng.uniform(4.0, 42.0)) for k in keys}


def _mapping_amounts(rng: random.Random) -> Dict[str, float]:
    keys = rng.sample(list(WAREHOUSES), rng.randint(3, 5))
    return {k: _round2(rng.uniform(80.0, 9400.0)) for k in keys}


def _mapping_counts(rng: random.Random) -> Dict[str, int]:
    keys = rng.sample(list(PRODUCTS), rng.randint(3, 5))
    return {k: rng.randint(2, 320) for k in keys}


def _record_row(rng: random.Random) -> Dict[str, Any]:
    return {"label": rng.choice(PRODUCTS), "amount": _round2(rng.uniform(12, 940)),
            "site": rng.choice(WAREHOUSES), "units": rng.randint(2, 180)}


def _record_list(rng: random.Random) -> List[Dict[str, Any]]:
    n = rng.randint(3, 6)
    sites = rng.sample(list(WAREHOUSES), min(n, len(WAREHOUSES)))
    rows = []
    for i in range(n):
        rows.append({"label": rng.choice(PRODUCTS),
                     "amount": _round2(rng.uniform(15, 2400)),
                     "site": sites[i % len(sites)],
                     "units": rng.randint(1, 260)})
    return rows


def _text_list_labels(rng: random.Random) -> List[str]:
    pool = list(PRODUCTS + MATERIALS)
    return rng.sample(pool, rng.randint(3, 6))


def _text_list_codes(rng: random.Random) -> List[str]:
    return [f"{rng.choice(PROJECTS)}-{rng.randint(100, 999)}"
            for _ in range(rng.randint(3, 6))]


HINTS: Dict[str, Tuple[str, Callable[[random.Random], Any]]] = {
    # name -> (semantic type, sampler)
    "money_price": (st.MONEY, _money_price),
    "money_total": (st.MONEY, _money_total),
    "money_budget": (st.MONEY, _money_budget),
    "money_fee": (st.MONEY, _money_fee),
    "quantity_units": (st.QUANTITY, _quantity),
    "quantity_stock": (st.QUANTITY, _quantity),
    "count_items": (st.COUNT, _count_items),
    "count_people": (st.COUNT, _count_people),
    "count_small": (st.COUNT, _count_small),
    "places": (st.COUNT, _places),
    "index_position": (st.INDEX, _index_position),
    "percent_discount": (st.PERCENTAGE, _percent_discount),
    "percent_tax": (st.PERCENTAGE, _percent_tax),
    "percent_growth": (st.PERCENTAGE, _percent_growth),
    "percent_margin": (st.PERCENTAGE, _percent_margin),
    "percent_share": (st.PERCENTAGE, _percent_share),
    "ratio_target": (st.RATIO, _ratio_target),
    "duration_hours": (st.DUR_H, _duration_hours),
    "duration_days": (st.DUR_D, _duration_days),
    "duration_minutes": (st.DUR_MIN, _duration_minutes),
    "duration_seconds": (st.DUR_S, _duration_seconds),
    "length_m": (st.LEN_M, _length_m),
    "length_km": (st.LEN_KM, _length_km),
    "mass_kg": (st.MASS_KG, _mass_kg),
    "mass_g": (st.MASS_G, _mass_g),
    "volume_l": (st.VOL_L, _volume_l),
    "volume_ml": (st.VOL_ML, _volume_ml),
    "temp_c": (st.TEMP_C, _temp_c),
    "temp_f": (st.TEMP_F, _temp_f),
    "bytes_size": (st.BYTES, _bytes_size),
    "score_points": (st.SCORE, _score_points),
    "generic_value": (st.GENERIC, lambda rng: _round2(rng.uniform(6, 940))),
    "list_prices": (st.NUMBER_LIST, _number_list(4.0, 780.0)),
    "list_readings": (st.NUMBER_LIST, _number_list(0.8, 96.0)),
    "list_quantities": (st.NUMBER_LIST, _number_list(2, 320, integral=True)),
    "list_durations_h": (st.NUMBER_LIST, _number_list(0.5, 40.0)),
    "text_label": (st.TEXT, _text_label),
    "text_note": (st.TEXT, _text_note),
    "text_record": (st.TEXT, _delimited_record),
    "separator": (st.TEXT, _separator),
    "identifier_code": (st.TEXT, _identifier_code),
    "currency_code": (st.TEXT, lambda rng: rng.choice(CURRENCIES)),
    "unit_word": (st.TEXT, lambda rng: rng.choice(UNIT_WORDS)),
    "extension": (st.TEXT, lambda rng: rng.choice(EXTENSIONS)),
    "scheme": (st.TEXT, lambda rng: rng.choice(["https", "http"])),
    "host": (st.TEXT, lambda rng: rng.choice(HOSTS)),
    "field_name": (st.TEXT, lambda rng: rng.choice(["amount", "units"])),
    "text_field_name": (st.TEXT, lambda rng: rng.choice(["label", "site"])),
    "path_file": (st.PATH, _path_file),
    "path_dir": (st.PATH, _path_dir),
    "url_link": (st.URL, _url_link),
    "date_iso": (st.DATE, _date_iso),
    "mapping_rates": (st.MAPPING, _mapping_rates),
    "mapping_amounts": (st.MAPPING, _mapping_amounts),
    "mapping_counts": (st.MAPPING, _mapping_counts),
    "record_row": (st.RECORD, _record_row),
    "record_list": (st.RECORD_LIST, _record_list),
    "text_list_labels": (st.TEXT_LIST, _text_list_labels),
    "text_list_codes": (st.TEXT_LIST, _text_list_codes),
    # calibrated roles: sampled provisionally, then fixed against the oracle
    "threshold_value": (st.GENERIC, lambda rng: _round2(rng.uniform(50, 900))),
    "threshold_money": (st.MONEY, lambda rng: _round2(rng.uniform(120, 9000))),
    "threshold_count": (st.COUNT, lambda rng: rng.randint(3, 90)),
    "threshold_hours": (st.DUR_H, lambda rng: _round2(rng.uniform(2, 60))),
    "threshold_ratio": (st.RATIO, _ratio_target),
    "threshold_percent": (st.PERCENTAGE, lambda rng: _round2(rng.uniform(3, 40))),
    "tolerance_value": (st.GENERIC, lambda rng: _round2(rng.uniform(2, 40))),
    "range_low": (st.GENERIC, lambda rng: _round2(rng.uniform(10, 200))),
    "range_high": (st.GENERIC, lambda rng: _round2(rng.uniform(200, 900))),
    "cut_low": (st.GENERIC, lambda rng: _round2(rng.uniform(20, 200))),
    "cut_high": (st.GENERIC, lambda rng: _round2(rng.uniform(200, 900))),
    "date_deadline": (st.DATE, _date_iso),
    "prefix_text": (st.TEXT, lambda rng: rng.choice(PROJECTS)),
    "needle_text": (st.TEXT, lambda rng: rng.choice(["depot", "logged", "hub"])),
}

#: hints whose value must be derived from the oracle instead of sampled freely
CALIBRATED_HINTS = frozenset({
    "threshold_value", "threshold_money", "threshold_count", "threshold_hours",
    "threshold_ratio", "threshold_percent", "tolerance_value", "range_low",
    "range_high", "cut_low", "cut_high", "date_deadline", "prefix_text",
    "needle_text",
})


def sem_of_hint(hint: str) -> str:
    if hint not in HINTS:
        raise ValueError_(f"unknown value hint {hint!r}")
    return HINTS[hint][0]


def sample_hint(hint: str, rng: random.Random) -> Any:
    if hint not in HINTS:
        raise ValueError_(f"unknown value hint {hint!r}")
    return HINTS[hint][1](rng)


# ── boundary-aware predicate calibration ─────────────────────────────────
@dataclass(frozen=True)
class Band:
    """How far the observed value must sit from the comparison constant."""
    name: str
    want: bool

    @property
    def near(self) -> bool:
        return self.name in ("near_true", "near_false")


def band_for(want: bool, near: bool) -> Band:
    if want:
        return Band("near_true" if near else "clear_true", True)
    return Band("near_false" if near else "clear_false", False)


def _offset(value: float, band: Band, rng: random.Random) -> float:
    scale = max(abs(value), 1.0)
    if band.near:
        return _round2(scale * rng.uniform(0.01, 0.06) + 0.05)
    return _round2(scale * rng.uniform(0.25, 0.7) + 2.0)


def _integral(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _constant(target: float, like: Any) -> Any:
    """A comparison constant in the same value domain as what it is compared to.

    A weekday index, a permission mask and an item count are integers, and a
    calibrated limit of 2.09 weekdays or 352.15 of a bitmask is not a hard sample
    but an impossible one -- the second critic rejected exactly those on
    ``units_semantically_valid``. Never negative, for the same reason.
    """
    if _integral(like):
        return max(0, int(round(target)))
    return _round2(target)


def _int_delta(delta: float) -> int:
    """At least one whole unit, so an integer comparison actually separates."""
    return max(1, int(round(delta)))


def coerce_constant(sem: str, value: Any) -> Any:
    """Force a calibrated constant into the value domain of the role that holds it.

    Calibration works from the executed value, which may be a float even when the
    role is a count of pallets; the constant is what the query states, so it has to
    obey the role's own semantics. A coercion that would flip the intended verdict
    is caught by the re-execution that follows, and the candidate is dropped.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if sem in (st.COUNT, st.QUANTITY):
        return max(0, int(round(float(value))))
    if sem == st.MONEY or (sem in st.PHYSICAL
                           and sem not in (st.TEMP_C, st.TEMP_F)):
        return max(0.01, _round2(value))
    if sem == st.PERCENTAGE:
        return _round2(min(400.0, max(-100.0, float(value))))
    if sem == st.RATIO:
        return round(min(5.0, max(-5.0, float(value))), 4)
    return value


def calibrate_predicate(capability: str, params: Sequence[str],
                        observed: Dict[str, Any], band: Band,
                        rng: random.Random) -> Dict[str, Any] | None:
    """Constants for a boolean node so that it evaluates to ``band.want``.

    ``params`` are the node's parameter names in declaration order and
    ``observed`` holds the already-executed values of its inputs, so the
    comparison constant is always chosen *after* the oracle value is known.
    Positional addressing keeps this independent of surface naming.
    Returns argument overrides, or None when the capability is not calibratable.
    """
    want = band.want
    data = observed.get(params[0]) if params else None

    if capability in ("comparison.at_least", "comparison.greater"):
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return None
        v = float(data)
        delta = _offset(v, band, rng)
        if _integral(data):
            step = _int_delta(delta)
            if want and int(data) - step < 0:
                step = int(data)          # 0 still satisfies "reaches at least"
            return {params[1]: _constant(int(data) - step if want
                                         else int(data) + step, data)}
        return {params[1]: _round2(v - delta if want else v + delta)}

    if capability == "validation.in_range":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return None
        v = float(data)
        span = _offset(v, band, rng)
        if _integral(data):
            span = _int_delta(span)
        if want:
            return {params[1]: _constant(v - span - 1.0, data),
                    params[2]: _constant(v + span + 1.0, data)}
        if rng.random() < 0.5:
            return {params[1]: _constant(v + span, data),
                    params[2]: _constant(v + span * 2.5 + 5.0, data)}
        return {params[1]: _constant(v - span * 2.5 - 5.0, data),
                params[2]: _constant(v - span, data)}

    if capability == "validation.tolerance":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return None
        v = float(data)
        tol = max(1.0, _offset(v, Band("clear_true", True), rng) * 0.4)
        drift = tol * (0.45 if want else 2.4)
        target = v + drift * (1 if rng.random() < 0.5 else -1)
        if _integral(data):
            # both constants have to be whole for an integer quantity, and the
            # tolerance has to stay wide enough to keep the intended verdict
            tol_i = max(1, int(round(tol)))
            drift_i = int(round(tol_i * (0.45 if want else 2.6))) or (
                0 if want else tol_i + 1)
            sign = 1 if rng.random() < 0.5 else -1
            return {params[1]: _constant(int(data) + sign * drift_i, data),
                    params[2]: tol_i}
        return {params[1]: _round2(target), params[2]: _round2(tol)}

    if capability == "validation.list_limit":
        if not isinstance(data, list) or not data:
            return None
        vals = [float(x) for x in data]
        peak, low = max(vals), min(vals)
        delta = _offset(peak, band, rng)
        integral = all(_integral(x) for x in data)
        if integral:
            step = _int_delta(delta)
            return {params[1]: int(peak) + step if want
                    else max(int(low) - 1, int(peak) - step)}
        return {params[1]: _round2(peak + delta if want
                                   else max(low - 0.01, peak - delta))}

    if capability == "boolean.divisible":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return None
        iv = int(round(float(data)))
        if abs(float(data) - iv) > 1e-9 or iv == 0:
            return None
        candidates = [k for k in (2, 3, 4, 5, 6, 7, 8, 9, 11, 12)
                      if (iv % k == 0) == want]
        return {params[1]: rng.choice(candidates)} if candidates else None

    if capability == "date.compare":
        if not isinstance(data, str):
            return None
        from datetime import date as _d, timedelta as _td
        try:
            d = _d.fromisoformat(data)
        except ValueError:
            return None
        shift = rng.randint(1, 3) if band.near else rng.randint(20, 200)
        other = d + _td(days=shift) if want else d - _td(days=shift)
        return {params[1]: other.isoformat()}

    if capability == "string.validate_prefix":
        if not isinstance(data, str) or len(data.strip()) < 3:
            return None
        text = data
        if want:
            cut = min(len(text), rng.randint(2, 5))
            return {params[1]: text[:cut]}
        return {params[1]: rng.choice(["zz-", "qx_", "unknown-"])}

    if capability == "string.validate_contains":
        if not isinstance(data, str) or len(data) < 5:
            return None
        if want:
            start = rng.randint(0, max(0, len(data) - 4))
            return {params[1]: data[start:start + rng.randint(3, 4)]}
        return {params[1]: rng.choice(["zzq", "qxv", "wkj"])}

    if capability == "path.validate_extension":
        if not isinstance(data, str) or "." not in data:
            return None
        actual = data.rsplit(".", 1)[1]
        if want:
            return {params[1]: actual}
        return {params[1]: rng.choice([e for e in EXTENSIONS if e != actual])}

    return None


def calibratable(capability: str) -> bool:
    return capability in {
        "comparison.at_least", "comparison.greater", "validation.in_range",
        "validation.tolerance", "validation.list_limit", "boolean.divisible",
        "date.compare", "string.validate_prefix", "string.validate_contains",
        "path.validate_extension",
    }


#: boolean ops whose value is decided by upstream boolean parents
COMBINATORS = {
    "boolean.and": ("all", 2), "boolean.or": ("any", 2),
    "boolean.not": ("not", 1), "boolean.xor": ("xor", 2),
    "decision.all_of": ("all", 3), "decision.any_of": ("any", 3),
    "decision.majority": ("majority", 3),
}


def parent_targets(kind: str, want: bool, n: int,
                   rng: random.Random) -> List[bool] | None:
    """Truth values for the parents of a combinator that yield ``want``."""
    if kind == "all":
        if want:
            return [True] * n
        flips = rng.sample(range(n), rng.randint(1, n))
        return [i not in flips for i in range(n)]
    if kind == "any":
        if not want:
            return [False] * n
        keep = rng.sample(range(n), rng.randint(1, n))
        return [i in keep for i in range(n)]
    if kind == "not":
        return [not want]
    if kind == "xor":
        if want:
            first = rng.random() < 0.5
            return [first, not first]
        both = rng.random() < 0.5
        return [both, both]
    if kind == "majority":
        if want:
            n_true = rng.choice([2, 3]) if n == 3 else n
        else:
            n_true = rng.choice([0, 1]) if n == 3 else 0
        idx = rng.sample(range(n), n_true)
        return [i in idx for i in range(n)]
    return None
