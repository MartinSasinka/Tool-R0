"""Hard deterministic validators for Pilot4.2 query contracts."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable

from .query_render import FORBIDDEN_GRAPH_PHRASES, query_template_fingerprint


def _present(value: Any, text: str) -> bool:
    token = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    return re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", text, re.I) is not None


def validate_query(record: Dict[str, Any]) -> Dict[str, Any]:
    text = str(record.get("question") or record.get("query") or "")
    contract = record.get("query_contract") or {}
    facts = contract.get("facts") or []
    layers: Dict[str, Dict[str, Any]] = {}
    def add(name: str, errors: list[str]) -> None:
        layers[name] = {"passed": not errors, "reasons": errors}
    add("V_QUERY_FACTS", [f"missing fact {f.get('role')}" for f in facts
                          if not _present(f.get("value"), text)])
    expected_numbers = [f.get("value") for f in facts
                        if isinstance(f.get("value"), (int, float)) and not isinstance(f.get("value"), bool)]
    rendered_numbers = [float(x) for x in re.findall(r"(?<!\w)-?\d+(?:\.\d+)?", text)]
    add("V_QUERY_NUMBERS", [] if all(any(abs(float(v) - n) < 1e-9
                                             for n in rendered_numbers)
                                      for v in expected_numbers)
        else ["contract numbers are not preserved"])
    entity = str(contract.get("entity") or "")
    add("V_QUERY_ENTITIES", [] if entity.lower() in text.lower() else ["entity missing"])
    units = sorted({str(f.get("unit")) for f in facts if f.get("unit")})
    add("V_QUERY_UNITS", [f"unit {u} missing" for u in units
                          if u.lower() not in text.lower()])
    target = str((contract.get("natural_language_assets") or {}).get(
        "target_phrase") or contract.get("target_role") or "").replace("_", " ")
    add("V_QUERY_TARGET", [] if target.lower() in text.lower() else ["target missing"])
    low = text.lower()
    add("V_QUERY_GRAPH_LEAK", [p for p in FORBIDDEN_GRAPH_PHRASES if p in low])
    expected_fp = record.get("query_template_fingerprint")
    actual_fp = query_template_fingerprint(text)
    add("V_QUERY_TEMPLATE", [] if not expected_fp or expected_fp == actual_fp
        else ["template fingerprint mismatch"])
    return {"passed": all(v["passed"] for v in layers.values()), "layers": layers}


def validate_template_distribution(records: Iterable[Dict[str, Any]],
                                   max_share: float = .06) -> Dict[str, Any]:
    rows = list(records)
    counts = Counter(r.get("query_template_fingerprint") for r in rows)
    top = counts.most_common(1)[0][1] if counts else 0
    share = top / max(len(rows), 1)
    return {"passed": share <= max_share, "top_share": share,
            "n_templates": len(counts)}
