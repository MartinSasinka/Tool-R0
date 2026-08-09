"""Registry / operation coverage proxies linking train to diagnostic outcomes."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .statistics import call_bucket, mean
from .surface_features import jaccard, normalize_tool_name, split_name_tokens, tool_surface_record


def _schema_sig(tool: Dict[str, Any]) -> Tuple[int, Tuple[str, ...], Tuple[str, ...]]:
    rec = tool_surface_record(tool)
    return (
        int(rec["n_params"]),
        tuple(sorted(rec["param_types"])),
        tuple(sorted(rec["output_types"])),
    )


def map_tool_confidence(
    diag_name: str,
    train_names: Set[str],
    train_norm: Dict[str, str],
    train_schema: Dict[str, Tuple],
    diag_tool: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if diag_name in train_names:
        return {"confidence": "EXACT", "mapped_to": diag_name, "level": "exact"}
    dn = normalize_tool_name(diag_name)
    if dn in train_norm:
        return {"confidence": "HIGH_PROXY", "mapped_to": train_norm[dn], "level": "normalized_lexical"}
    # token overlap proxy
    dt = set(split_name_tokens(diag_name))
    best_name, best = None, 0.0
    for tn in train_names:
        sc = jaccard(list(dt), split_name_tokens(tn))
        if sc > best:
            best, best_name = sc, tn
    if best >= 0.8:
        return {"confidence": "HIGH_PROXY", "mapped_to": best_name, "score": best, "level": "token_jaccard"}
    if best >= 0.5:
        return {"confidence": "MEDIUM_PROXY", "mapped_to": best_name, "score": best, "level": "token_jaccard"}
    # schema proxy
    if diag_tool is not None:
        sig = _schema_sig(diag_tool)
        for name, tsig in train_schema.items():
            if tsig == sig and sig[0] > 0:
                return {"confidence": "MEDIUM_PROXY", "mapped_to": name, "level": "schema"}
    if best >= 0.3:
        return {"confidence": "LOW_PROXY", "mapped_to": best_name, "score": best, "level": "token_jaccard"}
    return {"confidence": "UNMAPPED", "mapped_to": None, "level": "none"}


def build_train_indexes(train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    names: Set[str] = set()
    offered: Set[str] = set()
    schema: Dict[str, Tuple] = {}
    family: Dict[str, str] = {}
    for row in train_rows:
        for c in row.get("gold_calls") or []:
            n = str(c.get("name") or "")
            names.add(n)
        for t in row.get("tools") or []:
            if not isinstance(t, dict):
                continue
            n = str(t.get("name") or "")
            offered.add(n)
            schema[n] = _schema_sig(t)
            # provenance family if present on tools — rare; use name token head
            toks = split_name_tokens(n)
            if toks:
                family[n] = toks[0]
        prov = row.get("provenance") or {}
        if isinstance(prov, dict) and prov.get("semantic_program_family"):
            # attach to gold tools of this row
            for c in row.get("gold_calls") or []:
                family[str(c.get("name") or "")] = str(prov["semantic_program_family"])
    norm = {normalize_tool_name(n): n for n in names | offered}
    return {
        "gold_names": names,
        "offered_names": offered,
        "all_names": names | offered,
        "norm": norm,
        "schema": schema,
        "family": family,
    }


def task_coverage_features(
    diag_row: Dict[str, Any],
    train_idx: Dict[str, Any],
    *,
    topology_in_train: bool,
    outcome: str,
    c0_win: bool,
    d1_win: bool,
) -> Dict[str, Any]:
    gold = diag_row.get("output") or []
    tools = {str(t.get("name") or ""): t for t in (diag_row.get("tools") or []) if isinstance(t, dict)}
    mappings = []
    unmapped = 0
    exact = 0
    proxy = 0
    for c in gold:
        name = str(c.get("name") or "")
        m = map_tool_confidence(
            name,
            train_idx["all_names"],
            train_idx["norm"],
            train_idx["schema"],
            tools.get(name),
        )
        mappings.append({"tool": name, **m})
        if m["confidence"] == "EXACT":
            exact += 1
        elif m["confidence"] == "UNMAPPED":
            unmapped += 1
        else:
            proxy += 1
    n = len(gold) or 1
    exact_rate = exact / n
    proxy_schema_rate = (exact + proxy) / n
    # critical path ~ all gold tools in linear programs; flag if any unmapped
    unmapped_on_critical = unmapped > 0
    ood = (
        0.45 * (1.0 - exact_rate)
        + 0.25 * (1.0 - proxy_schema_rate)
        + 0.20 * (0.0 if topology_in_train else 1.0)
        + 0.10 * (unmapped / n)
    )
    return {
        "sample_id": str(diag_row.get("sample_id")),
        "n_gold_tools": len(gold),
        "exact_tool_coverage_rate": round(exact_rate, 4),
        "proxy_schema_coverage_rate": round(proxy_schema_rate, 4),
        "n_unmapped_gold_tools": unmapped,
        "unmapped_on_critical_path": int(unmapped_on_critical),
        "topology_in_train": int(topology_in_train),
        "combined_ood_score": round(ood, 4),
        "outcome": outcome,
        "c0_win": int(c0_win),
        "d1_win": int(d1_win),
        "delta_win": int(d1_win) - int(c0_win),
        "mapping_summary": "|".join(f"{m['tool']}:{m['confidence']}" for m in mappings),
        "note": "proxy mappings are hypotheses, not ground-truth semantic equivalence",
    }


def coverage_by_outcome(task_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = defaultdict(list)
    for r in task_rows:
        # bucket by exact coverage tertile
        er = float(r["exact_tool_coverage_rate"])
        if er >= 0.999:
            b = "exact_full"
        elif er >= 0.5:
            b = "exact_partial"
        elif er > 0:
            b = "exact_low"
        else:
            b = "exact_none"
        buckets[b].append(r)
        # also ood tertile-ish
        ood = float(r["combined_ood_score"])
        if ood < 0.25:
            buckets["ood_low"].append(r)
        elif ood < 0.5:
            buckets["ood_mid"].append(r)
        else:
            buckets["ood_high"].append(r)

    out = []
    for b, rows in sorted(buckets.items()):
        n = len(rows)
        gained = sum(1 for r in rows if r["outcome"] == "loss_to_win")
        lost = sum(1 for r in rows if r["outcome"] == "win_to_loss")
        out.append({
            "bucket": b,
            "n": n,
            "c0_wins": sum(r["c0_win"] for r in rows),
            "d1_wins": sum(r["d1_win"] for r in rows),
            "c0_win_rate": sum(r["c0_win"] for r in rows) / n if n else 0.0,
            "d1_win_rate": sum(r["d1_win"] for r in rows) / n if n else 0.0,
            "net_gain": gained - lost,
            "gained": gained,
            "lost": lost,
            "mean_ood": mean([float(r["combined_ood_score"]) for r in rows]),
        })
    return out
