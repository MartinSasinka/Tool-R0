"""Tool surface / schema / distractor audits (lexical proxies only)."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .graph_features import parse_reference
from .io import short_hash


def split_name_tokens(name: str) -> List[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    s = s.replace("-", "_").replace(".", "_")
    parts = [p for p in s.lower().split("_") if p]
    # strip trailing numeric suffixes from tokens
    out = []
    for p in parts:
        out.append(re.sub(r"\d+$", "", p) or p)
    return [t for t in out if t]


def normalize_tool_name(name: str) -> str:
    toks = split_name_tokens(str(name or ""))
    return "_".join(toks)


def tool_surface_record(tool: Dict[str, Any], *, role: str = "offered") -> Dict[str, Any]:
    name = str(tool.get("name") or "")
    params = tool.get("parameters") or {}
    if isinstance(params, list):
        # list-of-dicts form
        param_names = []
        param_types = []
        required = 0
        for p in params:
            if isinstance(p, dict):
                param_names.append(str(p.get("name") or p.get("key") or ""))
                param_types.append(str(p.get("type") or p.get("schema", {}).get("type") if isinstance(p.get("schema"), dict) else p.get("type") or ""))
                if p.get("required"):
                    required += 1
        n_params = len(param_names)
    elif isinstance(params, dict):
        # JSON-schema-like or name->spec
        if "properties" in params and isinstance(params["properties"], dict):
            props = params["properties"]
            param_names = sorted(str(k) for k in props.keys())
            param_types = [str((props[k] or {}).get("type") if isinstance(props[k], dict) else "") for k in param_names]
            req = params.get("required") or []
            required = len(req) if isinstance(req, list) else 0
            n_params = len(param_names)
        else:
            param_names = sorted(str(k) for k in params.keys())
            param_types = []
            for k in param_names:
                v = params[k]
                if isinstance(v, dict):
                    param_types.append(str(v.get("type") or ""))
                else:
                    param_types.append(type(v).__name__)
            required = sum(1 for k in param_names if isinstance(params.get(k), dict) and params[k].get("required"))
            n_params = len(param_names)
    else:
        param_names, param_types, required, n_params = [], [], 0, 0

    outs = tool.get("output_parameters") or tool.get("returns") or {}
    if isinstance(outs, dict):
        if "properties" in outs and isinstance(outs["properties"], dict):
            out_keys = sorted(str(k) for k in outs["properties"].keys())
            out_types = [str((outs["properties"][k] or {}).get("type") if isinstance(outs["properties"][k], dict) else "") for k in out_keys]
        else:
            out_keys = sorted(str(k) for k in outs.keys())
            out_types = [str((outs[k] or {}).get("type") if isinstance(outs[k], dict) else type(outs[k]).__name__) for k in out_keys]
    elif isinstance(outs, list):
        out_keys = [str(x.get("name") if isinstance(x, dict) else x) for x in outs]
        out_types = [str(x.get("type") if isinstance(x, dict) else "") for x in outs]
    else:
        out_keys, out_types = [], []

    desc = str(tool.get("description") or "")
    return {
        "name": name,
        "name_norm": normalize_tool_name(name),
        "name_tokens": split_name_tokens(name),
        "n_name_tokens": len(split_name_tokens(name)),
        "description": desc,
        "description_len": len(desc),
        "n_params": n_params,
        "n_required_params": required,
        "param_names": param_names,
        "param_types": param_types,
        "output_keys": out_keys,
        "output_types": out_types,
        "role": role,
    }


def _tfidf_cos(a: str, b: str) -> float:
    def toks(s: str) -> Counter:
        return Counter(re.findall(r"[a-z0-9_]+", s.lower()))

    ca, cb = toks(a), toks(b)
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    # binary-ish tf with idf over 2-doc collection
    def vec(c: Counter) -> List[float]:
        out = []
        for t in vocab:
            tf = c.get(t, 0)
            df = (1 if ca.get(t) else 0) + (1 if cb.get(t) else 0)
            idf = math.log(1 + 2 / (df or 1))
            out.append(tf * idf)
        return out

    va, vb = vec(ca), vec(cb)
    da = math.sqrt(sum(x * x for x in va))
    db = math.sqrt(sum(x * x for x in vb))
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (da * db)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def distractor_hardness(
    gold_tools: Sequence[Dict[str, Any]],
    offered_tools: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_names = {str(t.get("name") or "") for t in gold_tools}
    gold_recs = [tool_surface_record(t, role="gold") for t in gold_tools]
    distractors = [t for t in offered_tools if str(t.get("name") or "") not in gold_names]
    dist_recs = [tool_surface_record(t, role="distractor") for t in distractors]
    scores = []
    for d in dist_recs:
        best = 0.0
        best_g = None
        for g in gold_recs:
            lex = jaccard(d["name_tokens"], g["name_tokens"])
            p_overlap = jaccard(d["param_names"], g["param_names"])
            t_overlap = jaccard(d["param_types"], g["param_types"])
            desc = _tfidf_cos(d["description"], g["description"])
            # same rough family if sharing a contentful token
            family = bool(set(d["name_tokens"]) & set(g["name_tokens"]) - {"get", "set", "to", "of", "by"})
            score = 0.35 * lex + 0.25 * p_overlap + 0.15 * t_overlap + 0.25 * desc
            if score > best:
                best = score
                best_g = g["name"]
        scores.append({
            "distractor": d["name"],
            "nearest_gold": best_g,
            "hardness_proxy": round(best, 4),
            "note": "lexical/schema proxy — not semantic equivalence",
        })
    mean_h = sum(s["hardness_proxy"] for s in scores) / len(scores) if scores else 0.0
    return {
        "n_gold": len(gold_recs),
        "n_distractors": len(dist_recs),
        "mean_distractor_hardness_proxy": round(mean_h, 4),
        "max_distractor_hardness_proxy": max((s["hardness_proxy"] for s in scores), default=0.0),
        "pairs": scores,
    }


def collect_tool_universe(rows: Iterable[Dict[str, Any]], *, gold_key: str, tools_key: str = "tools") -> Tuple[List[Dict[str, Any]], Counter, Counter, Counter]:
    """Return surface rows + offered/gold/distractor-only frequency counters."""
    offered_c: Counter = Counter()
    gold_c: Counter = Counter()
    surfaces: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tools = row.get(tools_key) or []
        gold_calls = row.get(gold_key) or []
        gold_names = {str(c.get("name") or "") for c in gold_calls}
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name") or "")
            offered_c[name] += 1
            if name not in surfaces:
                surfaces[name] = tool_surface_record(t)
        for name in gold_names:
            gold_c[name] += 1
    distractor_only = Counter({k: v for k, v in offered_c.items() if gold_c[k] == 0})
    feats = []
    for name, surf in sorted(surfaces.items()):
        feats.append({
            **{k: (json_safe(v)) for k, v in surf.items()},
            "offered_freq": offered_c[name],
            "gold_freq": gold_c[name],
            "distractor_only_freq": distractor_only.get(name, 0),
        })
    return feats, offered_c, gold_c, distractor_only


def json_safe(v: Any) -> Any:
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    return v


def namespace_overlap(train_names: Set[str], diag_names: Set[str]) -> Dict[str, Any]:
    tr_norm = {normalize_tool_name(n): n for n in train_names}
    dg_norm = {normalize_tool_name(n): n for n in diag_names}
    exact = train_names & diag_names
    norm = set(tr_norm) & set(dg_norm)
    return {
        "n_train": len(train_names),
        "n_diagnostic": len(diag_names),
        "exact_overlap": len(exact),
        "exact_overlap_rate_vs_diag": len(exact) / max(1, len(diag_names)),
        "normalized_overlap": len(norm),
        "normalized_overlap_rate_vs_diag": len(norm) / max(1, len(dg_norm)),
        "exact_names": sorted(exact)[:50],
        "diag_only_exact": sorted(diag_names - train_names)[:50],
    }


def reference_syntax_rows(calls_by_id: Dict[str, List[Dict[str, Any]]], source: str) -> List[Dict[str, Any]]:
    rows = []
    for sid, calls in calls_by_id.items():
        keys = Counter()
        patterns = Counter()
        for c in calls:
            args = c.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            for v in args.values():
                ref = parse_reference(v)
                if ref:
                    patterns[ref["pattern_id"]] += 1
                    if ref["output_key"]:
                        keys[ref["output_key"]] += 1
        rows.append({
            "sample_id": sid,
            "source": source,
            "n_refs": sum(patterns.values()),
            "pattern_hist": dict(patterns),
            "output_key_hist": dict(keys),
            "dominant_output_key": keys.most_common(1)[0][0] if keys else "",
        })
    return rows


def track_of_row(row: Dict[str, Any]) -> str:
    prov = row.get("provenance") or {}
    if isinstance(prov, dict) and prov.get("track"):
        return str(prov["track"])
    # heuristic: A-track names often snake descriptive; G often short math
    return "unknown"
