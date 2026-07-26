"""Generic TargetProfile extraction from canonical target rows.

Canonical row (produced by a target adapter):
  {query, calls: [{name, arguments, label}], tools: [{name, description,
   param_types: {pname: type}, output_fields: [...]}], gold_answer}

Only aggregate statistics are stored (hygiene rule D04) — never raw
queries or gold programs.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

from ..schemas import TargetProfile
from ..util import arg_type_of, is_reference, sha256_obj


def _bucket_call_count(n: int, buckets: List[str]) -> str:
    for b in buckets:
        if b.endswith("+") and n >= int(b[:-1]):
            return b
        if not b.endswith("+") and n == int(b):
            return b
    return buckets[-1]


def _ref_graph(calls: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """edge: consumer index -> producer indices (via $var references)."""
    label_to_idx = {}
    for i, c in enumerate(calls):
        label = str(c.get("label", f"$var{i + 1}"))
        label_to_idx[label.strip("$").replace("_", "")] = i
    edges: Dict[int, List[int]] = defaultdict(list)

    def _scan(v: Any, i: int) -> None:
        if is_reference(v):
            key = v.strip().strip("$").split(".")[0].replace("_", "")
            if key in label_to_idx:
                edges[i].append(label_to_idx[key])
        elif isinstance(v, list):
            for item in v:
                _scan(item, i)
        elif isinstance(v, dict):
            for item in v.values():
                _scan(item, i)

    for i, c in enumerate(calls):
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            for v in args.values():
                _scan(v, i)
    return edges


def classify_motif(calls: List[Dict[str, Any]]) -> str:
    n = len(calls)
    edges = _ref_graph(calls)
    indeg = {i: len(set(edges.get(i, []))) for i in range(n)}
    consumed = Counter(p for ps in edges.values() for p in set(ps))
    if not edges:
        return "independent"
    if any(v > 2 for v in indeg.values()):
        return "branch_aggregate"
    chain_like = all(v <= 1 for v in indeg.values())
    if chain_like and all(consumed.get(i, 0) <= 1 for i in range(n)):
        return "linear" if len(edges) == n - 1 else "mixed"
    if any(v == 2 for v in indeg.values()):
        return "fan_in"
    return "mixed"


def dependency_depth(calls: List[Dict[str, Any]]) -> int:
    edges = _ref_graph(calls)
    depth: Dict[int, int] = {}

    def _d(i: int) -> int:
        if i in depth:
            return depth[i]
        ps = set(edges.get(i, []))
        depth[i] = 1 + (max((_d(p) for p in ps), default=0))
        return depth[i]

    return max((_d(i) for i in range(len(calls))), default=0)


def _dist(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {str(k): round(v / total, 6) for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    vs = sorted(vals)
    return {
        "mean": round(statistics.fmean(vals), 4),
        "min": vs[0], "max": vs[-1],
        "p25": vs[len(vs) // 4], "p50": vs[len(vs) // 2], "p75": vs[(3 * len(vs)) // 4],
    }


def featurize_row(row: Dict[str, Any], buckets: List[str]) -> Dict[str, Any]:
    """Shared structural featurizer used by profiling AND selection metrics."""
    calls = row["calls"]
    args = []
    for c in calls:
        a = c.get("arguments") or {}
        if isinstance(a, dict):
            args.extend(a.values())
    at = Counter(arg_type_of(v) for v in args)
    n_args = sum(at.values()) or 1
    return {
        "call_bucket": _bucket_call_count(len(calls), buckets),
        "call_count": len(calls),
        "motif": classify_motif(calls),
        "depth": dependency_depth(calls),
        "ref_share": at.get("reference", 0) / n_args,
        "numeric_string_share": at.get("numeric_string", 0) / n_args,
        "arg_types": at,
        "n_tools": len(row.get("tools") or []),
        "q_len": len(str(row.get("query", ""))),
        "answer_type": arg_type_of(row.get("gold_answer")),
    }


def extract_profile(rows: List[Dict[str, Any]], *, target: str, source: str,
                    buckets: List[str], failure_profile: Dict[str, Any],
                    profile_version: str) -> TargetProfile:
    feats = [featurize_row(r, buckets) for r in rows]
    call_c = Counter(f["call_bucket"] for f in feats)
    motif_c = Counter(f["motif"] for f in feats)
    depth_c = Counter(str(min(f["depth"], 6)) for f in feats)
    arg_c: Counter = Counter()
    for f in feats:
        arg_c.update(f["arg_types"])
    ans_c = Counter(f["answer_type"] for f in feats)
    ref_task = sum(1 for f in feats if f["ref_share"] > 0) / max(len(feats), 1)
    n_args = sum(arg_c.values()) or 1

    out_fields: Counter = Counter()
    name_tokens: Counter = Counter()
    desc_lens: List[float] = []
    single_word = 0
    n_names = 0
    relevant_ratios: List[float] = []
    for r in rows:
        gold_names = {c["name"] for c in r["calls"]}
        tools = r.get("tools") or []
        if tools:
            relevant_ratios.append(len(gold_names & {t["name"] for t in tools}) / len(tools))
        for t in tools:
            n_names += 1
            toks = str(t["name"]).split("_")
            name_tokens[len(toks)] += 1
            single_word += (len(toks) == 1)
            desc_lens.append(len(str(t.get("description", ""))))
            for f in (t.get("output_fields") or []):
                out_fields[f] += 1

    prof = TargetProfile(
        target=target, source=source, n_rows=len(rows),
        profile_version=profile_version,
        call_count_dist=_dist(call_c),
        motif_dist=_dist(motif_c),
        dependency_depth_dist=_dist(depth_c),
        reference_task_rate=round(ref_task, 4),
        reference_arg_share=round(arg_c.get("reference", 0) / n_args, 4),
        direct_arg_share=round(1 - arg_c.get("reference", 0) / n_args, 4),
        arg_type_dist=_dist(arg_c),
        numeric_string_rate=round(arg_c.get("numeric_string", 0) / n_args, 4),
        answer_type_dist=_dist(ans_c),
        output_field_names=_dist(out_fields),
        tools_per_task={**_stats([f["n_tools"] for f in feats]),
                        "hist": _dist(Counter(f["n_tools"] for f in feats))},
        relevant_ratio_mean=round(statistics.fmean(relevant_ratios), 4) if relevant_ratios else 0.0,
        tool_name_morphology={
            "tokens_per_name": _dist(name_tokens),
            "single_word_share": round(single_word / max(n_names, 1), 4),
        },
        tool_description_length=_stats(desc_lens),
        signature_similarity_mean=0.0,
        question_length=_stats([f["q_len"] for f in feats]),
        student_failure_profile=failure_profile,
    )
    prof.profile_hash = sha256_obj(prof.model_dump(exclude={"profile_hash"}))
    return prof


def profile_report_md(prof: TargetProfile) -> str:
    def _tbl(d: Dict[str, float], k1: str) -> str:
        lines = [f"| {k1} | share |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in d.items()]
        return "\n".join(lines)

    return f"""# TARGET PROFILE — {prof.target}

- source: `{prof.source}` (n={prof.n_rows})
- profile_version: `{prof.profile_version}`, hash: `{prof.profile_hash[:16]}`
- aggregate statistics only (no raw queries/programs stored)

## Call-count distribution
{_tbl(prof.call_count_dist, 'bucket')}

## Motif distribution
{_tbl(prof.motif_dist, 'motif')}

## Dependency depth
{_tbl(prof.dependency_depth_dist, 'depth')}

## Arguments
- reference task rate: **{prof.reference_task_rate:.3f}**
- reference arg share: **{prof.reference_arg_share:.3f}** (direct {prof.direct_arg_share:.3f})
- numeric-string arg rate: {prof.numeric_string_rate:.4f}

{_tbl(prof.arg_type_dist, 'arg type')}

## Answer types
{_tbl(prof.answer_type_dist, 'type')}

## Offered tools per task
`{prof.tools_per_task}`
- relevant/offered ratio mean: {prof.relevant_ratio_mean:.3f}

## Tool morphology
- name tokens: `{prof.tool_name_morphology['tokens_per_name']}`
- single-word name share: {prof.tool_name_morphology['single_word_share']:.3f}
- description length: `{prof.tool_description_length}`
- output field names: `{prof.output_field_names}`

## Question length
`{prof.question_length}`

## Student failure profile
`{prof.student_failure_profile}`
"""
