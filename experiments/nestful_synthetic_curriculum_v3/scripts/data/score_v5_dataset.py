#!/usr/bin/env python3
"""Score quality + diversity of a curriculum v5 dataset (read-only).

Reports (per corpus): unique tools, tool-family distribution, tool frequency
with max share, argument arity + type distribution, output-type distribution,
offered-tool-count distribution, reference/dependency motifs, answer types,
dedup, replay pass rate, registry-hash consistency.

Exit code 1 when a configurable threshold is violated:
  --max-tool-share      (default 0.10)   no tool may exceed this call share
  --min-unique-tools    (default 60)     distinct tools used in gold calls
  --min-replay-pass     (default 1.0)    fraction of rows passing replay

Usage:
  python score_v5_dataset.py data/curriculum_v5_registry/filtered/*.jsonl \
      --out reports/DATASET_QUALITY_V5.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _V3)

from lib.synthetic_gen_v5 import replay_row, question_hash, trace_hash  # noqa: E402
from lib.synthetic_tools import REGISTRY_VERSION, TOOLS, registry_hash  # noqa: E402

_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)(?:\.(\w+))?\$$")


def _load_rows(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="JSONL files or globs")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--max-tool-share", type=float, default=0.10)
    ap.add_argument("--min-unique-tools", type=int, default=60)
    ap.add_argument("--min-replay-pass", type=float, default=1.0)
    args = ap.parse_args()

    rows = _load_rows(args.inputs)
    if not rows:
        print("[score_v5] no rows loaded", file=sys.stderr)
        return 2
    print(f"[score_v5] loaded {len(rows)} rows")

    tool_freq: Counter = Counter()
    fam_freq: Counter = Counter()
    dom_freq: Counter = Counter()
    arity_freq: Counter = Counter()
    ptype_freq: Counter = Counter()
    otype_freq: Counter = Counter()
    offered_freq: Counter = Counter()
    ncalls_freq: Counter = Counter()
    answer_types: Counter = Counter()
    motifs: Counter = Counter()
    ref_motifs: Counter = Counter()
    reg_hashes: Counter = Counter()
    q_hashes, t_hashes = set(), set()
    dup_q = dup_t = 0
    replay_fail = 0

    for r in rows:
        ncalls_freq[r.get("num_calls")] += 1
        answer_types[r.get("answer_type")] += 1
        motifs[r.get("motif_type")] += 1
        offered_freq[len(r.get("tools") or [])] += 1
        reg_hashes[r.get("registry_hash") or "<missing>"] += 1

        qh, th = question_hash(r["question"]), trace_hash(r["gold_calls"])
        dup_q += qh in q_hashes
        dup_t += th in t_hashes
        q_hashes.add(qh)
        t_hashes.add(th)

        ok, _obs = replay_row(r)
        replay_fail += not ok

        labels_seen = set()
        for i, call in enumerate(r["gold_calls"]):
            name = call["name"]
            tool_freq[name] += 1
            spec = TOOLS.get(name)
            if spec:
                fam_freq[spec["family"]] += 1
                dom_freq[spec["domain"]] += 1
                arity_freq[len(spec["params"])] += 1
                otype_freq[spec["out_type"]] += 1
                for meta in spec["params"].values():
                    ptype_freq[meta["type"]] += 1
            else:
                fam_freq["<unregistered>"] += 1
            n_refs = 0
            for v in (call.get("arguments") or {}).values():
                m = _REF_RE.match(v.strip()) if isinstance(v, str) else None
                if m:
                    n_refs += 1
                    var = m.group(1)
                    if m.group(2):
                        ref_motifs["fielded_reference"] += 1
                    if var in labels_seen and var != f"var{i}":
                        ref_motifs["non_adjacent_reuse"] += 1
            if i > 0 and n_refs == 0:
                ref_motifs["literal_only_followup"] += 1
            elif n_refs > 1:
                ref_motifs["multi_reference_call"] += 1
            labels_seen.add((call.get("label") or "").lstrip("$"))

    total_calls = sum(tool_freq.values())
    top = tool_freq.most_common(10)
    max_share = top[0][1] / total_calls if total_calls else 0.0
    replay_pass = 1.0 - replay_fail / len(rows)

    violations = []
    if max_share > args.max_tool_share:
        violations.append(f"max_tool_share {max_share:.3f} > {args.max_tool_share}"
                          f" ({top[0][0]})")
    if len(tool_freq) < args.min_unique_tools:
        violations.append(f"unique_tools {len(tool_freq)} < {args.min_unique_tools}")
    if replay_pass < args.min_replay_pass:
        violations.append(f"replay_pass {replay_pass:.4f} < {args.min_replay_pass}")
    if dup_q or dup_t:
        violations.append(f"duplicates: {dup_q} question, {dup_t} trace")
    foreign = {h: c for h, c in reg_hashes.items() if h != registry_hash()}
    if foreign:
        violations.append(f"registry hash mismatch: {foreign} "
                          f"(current {registry_hash()[:16]}…)")

    report = {
        "rows": len(rows),
        "registry_version": REGISTRY_VERSION,
        "registry_hash_current": registry_hash(),
        "registry_hashes_in_data": dict(reg_hashes),
        "unique_tools_used": len(tool_freq),
        "registry_size": len(TOOLS),
        "total_gold_calls": total_calls,
        "max_tool_share": round(max_share, 4),
        "top10_tools": [{"tool": n, "count": c,
                         "share": round(c / total_calls, 4)} for n, c in top],
        "tool_frequency": dict(tool_freq.most_common()),
        "family_distribution": dict(fam_freq.most_common()),
        "domain_distribution": dict(dom_freq.most_common()),
        "arity_distribution": {str(k): v for k, v in sorted(arity_freq.items())},
        "param_type_distribution": dict(ptype_freq.most_common()),
        "output_type_distribution": dict(otype_freq.most_common()),
        "offered_tool_count_distribution":
            {str(k): v for k, v in sorted(offered_freq.items())},
        "num_calls_distribution": {str(k): v for k, v in sorted(ncalls_freq.items())},
        "answer_type_distribution": dict(answer_types.most_common()),
        "motif_distribution": dict(motifs.most_common()),
        "reference_motifs": dict(ref_motifs.most_common()),
        "duplicate_questions": dup_q,
        "duplicate_traces": dup_t,
        "replay_pass_rate": round(replay_pass, 4),
        "thresholds": {"max_tool_share": args.max_tool_share,
                       "min_unique_tools": args.min_unique_tools,
                       "min_replay_pass": args.min_replay_pass},
        "violations": violations,
        "ok": not violations,
    }

    print(json.dumps({k: report[k] for k in (
        "rows", "unique_tools_used", "max_tool_share", "replay_pass_rate",
        "num_calls_distribution", "output_type_distribution",
        "reference_motifs", "violations", "ok")}, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"[score_v5] report -> {args.out}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
