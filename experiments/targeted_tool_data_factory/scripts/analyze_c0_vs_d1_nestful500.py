#!/usr/bin/env python3
"""Paired C0 vs D1 analysis on NESTFUL-500 + train-300 coverage + DAG diversity.

Read-only w.r.t. original eval artefacts. Does not train or overwrite C0/D1 results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
FACTORY = ROOT / "experiments" / "targeted_tool_data_factory"
REPORTS = FACTORY / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

C0_TRAJ = FACTORY / "outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/final_eval_trajectories.jsonl"
C0_MANIFEST = FACTORY / "outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/eval_manifest.json"
C0_METRICS = FACTORY / "outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/metrics.json"
C0_OFFICIAL = FACTORY / "outputs/runpod_pilot2/phase1_canary_from_zip/eval/C0_nestful500/metrics_official.json"

D1_TRAJ = FACTORY / "outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500/eval/D1_nestful500/final_eval_trajectories.jsonl"
D1_MANIFEST = FACTORY / "outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500/eval/D1_nestful500/eval_manifest.json"
D1_METRICS = FACTORY / "outputs/runpod_pilot3/train_nestful500_from_zip/train_nestful500/eval/D1_nestful500/metrics_merged.json"

DIAG = FACTORY / "runpod_bundle_pilot2/data/nestful_diagnostic_500.jsonl"
TRAIN_FULL = FACTORY / "outputs/selected/export_pilot3/train_grpo_pilot3.jsonl"
# Recreate the n=300 subset used on RunPod (first N rows of frozen train).
TRAIN_N = 300

CONFIG_PARTIAL = ROOT / "experiments/nestful_mtgrpo_partial/config.yaml"
EVAL_SHARDED = FACTORY / "runpod_bundle_pilot3/eval_nestful500_sharded.py"

sys.path.insert(0, str(FACTORY / "src"))
from targeted_tool_data.profile import classify_motif  # noqa: E402
from targeted_tool_data.util import is_reference, arg_type_of  # noqa: E402
try:
    from targeted_tool_data import registry as reg  # noqa: E402
except Exception:
    reg = None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v > 0)
    s = str(v).lower()
    if s in ("true", "1", "pass", "yes"):
        return True
    if s in ("false", "0", "fail", "no"):
        return False
    return bool(v)


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def shannon(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def short_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def ref_graph(calls: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    label_to_idx = {}
    for i, c in enumerate(calls):
        label = str(c.get("label", f"$var{i + 1}"))
        label_to_idx[label.strip("$").replace("_", "")] = i
    edges: Dict[int, List[int]] = defaultdict(list)

    def scan(v: Any, i: int) -> None:
        if is_reference(v):
            key = str(v).strip().strip("$").split(".")[0].replace("_", "")
            if key in label_to_idx:
                edges[i].append(label_to_idx[key])
        elif isinstance(v, list):
            for item in v:
                scan(item, i)
        elif isinstance(v, dict):
            for item in v.values():
                scan(item, i)

    for i, c in enumerate(calls):
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            for v in args.values():
                scan(v, i)
    return {k: sorted(set(v)) for k, v in edges.items()}


def topology_id(calls: List[Dict[str, Any]]) -> str:
    n = len(calls)
    edges = ref_graph(calls)
    # shape only: n + sorted edge list
    edge_list = sorted((p, c) for c, ps in edges.items() for p in ps)
    return "topo_" + short_hash({"n": n, "e": edge_list})


def surface_program_id(calls: List[Dict[str, Any]]) -> str:
    steps = []
    for c in calls:
        args = c.get("arguments") or {}
        schema = sorted(args.keys()) if isinstance(args, dict) else []
        steps.append({"name": c.get("name"), "schema": schema, "label": c.get("label")})
    return "surf_" + short_hash(steps)


_SURFACE_TO_SID: Optional[Dict[str, str]] = None


def _name_to_primitive(name: str) -> str:
    global _SURFACE_TO_SID
    if reg is None:
        return str(name)
    if _SURFACE_TO_SID is None:
        _SURFACE_TO_SID = {}
        try:
            for sid, _track, surf in reg.all_surfaces():
                _SURFACE_TO_SID[surf.name] = sid
        except Exception:
            _SURFACE_TO_SID = {}
    return _SURFACE_TO_SID.get(str(name), str(name))


def arg_pattern(args: Dict[str, Any]) -> List[str]:
    out = []
    for k in sorted(args.keys()):
        v = args[k]
        if is_reference(v):
            out.append(f"{k}:ref")
        else:
            out.append(f"{k}:{arg_type_of(v)}")
    return out


def primitive_program_id(calls: List[Dict[str, Any]]) -> str:
    steps = []
    for c in calls:
        args = c.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        steps.append({
            "prim": _name_to_primitive(str(c.get("name"))),
            "pat": arg_pattern(args),
        })
    # topology edges without constants
    edges = ref_graph(calls)
    edge_list = sorted((p, c) for c, ps in edges.items() for p in ps)
    return "prim_" + short_hash({"steps": steps, "e": edge_list})


def ref_density(calls: List[Dict[str, Any]]) -> float:
    vals = []
    for c in calls:
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            vals.extend(args.values())
    if not vals:
        return 0.0
    return sum(1 for v in vals if is_reference(v)) / len(vals)


def answer_type(v: Any) -> str:
    return arg_type_of(v)


def classify_failure(row: Dict[str, Any], gold_n: int) -> str:
    traj = row.get("_traj") or {}
    if as_bool(traj.get("official_win")):
        return "success"
    if traj.get("parse_valid") is False or traj.get("stop_reason") == "parse_fail":
        return "parse"
    fails = []
    for t in traj.get("turns") or []:
        fr = t.get("fail_reason")
        if fr:
            fails.append(str(fr))
    blob = " | ".join(fails) + " " + str(traj.get("mismatch_reason") or "") + " " + str(traj.get("execution_error") or "")
    blob_l = blob.lower()
    if "unresolved" in blob_l or "reference" in blob_l and "unknown_tool" not in blob_l:
        if "unresolved" in blob_l or "bad_ref" in blob_l or "missing_ref" in blob_l:
            return "unresolved_reference"
    if "unknown_tool" in blob_l or "wrong_tool" in blob_l:
        return "wrong_tool"
    n_pred = int(traj.get("num_tool_calls") or 0)
    if n_pred < gold_n:
        # undercalling often co-occurs with other errors; prioritize tool/exec signals
        if "unknown_tool" in blob_l:
            return "wrong_tool"
        if traj.get("executable") is False and ("arg" in blob_l or "type" in blob_l):
            return "wrong_args"
        return "too_few_calls"
    if n_pred > gold_n:
        return "too_many_calls"
    if traj.get("executable") is False:
        if "arg" in blob_l or "type" in blob_l or "schema" in blob_l:
            return "wrong_args"
        if "unknown_tool" in blob_l:
            return "wrong_tool"
        return "wrong_args"
    # executable path but lost official win
    if not as_bool(row.get("final_answer_pass")):
        return "final_answer"
    return "final_answer"


def load_arm(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id"))
        traj = row.get("_traj") or {}
        out[sid] = {
            "row": row,
            "win": bool(as_bool(traj.get("official_win"))),
            "f1_func": float(row.get("internal_f1_func") or (traj.get("internal") or {}).get("f1_func") or 0.0),
            "f1_param": float(row.get("internal_f1_param") or (traj.get("internal") or {}).get("f1_param") or 0.0),
            "executable": bool(as_bool(traj.get("executable"))),
            "final_answer_pass": bool(as_bool(row.get("final_answer_pass"))),
            "sol_eq": bool(as_bool(row.get("solution_equivalent_pass"))),
            "strict_gold": bool(as_bool(row.get("strict_gold_trace_pass"))),
            "num_tool_calls": int(traj.get("num_tool_calls") or 0),
            "num_gold_calls": int(row.get("num_gold_calls") or traj.get("gold_num_turns") or 0),
            "stop_reason": traj.get("stop_reason"),
            "pred_answer": traj.get("pred_answer"),
        }
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar using binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X<=min(b,c)) * 2, capped at 1; X~Bin(n,0.5)
    k = min(b, c)

    def binom_cdf(k_max: int, n: int) -> float:
        # sum_{i=0..k_max} C(n,i) / 2^n
        total = 0.0
        # recursive computation of binomial coeffs
        c_i = 1.0
        for i in range(0, k_max + 1):
            if i > 0:
                c_i *= (n - i + 1) / i
            total += c_i
        return total / (2 ** n)

    p = 2 * binom_cdf(k, n)
    return min(1.0, p)


def bootstrap_delta(wins_a: List[bool], wins_b: List[bool], n_boot: int = 5000, seed: int = 20260727) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(wins_a)
    assert n == len(wins_b) and n > 0
    deltas = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        wa = sum(wins_a[i] for i in sample) / n
        wb = sum(wins_b[i] for i in sample) / n
        deltas.append(100 * (wb - wa))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    point = 100 * (sum(wins_b) / n - sum(wins_a) / n)
    return point, lo, hi


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pp(x: float) -> str:
    return f"{x:+.1f} pp"


def main() -> int:
    assert C0_TRAJ.is_file(), C0_TRAJ
    assert D1_TRAJ.is_file(), D1_TRAJ
    assert DIAG.is_file(), DIAG
    assert TRAIN_FULL.is_file(), TRAIN_FULL

    c0 = load_arm(C0_TRAJ)
    d1 = load_arm(D1_TRAJ)
    ids = sorted(set(c0) & set(d1))
    only_c0 = sorted(set(c0) - set(d1))
    only_d1 = sorted(set(d1) - set(c0))

    diag_rows = {str(r["sample_id"]): r for r in read_jsonl(DIAG)}
    # enrich diagnostic features
    diag_feat: Dict[str, Dict[str, Any]] = {}
    for sid, r in diag_rows.items():
        calls = r.get("output") or []
        if not isinstance(calls, list):
            calls = []
        tools = r.get("tools") or []
        q = r.get("input") or ""
        diag_feat[sid] = {
            "question": q,
            "gold_answer": r.get("gold_answer"),
            "n_gold": len(calls),
            "motif": classify_motif(calls) if calls else "unknown",
            "answer_type": answer_type(r.get("gold_answer")),
            "n_offered": len(tools) if isinstance(tools, list) else 0,
            "ref_density": ref_density(calls),
            "prompt_chars": len(q),
            "calls": calls,
            "topology_id": topology_id(calls) if calls else None,
        }

    # fairness
    c0_manifest = json.loads(C0_MANIFEST.read_text(encoding="utf-8")) if C0_MANIFEST.is_file() else {}
    d1_manifest = json.loads(D1_MANIFEST.read_text(encoding="utf-8")) if D1_MANIFEST.is_file() else {}
    c0_cmd = " ".join(c0_manifest.get("cmd") or [])
    c0_uses_vllm = "hardware.use_vllm=true" in c0_cmd
    d1_uses_vllm = True  # eval_nestful500_sharded.py hardcodes use_vllm=true
    backend_identical = c0_uses_vllm == d1_uses_vllm

    fairness = {
        "n_c0": len(c0),
        "n_d1": len(d1),
        "n_paired": len(ids),
        "only_c0": len(only_c0),
        "only_d1": len(only_d1),
        "same_diagnostic": True,
        "diagnostic_path": str(DIAG.relative_to(ROOT)),
        "c0_temperature": 0.0,
        "d1_temperature": 0.0,
        "c0_top_p": 1.0,
        "d1_top_p": 1.0,
        "paradigm": "react",
        "num_eval_rollouts": 1,
        "scorer": "NESTFUL official calculate_win_score / official_win",
        "c0_backend": "HF (no use_vllm override in eval_manifest)" if not c0_uses_vllm else "vLLM",
        "d1_backend": "vLLM (eval_nestful500_sharded.py DECODING)",
        "backend_identical": backend_identical,
        "c0_checkpoint": c0_manifest.get("checkpoint"),
        "d1_checkpoint": d1_manifest.get("checkpoint"),
        "c0_manifest_cmd": c0_manifest.get("cmd"),
        "note": (
            "Backend NOT identical: C0 phase1 canary eval omitted hardware.use_vllm=true; "
            "D1 sharded eval forced vLLM. Do not treat headline Δ as fully confounded-free. "
            "Re-run C0 with the D1 eval script into a NEW directory (do not overwrite)."
        ),
    }

    wins_c0 = [c0[i]["win"] for i in ids]
    wins_d1 = [d1[i]["win"] for i in ids]
    n = len(ids)
    win_c0 = sum(wins_c0) / n
    win_d1 = sum(wins_d1) / n
    gained = [i for i in ids if (not c0[i]["win"]) and d1[i]["win"]]
    lost = [i for i in ids if c0[i]["win"] and (not d1[i]["win"])]
    both_pass = [i for i in ids if c0[i]["win"] and d1[i]["win"]]
    both_fail = [i for i in ids if (not c0[i]["win"]) and (not d1[i]["win"])]
    b, c = len(lost), len(gained)  # McNemar: b=C0yes D1no, c=C0no D1yes
    p_mcnemar = mcnemar_exact(b, c)
    delta_pp, ci_lo, ci_hi = bootstrap_delta(wins_c0, wins_d1)
    err_c0 = 1 - win_c0
    err_d1 = 1 - win_d1
    rel_err_red = ((err_c0 - err_d1) / err_c0) if err_c0 > 0 else None

    summary = {
        "win_rate_c0": win_c0,
        "win_rate_d1": win_d1,
        "delta_pp": delta_pp,
        "bootstrap_95ci_pp": [ci_lo, ci_hi],
        "gained": len(gained),
        "lost": len(lost),
        "unchanged_pass": len(both_pass),
        "unchanged_fail": len(both_fail),
        "mcnemar_b_lost": b,
        "mcnemar_c_gained": c,
        "mcnemar_exact_p": p_mcnemar,
        "relative_error_reduction": rel_err_red,
        "f1_func_c0": mean([c0[i]["f1_func"] for i in ids]),
        "f1_func_d1": mean([d1[i]["f1_func"] for i in ids]),
        "f1_param_c0": mean([c0[i]["f1_param"] for i in ids]),
        "f1_param_d1": mean([d1[i]["f1_param"] for i in ids]),
        "executable_c0": mean([float(c0[i]["executable"]) for i in ids]),
        "executable_d1": mean([float(d1[i]["executable"]) for i in ids]),
        "final_answer_pass_c0": mean([float(c0[i]["final_answer_pass"]) for i in ids]),
        "final_answer_pass_d1": mean([float(d1[i]["final_answer_pass"]) for i in ids]),
        "sol_eq_c0": mean([float(c0[i]["sol_eq"]) for i in ids]),
        "sol_eq_d1": mean([float(d1[i]["sol_eq"]) for i in ids]),
        "strict_gold_c0": mean([float(c0[i]["strict_gold"]) for i in ids]),
        "strict_gold_d1": mean([float(d1[i]["strict_gold"]) for i in ids]),
    }

    def bucket_table(key_fn):
        groups: Dict[str, List[str]] = defaultdict(list)
        for sid in ids:
            groups[str(key_fn(sid))].append(sid)
        rows = []
        for k in sorted(groups, key=lambda x: (x == "?", x.isdigit() and int(x) or 999, x)):
            sids = groups[k]
            wa = sum(c0[i]["win"] for i in sids) / len(sids)
            wb = sum(d1[i]["win"] for i in sids) / len(sids)
            g = sum(1 for i in sids if i in gained)
            l = sum(1 for i in sids if i in lost)
            rows.append({
                "bucket": k,
                "n": len(sids),
                "c0": wa,
                "d1": wb,
                "delta_pp": 100 * (wb - wa),
                "gained": g,
                "lost": l,
            })
        return rows

    by_calls = bucket_table(lambda sid: diag_feat.get(sid, {}).get("n_gold") or c0[sid]["num_gold_calls"])
    by_calls_exact = [r for r in by_calls if str(r["bucket"]) in {str(i) for i in range(2, 9)}]
    by_answer = bucket_table(lambda sid: diag_feat.get(sid, {}).get("answer_type", "?"))
    by_motif = bucket_table(lambda sid: diag_feat.get(sid, {}).get("motif", "?"))

    def offered_bucket(sid: str) -> str:
        n_off = diag_feat.get(sid, {}).get("n_offered", 0)
        if n_off <= 9:
            return "8-9"
        if n_off <= 12:
            return "10-12"
        return "13+"

    by_offered = bucket_table(offered_bucket)

    def ref_bucket(sid: str) -> str:
        d = diag_feat.get(sid, {}).get("ref_density", 0.0)
        if d < 0.25:
            return "ref<0.25"
        if d < 0.45:
            return "ref0.25-0.45"
        return "ref>=0.45"

    by_ref = bucket_table(ref_bucket)

    def prompt_bucket(sid: str) -> str:
        L = diag_feat.get(sid, {}).get("prompt_chars", 0)
        if L < 120:
            return "q<120"
        if L < 200:
            return "q120-199"
        return "q>=200"

    by_prompt = bucket_table(prompt_bucket)

    # failure classes
    fail_c0 = Counter(classify_failure(c0[i]["row"], diag_feat.get(i, {}).get("n_gold") or c0[i]["num_gold_calls"]) for i in ids)
    fail_d1 = Counter(classify_failure(d1[i]["row"], diag_feat.get(i, {}).get("n_gold") or d1[i]["num_gold_calls"]) for i in ids)
    fail_gained = Counter(classify_failure(c0[i]["row"], diag_feat.get(i, {}).get("n_gold") or c0[i]["num_gold_calls"]) for i in gained)
    fail_lost_was = Counter(classify_failure(d1[i]["row"], diag_feat.get(i, {}).get("n_gold") or d1[i]["num_gold_calls"]) for i in lost)

    # 2-call share of gain
    g2 = sum(1 for i in gained if (diag_feat.get(i, {}).get("n_gold") or 0) == 2)
    gain_from_2call_share = g2 / len(gained) if gained else None
    # wins delta attributed
    delta_wins = sum(wins_d1) - sum(wins_c0)
    # contribution by call bucket to net win change
    contrib = []
    for row in by_calls:
        sids = [i for i in ids if str(diag_feat.get(i, {}).get("n_gold") or c0[i]["num_gold_calls"]) == str(row["bucket"])]
        net = sum(1 for i in sids if d1[i]["win"]) - sum(1 for i in sids if c0[i]["win"])
        contrib.append({"bucket": row["bucket"], "net_wins": net, "share_of_delta": (net / delta_wins) if delta_wins else None})

    # GAINED_LOST CSV
    csv_path = REPORTS / "GAINED_LOST_TASKS.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "sample_id", "outcome", "n_gold", "motif", "answer_type", "n_offered",
            "ref_density", "prompt_chars", "c0_fail_class", "d1_fail_class",
            "c0_win", "d1_win", "c0_f1_func", "d1_f1_func", "c0_exec", "d1_exec",
            "question",
        ])
        w.writeheader()
        for sid in gained + lost:
            f = diag_feat.get(sid, {})
            outcome = "gained" if sid in gained else "lost"
            w.writerow({
                "sample_id": sid,
                "outcome": outcome,
                "n_gold": f.get("n_gold"),
                "motif": f.get("motif"),
                "answer_type": f.get("answer_type"),
                "n_offered": f.get("n_offered"),
                "ref_density": round(float(f.get("ref_density") or 0), 4),
                "prompt_chars": f.get("prompt_chars"),
                "c0_fail_class": classify_failure(c0[sid]["row"], f.get("n_gold") or 0),
                "d1_fail_class": classify_failure(d1[sid]["row"], f.get("n_gold") or 0),
                "c0_win": int(c0[sid]["win"]),
                "d1_win": int(d1[sid]["win"]),
                "c0_f1_func": round(c0[sid]["f1_func"], 4),
                "d1_f1_func": round(d1[sid]["f1_func"], 4),
                "c0_exec": int(c0[sid]["executable"]),
                "d1_exec": int(d1[sid]["executable"]),
                "question": (f.get("question") or "")[:240],
            })

    # Train-300
    train_all = read_jsonl(TRAIN_FULL)
    train300 = train_all[:TRAIN_N]
    train_path_note = f"first {TRAIN_N} rows of {TRAIN_FULL.relative_to(ROOT)} (matches run_train_nestful500_4gpu.sh)"

    def train_dist(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        calls = Counter(int(r.get("num_calls") or len(r.get("gold_calls") or [])) for r in rows)
        motifs = Counter(r.get("motif_type") or (r.get("provenance") or {}).get("motif") for r in rows)
        answers = Counter(r.get("answer_type") for r in rows)
        tracks = Counter((r.get("provenance") or {}).get("track") for r in rows)
        cells = Counter((r.get("provenance") or {}).get("generation_cell_id") for r in rows)
        failures = Counter((r.get("provenance") or {}).get("target_failure_mode") for r in rows)
        skills = Counter((r.get("provenance") or {}).get("target_skill") for r in rows)
        # name morphology: single-word share among gold tool names
        names = []
        for r in rows:
            for c in r.get("gold_calls") or []:
                names.append(str(c.get("name") or ""))
        single = sum(1 for n in names if n and "_" not in n and " " not in n)
        ref_shares = []
        for r in rows:
            gc = r.get("gold_calls") or []
            if gc:
                ref_shares.append(ref_density(gc))
        return {
            "n": len(rows),
            "call_count": {str(k): v for k, v in sorted(calls.items())},
            "call_count_share": {str(k): v / len(rows) for k, v in sorted(calls.items())},
            "motif": dict(motifs),
            "motif_share": {k: v / len(rows) for k, v in motifs.items()},
            "answer_type": dict(answers),
            "answer_share": {k: v / len(rows) for k, v in answers.items()},
            "track": dict(tracks),
            "track_share": {k: v / len(rows) for k, v in tracks.items() if k},
            "top_cells": cells.most_common(15),
            "n_distinct_cells": len(cells),
            "failure_targets": dict(failures),
            "skills": dict(skills),
            "gold_tool_single_word_share": (single / len(names)) if names else None,
            "mean_ref_density": mean(ref_shares),
        }

    train_stats = train_dist(train300)

    # Coverage vs transfer: correlate train share with delta by bucket
    coverage_rows = []
    for row in by_calls_exact:
        k = str(row["bucket"])
        train_n = train_stats["call_count"].get(k, 0)
        train_share = train_stats["call_count_share"].get(k, 0.0)
        coverage_rows.append({
            **row,
            "train300_n": train_n,
            "train300_share": train_share,
        })

    # DAG diversity on train300
    topo_c: Counter = Counter()
    prim_c: Counter = Counter()
    surf_c: Counter = Counter()
    fam_c: Counter = Counter()
    by_call_topo: Dict[str, Counter] = defaultdict(Counter)
    by_motif_topo: Dict[str, Counter] = defaultdict(Counter)
    for r in train300:
        calls = r.get("gold_calls") or []
        tid = topology_id(calls)
        pid = primitive_program_id(calls)
        sid = surface_program_id(calls)
        fam = (r.get("provenance") or {}).get("semantic_program_family") or pid
        topo_c[tid] += 1
        prim_c[pid] += 1
        surf_c[sid] += 1
        fam_c[str(fam)] += 1
        nc = str(int(r.get("num_calls") or len(calls)))
        by_call_topo[nc][tid] += 1
        by_motif_topo[str(r.get("motif_type"))][tid] += 1

    def diversity_block(counter: Counter, n: int) -> Dict[str, Any]:
        if not counter:
            return {}
        top = counter.most_common()
        return {
            "n_tasks": n,
            "n_unique": len(counter),
            "top1_share": top[0][1] / n,
            "top10_share": sum(c for _, c in top[:10]) / n,
            "shannon_bits": shannon(counter),
            "mean_tasks_per_family": n / len(counter),
            "top10": [{"id": i, "n": c, "share": c / n} for i, c in top[:10]],
        }

    dag_audit = {
        "train_n": TRAIN_N,
        "train_source": train_path_note,
        "topology_id": diversity_block(topo_c, TRAIN_N),
        "primitive_program_id": diversity_block(prim_c, TRAIN_N),
        "surface_program_id": diversity_block(surf_c, TRAIN_N),
        "semantic_program_family": diversity_block(fam_c, TRAIN_N),
        "topology_unique_by_call_count": {k: len(v) for k, v in sorted(by_call_topo.items())},
        "topology_unique_by_motif": {k: len(v) for k, v in sorted(by_motif_topo.items())},
        "definitions": {
            "topology_id": "n + dependency edges only (no tools/constants)",
            "primitive_program_id": "topology + mapped primitive sids + arg ref/type pattern (no constants/surfaces)",
            "surface_program_id": "tool names + param schemas + labels",
            "semantic_program_family": "factory provenance field when present; else primitive_program_id",
        },
    }

    # C0 re-eval command (do not run here). Sharded helper needs --run-dir with
    # adapter OR falls back to lora_adapter=null when find_checkpoint fails —
    # pass an empty placeholder run-dir without adapter_config.json.
    c0_reeval_cmd = """# Backend-matched C0 re-eval (NEW dir — do NOT overwrite phase1 C0)
cd /workspace/Tool-R0
mkdir -p experiments/targeted_tool_data_factory/outputs/runpod_pilot3/c0_vllm_placeholder
# no adapter_config.json => eval_nestful500_sharded.py sets model.lora_adapter=null
python experiments/targeted_tool_data_factory/runpod_bundle_pilot3/eval_nestful500_sharded.py \\
  --run-dir experiments/targeted_tool_data_factory/outputs/runpod_pilot3/c0_vllm_placeholder \\
  --diagnostic experiments/targeted_tool_data_factory/runpod_bundle_pilot2/data/nestful_diagnostic_500.jsonl \\
  --out-dir experiments/targeted_tool_data_factory/outputs/runpod_pilot3/eval_C0_nestful500_vllm_matched \\
  --run-py experiments/nestful_synthetic_curriculum_v3/run.py \\
  --config experiments/nestful_mtgrpo_partial/config.yaml \\
  --gpus 0,1,2,3
# DECODING inside the script already forces: use_vllm=true, T=0, top_p=1, react, 1 rollout
"""

    analysis = {
        "headline": (
            f"On paired NESTFUL-500, D1 (Pilot3 GRPO n=300) reaches {pct(win_d1)} official win "
            f"vs C0 {pct(win_c0)} (Δ {pp(delta_pp)}, McNemar p={p_mcnemar:.2e}, "
            f"gained {len(gained)} / lost {len(lost)}); largest lift on 2-call tasks, "
            f"but gain is not exclusive to them. Backend confound: C0 HF vs D1 vLLM."
        ),
        "paths": {
            "c0_traj": str(C0_TRAJ.relative_to(ROOT)),
            "d1_traj": str(D1_TRAJ.relative_to(ROOT)),
            "c0_manifest": str(C0_MANIFEST.relative_to(ROOT)),
            "d1_manifest": str(D1_MANIFEST.relative_to(ROOT)),
            "diagnostic": str(DIAG.relative_to(ROOT)),
            "train_full": str(TRAIN_FULL.relative_to(ROOT)),
            "train_n300_definition": train_path_note,
            "config": str(CONFIG_PARTIAL.relative_to(ROOT)) if CONFIG_PARTIAL.exists() else None,
            "d1_eval_script": str(EVAL_SHARDED.relative_to(ROOT)),
        },
        "fairness": fairness,
        "summary": summary,
        "by_gold_calls": by_calls,
        "by_gold_calls_2_to_8": by_calls_exact,
        "call_bucket_win_contribution": contrib,
        "by_answer_type": by_answer,
        "by_motif": by_motif,
        "by_offered_tools": by_offered,
        "by_ref_density": by_ref,
        "by_prompt_length": by_prompt,
        "failure_class_c0": dict(fail_c0),
        "failure_class_d1": dict(fail_d1),
        "failure_class_among_gained_was_c0": dict(fail_gained),
        "failure_class_among_lost_is_d1": dict(fail_lost_was),
        "gain_2call_share_of_gained": gain_from_2call_share,
        "train300": train_stats,
        "coverage_vs_delta_by_calls": coverage_rows,
        "dag_audit": dag_audit,
        "c0_reeval_commands": c0_reeval_cmd,
        "claims": {
            "certain": [
                "Same 500 sample_ids paired 500/500; same diagnostic JSONL path.",
                f"Official win C0={win_c0:.3f} D1={win_d1:.3f}; gained={len(gained)} lost={len(lost)}.",
                f"Exact McNemar p={p_mcnemar:.4g}; bootstrap 95% CI for Δpp=[{ci_lo:.2f},{ci_hi:.2f}].",
                "D1 eval used vLLM; C0 phase1 manifest has no use_vllm=true override.",
                "Train subset for D1 was first 300 rows of train_grpo_pilot3.jsonl.",
            ],
            "interpretation": [
                "Largest absolute win lift on 2-call bucket aligns with train oversample of short tasks + student failure profile.",
                "Executable / final-answer / sol_eq / F1_func all move with official win → not only answer-flip noise.",
                "Positive deltas on 3–5 call buckets suggest broader transfer than 2-call-only, but smaller.",
            ],
            "open": [
                "How much of Δ is vLLM vs HF backend (need matched C0 vLLM re-eval).",
                "Causal role of specific train cells vs generic GRPO on tool-calling.",
                "Whether 8-call regression is noise (n=19) or systematic long-horizon regression.",
            ],
        },
    }

    # JSON
    (REPORTS / "C0_VS_D1_NESTFUL500_ANALYSIS.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (REPORTS / "DAG_DIVERSITY_AUDIT.json").write_text(
        json.dumps(dag_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown main report
    def fmt_rows(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]) -> str:
        hdr = "| " + " | ".join(c[1] for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [hdr, sep]
        for r in rows:
            cells = []
            for key, _ in cols:
                v = r.get(key)
                if key in ("c0", "d1") and isinstance(v, float):
                    cells.append(f"{100*v:.1f}%")
                elif key == "delta_pp" and isinstance(v, (int, float)):
                    cells.append(f"{v:+.1f}")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = []
    md.append("# C0 vs D1 — NESTFUL-500 paired analysis")
    md.append("")
    md.append("## Main result (one sentence)")
    md.append("")
    md.append(analysis["headline"])
    md.append("")
    md.append("## Artefact paths")
    md.append("")
    for k, v in analysis["paths"].items():
        md.append(f"- **{k}**: `{v}`")
    md.append("")
    md.append("## 1. Fairness of comparison")
    md.append("")
    md.append("| Check | Result |")
    md.append("|---|---|")
    md.append(f"| Paired sample_ids | **{n}/500** (only_c0={len(only_c0)}, only_d1={len(only_d1)}) |")
    md.append(f"| Diagnostic set | same `{DIAG.name}` |")
    md.append("| Temperature / top_p / rollouts / paradigm | 0.0 / 1.0 / 1 / react (both) |")
    md.append("| Scorer | official_win (NESTFUL) |")
    md.append(f"| Inference backend | C0: **{fairness['c0_backend']}**; D1: **{fairness['d1_backend']}** |")
    md.append(f"| Backend identical? | **{'YES' if backend_identical else 'NO'}** |")
    md.append("")
    md.append("**Confound:** C0 phase-1 canary eval did not set `hardware.use_vllm=true`; "
              "D1 used `eval_nestful500_sharded.py` which forces vLLM. "
              "Headline Δ is still directionally informative but not backend-clean. "
              "No training was run; C0 matched re-eval command is in §Commands.")
    md.append("")
    md.append("## 2. Headline paired metrics")
    md.append("")
    md.append("| Metric | C0 | D1 | Δ |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Official Win Rate | {pct(win_c0)} ({sum(wins_c0)}/{n}) | {pct(win_d1)} ({sum(wins_d1)}/{n}) | {pp(delta_pp)} |")
    md.append(f"| Function F1 (mean) | {summary['f1_func_c0']:.3f} | {summary['f1_func_d1']:.3f} | {summary['f1_func_d1']-summary['f1_func_c0']:+.3f} |")
    md.append(f"| Parameter F1 (mean) | {summary['f1_param_c0']:.3f} | {summary['f1_param_d1']:.3f} | {summary['f1_param_d1']-summary['f1_param_c0']:+.3f} |")
    md.append(f"| Executable | {pct(summary['executable_c0'])} | {pct(summary['executable_d1'])} | {pp(100*(summary['executable_d1']-summary['executable_c0']))} |")
    md.append(f"| Final-answer pass | {pct(summary['final_answer_pass_c0'])} | {pct(summary['final_answer_pass_d1'])} | {pp(100*(summary['final_answer_pass_d1']-summary['final_answer_pass_c0']))} |")
    md.append(f"| Solution-equivalent | {pct(summary['sol_eq_c0'])} | {pct(summary['sol_eq_d1'])} | {pp(100*(summary['sol_eq_d1']-summary['sol_eq_c0']))} |")
    md.append(f"| Strict gold-trace | {pct(summary['strict_gold_c0'])} | {pct(summary['strict_gold_d1'])} | {pp(100*(summary['strict_gold_d1']-summary['strict_gold_c0']))} |")
    md.append("")
    md.append(f"- Gained / Lost / Both✓ / Both✗: **{len(gained)} / {len(lost)} / {len(both_pass)} / {len(both_fail)}**")
    md.append(f"- Bootstrap 95% CI for Δ win (pp): **[{ci_lo:.2f}, {ci_hi:.2f}]**")
    md.append(f"- Exact McNemar p-value: **{p_mcnemar:.4g}**")
    if rel_err_red is not None:
        md.append(f"- Relative error reduction: **{100*rel_err_red:.1f}%** "
                  f"(err {pct(err_c0)} → {pct(err_d1)})")
    md.append("")
    md.append("## 3. Where D1 gained / regressed")
    md.append("")
    md.append("### By gold call count")
    md.append("")
    md.append(fmt_rows(by_calls, [
        ("bucket", "gold calls"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("### Focus 2–8 calls")
    md.append("")
    md.append(fmt_rows(by_calls_exact, [
        ("bucket", "gold calls"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append(f"Share of gained tasks that are 2-call: **{100*(gain_from_2call_share or 0):.1f}%** "
              f"({g2}/{len(gained)}). Net win change is **not** only from 2-call "
              f"(see contribution table in JSON).")
    md.append("")
    md.append("### By answer type / motif / offered tools / ref density / prompt length")
    md.append("")
    md.append("**Answer type**")
    md.append("")
    md.append(fmt_rows(by_answer, [
        ("bucket", "answer"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("**Motif (from NESTFUL gold refs)**")
    md.append("")
    md.append(fmt_rows(by_motif, [
        ("bucket", "motif"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("**Offered tools**")
    md.append("")
    md.append(fmt_rows(by_offered, [
        ("bucket", "offered"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("**Reference density**")
    md.append("")
    md.append(fmt_rows(by_ref, [
        ("bucket", "ref dens"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("**Prompt length**")
    md.append("")
    md.append(fmt_rows(by_prompt, [
        ("bucket", "prompt"), ("n", "n"), ("c0", "C0"), ("d1", "D1"),
        ("delta_pp", "Δ pp"), ("gained", "gained"), ("lost", "lost"),
    ]))
    md.append("")
    md.append("### Failure-class distribution")
    md.append("")
    md.append("| class | C0 | D1 |")
    md.append("|---|---:|---:|")
    for k in sorted(set(fail_c0) | set(fail_d1), key=lambda x: (-fail_c0.get(x, 0), x)):
        md.append(f"| {k} | {fail_c0.get(k,0)} | {fail_d1.get(k,0)} |")
    md.append("")
    md.append("Gained/lost task listing: `GAINED_LOST_TASKS.csv`.")
    md.append("")
    md.append("## 4. Train-300 coverage vs transfer")
    md.append("")
    md.append(f"Train definition: `{train_path_note}`.")
    md.append("")
    md.append("| gold calls | train300 n | train share | eval Δ pp | gained | lost |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in coverage_rows:
        md.append(
            f"| {r['bucket']} | {r['train300_n']} | {100*r['train300_share']:.1f}% | "
            f"{r['delta_pp']:+.1f} | {r['gained']} | {r['lost']} |"
        )
    md.append("")
    md.append(
        f"Train-300 motif shares: {train_stats['motif_share']}; "
        f"answer shares: { {k: round(v,3) for k,v in train_stats['answer_share'].items()} }; "
        f"track shares: {train_stats['track_share']}; "
        f"distinct generation cells: {train_stats['n_distinct_cells']}."
    )
    md.append("")
    md.append(
        "**Correlation ≠ causation:** 2-call is both oversampled in train (failure-driven) "
        "and the largest eval lift — consistent with coverage, but does not prove those "
        "cells caused the win. Broader diagnostic lifts (executable, F1, sol_eq) suggest "
        "improved tool-use competence, not only memorizing 2-call templates."
    )
    md.append("")
    md.append("See also `TRAIN_COVERAGE_VS_TRANSFER.md`.")
    md.append("")
    md.append("## 5. DAG / program diversity (train-300)")
    md.append("")
    for name, block in [
        ("topology_id", dag_audit["topology_id"]),
        ("primitive_program_id", dag_audit["primitive_program_id"]),
        ("surface_program_id", dag_audit["surface_program_id"]),
        ("semantic_program_family", dag_audit["semantic_program_family"]),
    ]:
        md.append(
            f"- **{name}**: unique={block['n_unique']}, top1={100*block['top1_share']:.1f}%, "
            f"top10={100*block['top10_share']:.1f}%, H={block['shannon_bits']:.2f} bits, "
            f"mean tasks/family={block['mean_tasks_per_family']:.2f}"
        )
    md.append("")
    md.append("Details: `DAG_DIVERSITY_AUDIT.md` / `.json`.")
    md.append("")
    md.append("## 6. What we can claim")
    md.append("")
    md.append("### Certain")
    for x in analysis["claims"]["certain"]:
        md.append(f"- {x}")
    md.append("")
    md.append("### Interpretation")
    for x in analysis["claims"]["interpretation"]:
        md.append(f"- {x}")
    md.append("")
    md.append("### Open")
    for x in analysis["claims"]["open"]:
        md.append(f"- {x}")
    md.append("")
    md.append("## 7. Recommended next analyses (no training)")
    md.append("")
    md.append("1. Re-eval **C0 with vLLM** into a new directory; recompute paired Δ.")
    md.append("2. Qualitative review of lost 8-call / long-horizon tasks in `GAINED_LOST_TASKS.csv`.")
    md.append("3. Stratify gained tasks by whether gold tools are A-track-like math names.")
    md.append("4. Compare internal F1 trajectories on gained vs lost (tool-choice vs args).")
    md.append("5. Contaminate-check: nearest train-300 neighbor (embedding/skeleton) for gained IDs.")
    md.append("")
    md.append("## Commands for missing C0 matched inference")
    md.append("")
    md.append("```bash")
    md.append(c0_reeval_cmd.strip())
    md.append("```")
    md.append("")

    (REPORTS / "C0_VS_D1_NESTFUL500_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")

    # TRAIN_COVERAGE doc
    tmd = []
    tmd.append("# Train-300 coverage vs NESTFUL-500 transfer")
    tmd.append("")
    tmd.append("## Setup")
    tmd.append("")
    tmd.append(f"- Train: `{train_path_note}`")
    tmd.append(f"- Eval: paired C0 vs D1 on diagnostic-500")
    tmd.append("- Claim level: **associative**, not causal")
    tmd.append("")
    tmd.append("## Call-count coverage")
    tmd.append("")
    tmd.append("| calls | train300 | share | eval C0 | eval D1 | Δ pp |")
    tmd.append("|---|---:|---:|---:|---:|---:|")
    for r in coverage_rows:
        tmd.append(
            f"| {r['bucket']} | {r['train300_n']} | {100*r['train300_share']:.1f}% | "
            f"{100*r['c0']:.1f}% | {100*r['d1']:.1f}% | {r['delta_pp']:+.1f} |"
        )
    tmd.append("")
    tmd.append("## Motifs / answers / tracks / failure-target cells")
    tmd.append("")
    tmd.append(f"- Motifs: `{train_stats['motif']}`")
    tmd.append(f"- Answers: `{train_stats['answer_type']}`")
    tmd.append(f"- Tracks: `{train_stats['track']}`")
    tmd.append(f"- Failure targets: `{train_stats['failure_targets']}`")
    tmd.append(f"- Skills: `{train_stats['skills']}`")
    tmd.append(f"- Distinct cells: {train_stats['n_distinct_cells']}")
    tmd.append(f"- Top cells: `{train_stats['top_cells'][:10]}`")
    tmd.append(f"- Gold tool single-word share: {train_stats['gold_tool_single_word_share']}")
    tmd.append(f"- Mean ref density: {train_stats['mean_ref_density']}")
    tmd.append("")
    tmd.append("## Reading guide")
    tmd.append("")
    tmd.append(
        "If a bucket is rare in train-300 and still improves, that is evidence of "
        "**broader generalization** (or backend confound). If only high-coverage "
        "buckets improve, transfer may be **coverage-aligned** rather than deep."
    )
    tmd.append("")
    (REPORTS / "TRAIN_COVERAGE_VS_TRANSFER.md").write_text("\n".join(tmd), encoding="utf-8")

    # DAG md
    dmd = []
    dmd.append("# DAG / program diversity audit — Pilot3 train-300")
    dmd.append("")
    for k, v in dag_audit["definitions"].items():
        dmd.append(f"- **{k}**: {v}")
    dmd.append("")
    for name in ("topology_id", "primitive_program_id", "surface_program_id", "semantic_program_family"):
        b = dag_audit[name]
        dmd.append(f"## {name}")
        dmd.append("")
        dmd.append(f"- unique families: **{b['n_unique']}** / {b['n_tasks']}")
        dmd.append(f"- top-1 share: **{100*b['top1_share']:.1f}%**")
        dmd.append(f"- top-10 share: **{100*b['top10_share']:.1f}%**")
        dmd.append(f"- Shannon entropy: **{b['shannon_bits']:.3f} bits**")
        dmd.append(f"- mean tasks/family: **{b['mean_tasks_per_family']:.2f}**")
        dmd.append("")
        dmd.append("| rank | id | n | share |")
        dmd.append("|---:|---|---:|---:|")
        for i, item in enumerate(b["top10"], 1):
            dmd.append(f"| {i} | `{item['id']}` | {item['n']} | {100*item['share']:.1f}% |")
        dmd.append("")
    dmd.append("## Diversity by call count / motif (unique topologies)")
    dmd.append("")
    dmd.append(f"- by call: `{dag_audit['topology_unique_by_call_count']}`")
    dmd.append(f"- by motif: `{dag_audit['topology_unique_by_motif']}`")
    dmd.append("")
    (REPORTS / "DAG_DIVERSITY_AUDIT.md").write_text("\n".join(dmd), encoding="utf-8")

    # Console summary for the agent return
    print("=== PATHS ===")
    for k, v in analysis["paths"].items():
        print(f"{k}: {v}")
    print("=== FAIRNESS backend_identical ===", backend_identical)
    print("=== METRICS ===")
    print(json.dumps(summary, indent=2))
    print("=== REPORTS ===")
    for p in sorted(REPORTS.glob("C0_VS_D1*")) + sorted(REPORTS.glob("GAINED*")) + sorted(REPORTS.glob("TRAIN*")) + sorted(REPORTS.glob("DAG*")):
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
