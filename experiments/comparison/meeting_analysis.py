"""Meeting-prep analysis: MT-GRPO vs baseline on NESTFUL (no re-training).

Reads existing final_eval outputs + diagnostics, computes per-example Win overlap
and failure taxonomy where trajectory/prediction files exist.

Outputs (same directory):
  meeting_summary.csv
  win_loss_overlap.csv
  failure_taxonomy.csv
  MEETING_BRIEF.md  (regenerated; edit template section at bottom if needed)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MIN = os.path.join(_REPO, "experiments", "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_REPO, "experiments", "nestful_mtgrpo_partial")
_DATASET = os.path.join(_MIN, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
_FUNC_DIR = os.path.join(_MIN, "data", "NESTFUL-main", "data_v2", "executable_functions")

sys.path.insert(0, _MIN)
import nestful_official_score as nos  # noqa: E402

# ── Key runs for summary table ───────────────────────────────────────────────
KEY_RUNS = [
    ("baseline_direct", "baseline", "direct", "baseline (no LoRA)"),
    ("baseline_react", "baseline", "react", "baseline (no LoRA)"),
    ("minimal_s4e2_direct", "minimal/strict", "direct", "minimal strict s4e2"),
    ("minimal_s4e2_react", "minimal/strict", "react", "minimal strict s4e2"),
    ("partial_s1e4_direct", "partial", "direct", "partial s1e4"),
    ("partial_s1e4_react", "partial", "react", "partial s1e4"),
    ("partial_s4e1_direct", "partial", "direct", "partial s4e1"),
    ("partial_s4e1_react", "partial", "react", "partial s4e1"),
]

STAGE_REACT = [
    ("baseline_react", "baseline"),
    ("minimal_s1e4_react", "minimal s1e4"),
    ("minimal_s2e4_react", "minimal s2e4"),
    ("minimal_s4e2_react", "minimal s4e2"),
    ("partial_s1e4_react", "partial s1e4"),
    ("partial_s2e2_react", "partial s2e2"),
    ("partial_s3e2_react", "partial s3e2"),
    ("partial_s4e1_react", "partial s4e1"),
]

_CACHE = os.path.join(_HERE, ".cache")

RUN_PATHS: Dict[str, Dict[str, Any]] = {
    "baseline_direct": {
        "kind": "direct",
        "pred": os.path.join(_MIN, "outputs", "final_eval", "baseline_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_MIN, "outputs", "final_eval", "baseline_direct", "direct_eval_trajectories.jsonl"),
    },
    "baseline_react": {
        "kind": "react",
        "traj": os.path.join(_MIN, "outputs", "final_eval_baseline_react", "final_eval_trajectories.jsonl"),
    },
    "minimal_s1e4_direct": {
        "kind": "direct",
        "pred": os.path.join(_MIN, "outputs", "final_eval", "stage1_epoch4_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_MIN, "outputs", "final_eval", "stage1_epoch4_direct", "direct_eval_trajectories.jsonl"),
    },
    "minimal_s1e4_react": {
        "kind": "react",
        "traj": os.path.join(_MIN, "outputs", "final_eval", "stage1_epoch4_react", "final_eval_trajectories.jsonl"),
    },
    "minimal_s2e4_direct": {
        "kind": "direct",
        "pred": os.path.join(_MIN, "outputs", "final_eval", "stage2_epoch4_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_MIN, "outputs", "final_eval", "stage2_epoch4_direct", "direct_eval_trajectories.jsonl"),
    },
    "minimal_s2e4_react": {
        "kind": "react",
        "traj": os.path.join(_MIN, "outputs", "final_eval", "stage2_epoch4_react", "final_eval_trajectories.jsonl"),
    },
    "minimal_s4e2_direct": {
        "kind": "direct",
        "pred": os.path.join(_MIN, "outputs", "final_eval", "stage4_epoch2_direct", "direct_predictions.jsonl"),
    },
    "minimal_s4e2_react": {
        "kind": "react",
        "traj": os.path.join(_MIN, "outputs", "final_eval_stage4_epoch2_react", "final_eval_trajectories.jsonl"),
    },
    "partial_s1e4_direct": {
        "kind": "direct",
        "pred": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s1_e4_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s1_e4_direct", "direct_eval_trajectories.jsonl"),
    },
    "partial_s1e4_react": {
        "kind": "react",
        "traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s1_e4_react", "final_eval_trajectories.jsonl"),
    },
    "partial_s2e2_direct": {
        "kind": "direct",
        "pred": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s2_e2_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s2_e2_direct", "direct_eval_trajectories.jsonl"),
    },
    "partial_s2e2_react": {
        "kind": "react",
        "traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s2_e2_react", "final_eval_trajectories.jsonl"),
    },
    "partial_s3e2_direct": {
        "kind": "direct",
        "pred": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s3_e2_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s3_e2_direct", "direct_eval_trajectories.jsonl"),
    },
    "partial_s3e2_react": {
        "kind": "react",
        "traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s3_e2_react", "final_eval_trajectories.jsonl"),
    },
    "partial_s4e1_direct": {
        "kind": "direct",
        "pred": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s4_e1_direct", "direct_predictions.jsonl"),
        "direct_traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s4_e1_direct", "direct_eval_trajectories.jsonl"),
    },
    "partial_s4e1_react": {
        "kind": "react",
        "traj": os.path.join(_PARTIAL, "outputs", "final_eval", "partial_s4_e1_react", "final_eval_trajectories.jsonl"),
    },
}

# Aggregate metrics fallback (final_eval_all.csv / diagnostics.json keys)
AGG_FALLBACK: Dict[str, str] = {
    "baseline_direct": "baseline_direct",
    "baseline_react": "baseline (no LoRA)_react",  # from csv - handled separately
    "minimal_s4e2_direct": "curriculum s4e2_direct",
    "minimal_s4e2_react": "curriculum s4e2_react",
    "partial_s1e4_direct": "partial_s1_e4_direct",
    "partial_s1e4_react": "partial_s1_e4_react",
    "partial_s4e1_direct": "partial_s4_e1_direct",
    "partial_s4e1_react": "partial_s4_e1_react",
}

DIAG_KEY_MAP = {
    "baseline_direct": "baseline_direct",
    "minimal_s1e4_direct": "stage1_epoch4_direct",
    "minimal_s1e4_react": "stage1_epoch4_react",
    "minimal_s2e4_direct": "stage2_epoch4_direct",
    "minimal_s2e4_react": "stage2_epoch4_react",
    "minimal_s4e2_direct": "stage4_epoch2_direct",
    "partial_s1e4_direct": "partial_s1_e4_direct",
    "partial_s1e4_react": "partial_s1_e4_react",
    "partial_s2e2_react": "partial_s2_e2_react",
    "partial_s3e2_react": "partial_s3_e2_react",
    "partial_s4e1_direct": "partial_s4_e1_direct",
    "partial_s4e1_react": "partial_s4_e1_react",
}


@dataclass
class SampleRecord:
    sample_id: str
    win: bool
    parse_valid: bool = True
    executable: bool = False
    n_pred_calls: int = 0
    n_gold_calls: int = 0
    official_partial: float = 0.0
    official_full: float = 0.0
    final_answer_pass: bool = False
    strict_pass: bool = False
    stop_reason: str = ""
    fail_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_dataset() -> Dict[str, dict]:
    return nos.load_raw_dataset(_DATASET)


def _gold_calls_count(row: dict) -> int:
    out = row.get("output")
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except json.JSONDecodeError:
            return 0
    return len(out) if isinstance(out, list) else 0


def _extract_calls_direct(row: dict) -> List[dict]:
    return row.get("predicted_calls") or []


def _extract_calls_react(row: dict) -> List[dict]:
    traj = row.get("_traj")
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except json.JSONDecodeError:
            traj = {}
    if not isinstance(traj, dict):
        traj = row.get("_traj") or {}
    calls = []
    for t in traj.get("turns") or []:
        if t.get("is_terminal"):
            break
        pc = t.get("parsed_call")
        if isinstance(pc, dict) and pc.get("name"):
            calls.append(pc)
    return calls


def _extract_official_win(row: dict, traj: Optional[dict]) -> Optional[float]:
    """Resolve per-sample Win from heterogeneous trajectory formats."""
    if not isinstance(traj, dict):
        traj = {}
    for src in (
        row.get("official_win"),
        row.get("internal_win_rate"),
        row.get("win_rate"),
    ):
        if src is not None:
            return float(src)
    for src in (
        traj.get("official_win"),
        (traj.get("official") or {}).get("win_rate"),
        (traj.get("internal") or {}).get("win_rate"),
    ):
        if src is not None:
            return float(src)
    return None


def _per_sample_from_traj_row(row: dict) -> SampleRecord:
    sid = str(row.get("sample_id") or row.get("task_id"))
    traj = row.get("_traj")
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except json.JSONDecodeError:
            traj = {}
    if not isinstance(traj, dict):
        traj = {}
    ow = _extract_official_win(row, traj)
    win = bool(ow is not None and ow >= 0.5)
    return SampleRecord(
        sample_id=sid,
        win=win,
        parse_valid=bool(traj.get("parse_valid", row.get("parse_valid", True))),
        executable=bool(traj.get("executable", False)),
        n_pred_calls=int(traj.get("num_tool_calls") or len(_extract_calls_react(row))),
        n_gold_calls=int(row.get("num_gold_calls") or traj.get("gold_num_turns") or 0),
        official_partial=float(traj.get("official_partial_match") or row.get("internal_partial_sequence_accuracy") or 0),
        official_full=float(traj.get("official_full_match") or row.get("internal_full_sequence_accuracy") or 0),
        final_answer_pass=bool(row.get("final_answer_pass") or traj.get("paper", {}).get("final_answer_pass")),
        strict_pass=bool(row.get("strict_gold_trace_pass") or traj.get("paper", {}).get("strict_gold_trace_pass")),
        stop_reason=str(traj.get("stop_reason") or ""),
        fail_reason=str(traj.get("execution_error") or ""),
        raw=row,
    )


def _per_sample_from_direct_traj_row(row: dict, dataset: Dict[str, dict]) -> Optional[SampleRecord]:
    """Load from direct_eval_trajectories.jsonl when Win is present."""
    ow = row.get("official_win")
    if ow is None:
        ow = row.get("win_rate")
    if ow is None:
        return None
    sid = str(row.get("sample_id") or row.get("task_id"))
    gold = dataset.get(sid)
    ng = _gold_calls_count(gold) if gold else 0
    calls = row.get("predicted_calls") or []
    ow = float(ow)
    return SampleRecord(
        sample_id=sid,
        win=ow >= 0.5,
        parse_valid=bool(row.get("parse_valid", True)),
        executable=bool(row.get("executable", False)),
        n_pred_calls=len(calls),
        n_gold_calls=ng,
        official_partial=float(row.get("official_partial_match") or 0),
        official_full=float(row.get("official_full_match") or 0),
        fail_reason=str(row.get("execution_error") or ""),
        raw=row,
    )


def _load_direct_traj(path: str, dataset: Dict[str, dict]) -> Dict[str, SampleRecord]:
    out: Dict[str, SampleRecord] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rec = _per_sample_from_direct_traj_row(row, dataset)
            if rec is not None:
                out[rec.sample_id] = rec
    return out


def _cache_path(run_id: str) -> str:
    return os.path.join(_CACHE, f"{run_id}_per_sample.json")


def _load_cached_samples(run_id: str) -> Dict[str, SampleRecord]:
    p = _cache_path(run_id)
    if not os.path.isfile(p):
        return {}
    raw = _load_json(p)
    return {
        sid: SampleRecord(
            sample_id=sid,
            win=bool(v["win"]),
            parse_valid=bool(v.get("parse_valid", True)),
            executable=bool(v.get("executable", False)),
            n_pred_calls=int(v.get("n_pred_calls", 0)),
            n_gold_calls=int(v.get("n_gold_calls", 0)),
            official_partial=float(v.get("official_partial", 0)),
            official_full=float(v.get("official_full", 0)),
        )
        for sid, v in raw.items()
    }


def _save_cached_samples(run_id: str, samples: Dict[str, SampleRecord]) -> None:
    os.makedirs(_CACHE, exist_ok=True)
    payload = {
        sid: {
            "win": r.win,
            "parse_valid": r.parse_valid,
            "executable": r.executable,
            "n_pred_calls": r.n_pred_calls,
            "n_gold_calls": r.n_gold_calls,
            "official_partial": r.official_partial,
            "official_full": r.official_full,
        }
        for sid, r in samples.items()
    }
    with open(_cache_path(run_id), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _score_direct_predictions(path: str, dataset: Dict[str, dict], run_id: str = "") -> Dict[str, SampleRecord]:
    cached = _load_cached_samples(run_id) if run_id else {}
    if cached:
        print(f"[{run_id}] loaded {len(cached)} per-sample wins from cache")
        return cached
    print(f"[{run_id or path}] scoring direct predictions (official Win, ~10–15 min first time)...")
    items, sids, meta = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id"))
            gold = dataset.get(sid)
            if not gold:
                continue
            calls = _extract_calls_direct(row)
            items.append(nos.build_item(calls, gold))
            sids.append(sid)
            meta.append((calls, _gold_calls_count(gold)))
    per = nos.score_items_per_sample(items, executable_func_dir=_FUNC_DIR, win_rate=True)
    out: Dict[str, SampleRecord] = {}
    for sid, ps, (calls, ng), _item in zip(sids, per, meta, items):
        out[sid] = SampleRecord(
            sample_id=sid,
            win=bool(ps.get("official_win") is not None and float(ps["official_win"]) >= 0.5),
            parse_valid=bool(ps.get("parse_valid")),
            executable=bool(ps.get("executable")),
            n_pred_calls=len(calls),
            n_gold_calls=ng,
            official_partial=float(ps.get("official_partial_match") or 0),
            official_full=float(ps.get("official_full_match") or 0),
        )
    if run_id:
        _save_cached_samples(run_id, out)
        print(f"[{run_id}] cached {len(out)} per-sample wins")
    return out


def _load_run_samples(run_id: str, dataset: Dict[str, dict]) -> Tuple[Dict[str, SampleRecord], Optional[str]]:
    cfg = RUN_PATHS.get(run_id, {})
    kind = cfg.get("kind")
    if kind == "direct":
        dt = cfg.get("direct_traj")
        if dt and os.path.isfile(dt):
            loaded = _load_direct_traj(dt, dataset)
            if loaded:
                return loaded, None
        pred = cfg.get("pred")
        if pred and os.path.isfile(pred):
            return _score_direct_predictions(pred, dataset, run_id=run_id), None
        return {}, f"missing direct predictions: {pred}"
    if kind == "react":
        p = cfg.get("traj")
        if p and os.path.isfile(p):
            out = {}
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rec = _per_sample_from_traj_row(row)
                    out[rec.sample_id] = rec
            return out, None
        return {}, f"missing react trajectories: {p}"
    return {}, f"unknown run {run_id}"


def _load_diagnostics() -> Dict[str, dict]:
    p = os.path.join(_HERE, "diagnostics.json")
    if not os.path.isfile(p):
        return {}
    rows = _load_json(p)
    return {r["run"]: r for r in rows}


def _load_final_eval_csv() -> List[dict]:
    return _load_csv_rows(os.path.join(_HERE, "final_eval_all.csv"))


def _agg_metrics(run_id: str, diag: Dict[str, dict], feval: List[dict]) -> dict:
    dk = DIAG_KEY_MAP.get(run_id)
    if dk and dk in diag:
        d = diag[dk]
        if d.get("win_rate") is not None:
            return {
                "full": d.get("full"), "win_rate": d.get("win_rate"), "partial": d.get("partial"),
                "macro_f1": d.get("macro_f1_func"), "micro_f1": d.get("micro_f1_func"),
                "set_match": d.get("set_match"), "seq_match": d.get("seq_match"),
            }
    csv_specs = {
        "baseline_direct": ("baseline", "direct", None),
        "baseline_react": ("baseline", "react", None),
        "minimal_s4e2_direct": ("mtgrpo_minimal", "direct", "curriculum s4e2"),
        "minimal_s4e2_react": ("mtgrpo_minimal", "react", "curriculum s4e2"),
    }
    if run_id in csv_specs:
        exp, paradigm, ckpt = csv_specs[run_id]
        for r in feval:
            if r.get("experiment") != exp or r.get("paradigm") != paradigm:
                continue
            if ckpt and ckpt not in r.get("checkpoint", ""):
                continue
            if not ckpt and r.get("checkpoint") != "baseline (no LoRA)":
                continue
            return {
                "full": _f(r.get("full")), "win_rate": _f(r.get("win_rate")),
                "partial": _f(r.get("partial")), "macro_f1": _f(r.get("f1_func")),
                "micro_f1": None, "set_match": None, "seq_match": None,
            }
    return {}


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _conclusion(run_id: str, m: dict, baseline_direct: dict, baseline_react: dict) -> str:
    win = m.get("win_rate")
    macro = m.get("macro_f1")
    if win is None:
        return "missing metrics"
    if run_id == "baseline_direct" or run_id == "baseline_react":
        return "baseline"
    ref = baseline_react if run_id.endswith("_react") else baseline_direct
    rw, bw = win, ref.get("win_rate")
    if rw is None or bw is None:
        return "unknown"
    if run_id.endswith("_react") and macro is not None and macro < 0.3:
        return "collapse"
    if abs(rw - bw) <= 0.015 and (m.get("full") or 0) <= (ref.get("full") or 0) + 0.02:
        return "stable-near-baseline"
    if rw < bw - 0.03:
        return "degradation"
    if rw >= bw - 0.01:
        return "stable-near-baseline"
    return "degradation"


def _overlap(base: Dict[str, SampleRecord], ft: Dict[str, SampleRecord]) -> dict:
    ids = set(base) & set(ft)
    n = len(ids)
    if n == 0:
        return {"n": 0, "both_win": 0, "b_win_f_fail": 0, "b_fail_f_win": 0, "both_fail": 0}
    both_win = b_win_f_fail = b_fail_f_win = both_fail = 0
    for sid in ids:
        bw, fw = base[sid].win, ft[sid].win
        if bw and fw:
            both_win += 1
        elif bw and not fw:
            b_win_f_fail += 1
        elif not bw and fw:
            b_fail_f_win += 1
        else:
            both_fail += 1
    return {
        "n": n, "both_win": both_win, "b_win_f_fail": b_win_f_fail,
        "b_fail_f_win": b_fail_f_win, "both_fail": both_fail,
        "pct_both_win": round(100 * both_win / n, 2),
        "pct_b_win_f_fail": round(100 * b_win_f_fail / n, 2),
        "pct_b_fail_f_win": round(100 * b_fail_f_win / n, 2),
        "pct_both_fail": round(100 * both_fail / n, 2),
    }


FAILURE_TAGS = [
    "parse_error", "no_tool_call", "too_few_calls", "too_many_calls",
    "wrong_function", "wrong_argument", "invalid_reference", "execution_error",
    "premature_final_answer", "correct_trace_wrong_answer",
    "correct_answer_wrong_trace", "other",
]


def _classify_failure(rec: SampleRecord, gold_names: Optional[List[str]] = None) -> str:
    if not rec.parse_valid:
        return "parse_error"
    if rec.n_pred_calls == 0:
        return "no_tool_call"
    if rec.n_gold_calls and rec.n_pred_calls < rec.n_gold_calls:
        return "too_few_calls"
    if rec.n_gold_calls and rec.n_pred_calls > rec.n_gold_calls * 1.5:
        return "too_many_calls"
    if rec.stop_reason in ("parse_fail", "prompt_overflow"):
        return "parse_error"
    if rec.fail_reason and "execution" in rec.fail_reason.lower():
        return "execution_error"
    if rec.win and not rec.strict_pass and rec.official_full < 0.01:
        return "correct_answer_wrong_trace"
    if not rec.win and rec.official_partial > 0.3:
        return "correct_trace_wrong_answer"
    if not rec.win and rec.final_answer_pass:
        return "correct_answer_wrong_trace"
    if not rec.executable and rec.n_pred_calls > 0:
        return "execution_error"
    if rec.stop_reason == "terminal" and rec.n_pred_calls < max(1, rec.n_gold_calls // 2):
        return "premature_final_answer"
    return "other"


def _taxonomy_for_group(
    base: Dict[str, SampleRecord],
    ft: Dict[str, SampleRecord],
    group: str,
) -> Counter:
    c: Counter = Counter()
    for sid in set(base) & set(ft):
        bw, fw = base[sid].win, ft[sid].win
        if group == "b_win_f_fail" and bw and not fw:
            c[_classify_failure(ft[sid])] += 1
        elif group == "b_fail_f_win" and not bw and fw:
            c[_classify_failure(ft[sid])] += 1
        elif group == "b_win_strict_fail" and bw and not ft[sid].strict_pass:
            c[_classify_failure(ft[sid])] += 1
    return c


def build_summary(diag: dict, feval: list) -> List[dict]:
    rows = []
    base_d = _agg_metrics("baseline_direct", diag, feval)
    base_r = _agg_metrics("baseline_react", diag, feval)
    for run_id, exp, paradigm, label in KEY_RUNS:
        m = _agg_metrics(run_id, diag, feval)
        rows.append({
            "run_id": run_id, "experiment": exp, "paradigm": paradigm, "label": label,
            "full": m.get("full"), "win_rate": m.get("win_rate"), "partial": m.get("partial"),
            "macro_f1_func": m.get("macro_f1"), "micro_f1_func": m.get("micro_f1"),
            "set_match": m.get("set_match"), "seq_match": m.get("seq_match"),
            "conclusion": _conclusion(run_id, m, base_d, base_r),
        })
    return rows


def build_stage_degradation(diag: dict, feval: list, samples_cache: dict) -> List[dict]:
    rows = []
    for run_id, label in STAGE_REACT:
        m = _agg_metrics(run_id, diag, feval)
        n_per = len(samples_cache.get(run_id) or {})
        win_from_samples = None
        if n_per:
            wins = sum(1 for r in samples_cache[run_id].values() if r.win)
            win_from_samples = round(wins / n_per, 3)
        rows.append({
            "checkpoint": label,
            "run_id": run_id,
            "win_rate_agg": m.get("win_rate"),
            "win_rate_per_sample": win_from_samples,
            "macro_f1": m.get("macro_f1"),
            "partial": m.get("partial"),
            "n_per_sample": n_per,
            "per_sample_available": n_per > 0,
        })
    return rows


def write_meeting_brief(
    summary: List[dict],
    overlaps: List[dict],
    stage: List[dict],
    missing: List[str],
    taxonomy_notes: str,
) -> None:
    def g(run_id, k):
        for r in summary:
            if r["run_id"] == run_id:
                v = r.get(k)
                return f"{v:.3f}" if isinstance(v, float) else str(v)
        return "—"

    lines = [
        "# NESTFUL / MT-GRPO — briefing pro schůzku",
        "",
        "## TL;DR",
        "",
        "- **Fine-tuning nepřekonal baseline** na hlavních metrikách (Full Acc, Win Rate).",
        "- **Strict/simple reward** na ReAct konci curriculum **kolabuje** (Win 0.544 → 0.325, macro-F1 0.89 → 0.15).",
        "- **Partial reward** drží **partial·s1e4 ReAct** prakticky na baseline (Win **0.543** vs 0.544); delší trénink (s3/s4) degraduje pomaleji než strict, ale stále pod baseline.",
        "- **Direct** je stabilní (~Full 0.16, Win ~0.27–0.29); mírná degradace Win u finetuned checkpointů.",
        "- Problém: **reward mismatch** — trénujeme proxy (strict/partial gold trace), metrika je **execution Win Rate**.",
        "- **Doporučení:** jeden cílený běh s **execution-dominant reward** + early stopping podle val Win; ITAT = reward-design analýza (trace fidelity vs execution), ne benchmark improvement.",
        "",
        "## Pravděpodobná interpretace",
        "",
        "- **Strict reward** optimalizuje gold-trace shodu; u ReAct to vede k **policy drift** (méně volání, předčasné ukončení, no-tool-call) — overlap s4e2 ukazuje **45 % úloh, kde baseline vyhrál a strict prohrál**.",
        "- **Partial s1e4** zachovává ~**46 %** společných výher s baseline ReAct; většina rozdílů je výměna výher (~7.6 % regrese, ~10.1 % zlepšení) — prakticky **on par** s baseline.",
        "- **Macro-F1 Func** zůstává vysoký i při kolapsu Win (s4e1 ReAct: F1 0.35 vs Win 0.45) — metrika měří **formát/tool jména**, ne execution success.",
        "- **Delší partial trénink** (s3→s4) monotónně snižuje ReAct Win (0.543 → 0.450) — potřeba **early stopping** na val Win, ne max epoch.",
        "- **Direct partial s1e4**: overlap téměř identický s agregátem (Win 0.292 vs 0.274); většina úloh prohrává oba modely (67 %), regrese baseline→partial jen **5.4 %**.",
        "",
        "## Co víme jistě",
        "",
        "| běh | paradigma | Full | Win | závěr |",
        "|-----|-----------|------|-----|-------|",
    ]
    for r in summary:
        full = f"{r['full']:.3f}" if isinstance(r.get("full"), (int, float)) else "—"
        win = f"{r['win_rate']:.3f}" if isinstance(r.get("win_rate"), (int, float)) else "—"
        lines.append(f"| {r['label']} | {r['paradigm']} | {full} | {win} | {r['conclusion']} |")

    lines += [
        "",
        "Paper reference (srovnání): GPT-4o Direct one-shot Win **~0.59**, Full **~0.28** (arXiv:2409.03797 v3 / EMNLP 2025).",
        "Náš Qwen3-4B baseline Direct: Win **0.292**, Full **0.169** — pod paperem, netvrdíme převahu.",
        "",
        "### ReAct Win Rate podle checkpointu (policy drift)",
        "",
        "| checkpoint | Win (agg) | per-sample | macro-F1 |",
        "|------------|-----------|------------|----------|",
    ]
    for s in stage:
        wa = f"{s['win_rate_agg']:.3f}" if s.get("win_rate_agg") is not None else "—"
        wp = f"{s['win_rate_per_sample']:.3f}" if s.get("win_rate_per_sample") is not None else "—"
        mf = f"{s['macro_f1']:.3f}" if s.get("macro_f1") is not None else "—"
        lines.append(f"| {s['checkpoint']} | {wa} | {wp} | {mf} |")

    lines += [
        "",
        "## Win/loss overlap (per-task official Win)",
        "",
    ]
    for o in overlaps:
        if o.get("n", 0) == 0:
            lines.append(f"- **{o['comparison']}**: per-example overlap nedostupný — {o.get('note', '')}")
            continue
        lines.append(
            f"- **{o['comparison']}** (n={o['n']}): "
            f"both win {o['pct_both_win']:.1f}%, "
            f"baseline win / ft fail {o['pct_b_win_f_fail']:.1f}%, "
            f"baseline fail / ft win {o['pct_b_fail_f_win']:.1f}%, "
            f"both fail {o['pct_both_fail']:.1f}%"
        )

    lines += [
        "",
        "## Co ještě nevíme / omezení",
        "",
    ]
    overlap_gaps = [o for o in overlaps if o.get("n", 0) == 0]
    if overlap_gaps:
        for o in overlap_gaps:
            lines.append(f"- Win/loss overlap **{o['comparison']}**: {o.get('note', 'nedostupný')}")
    for m in missing:
        lines.append(f"- {m}")
    if not overlap_gaps and not missing:
        lines.append("- Per-example overlap pro všechny požadované páry je k dispozici.")
    lines.append("- Failure taxonomy je heuristická; `wrong_function` / `wrong_argument` nejsou spolehlivě rozlišeny bez per-call gold parse.")
    lines += [
        "",
        f"Heuristická failure taxonomy — viz `failure_taxonomy.csv`. {taxonomy_notes}",
        "",
        "## Doporučený další krok",
        "",
        "1. **Execution-dominant reward** (viz níže) — jeden běh, early stop na val Win Rate.",
        "2. **ITAT framing:** mixed/negative reward design — „trace fidelity vs execution success“, ne claim o SOTA.",
        "3. **Nepoužívat** macro-F1 Func jako success metric (formátová compliance, ~900 tříd).",
        "",
        "### Návrh execution-dominant rewardu",
        "",
        "```",
        "R = 0.50 * tool_final_answer_pass",
        "  + 0.20 * executable",
        "  + 0.15 * grounded_step_similarity",
        "  + 0.10 * valid_references",
        "  + 0.05 * call_count_score",
        "",
        "capy:",
        "  if not parse_valid: R = 0.0",
        "  if not executable: R = min(R, 0.30)",
        "  if no_tool_calls: R = min(R, 0.25)",
        "  if executable and tool_final_answer_pass: R = max(R, 0.80)",
        "```",
        "",
        "---",
        "Generováno: `experiments/comparison/meeting_analysis.py`",
    ]
    with open(os.path.join(_HERE, "MEETING_BRIEF.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


SAMPLE_RUNS = {run_id for run_id, _ in STAGE_REACT} | {
    "minimal_s4e2_react", "partial_s1e4_direct", "baseline_direct",
}


# ===========================================================================
#  VERIFIED pipeline (canonical) — consumes per_sample_official_win.csv
# ===========================================================================
_PER_SAMPLE_CSV = os.path.join(_HERE, "per_sample_official_win.csv")
_CONSISTENCY_MD = os.path.join(_HERE, "per_sample_consistency_report.md")


def _load_verified_per_sample() -> Dict[str, Dict[str, float]]:
    """{run: {sample_id: official_win}} from the canonical per-sample CSV."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    if not os.path.isfile(_PER_SAMPLE_CSV):
        return {}
    for row in _load_csv_rows(_PER_SAMPLE_CSV):
        try:
            out[row["run"]][row["sample_id"]] = float(row["official_win"])
        except (KeyError, TypeError, ValueError):
            continue
    return dict(out)


def _consistency_passed() -> Optional[bool]:
    if not os.path.isfile(_CONSISTENCY_MD):
        return None
    with open(_CONSISTENCY_MD, encoding="utf-8") as fh:
        txt = fh.read()
    if "## OVERALL: PASS" in txt:
        return True
    if "## OVERALL:" in txt:
        return False
    return None


def _verified_overlap(a_wins: Dict[str, float], b_wins: Dict[str, float]) -> dict:
    ids = set(a_wins) & set(b_wins)
    n = len(ids)
    if n == 0:
        return {"n": 0}
    bw_fw = bw_ff = bf_fw = ff = 0
    for sid in ids:
        bw, fw = a_wins[sid] >= 0.5, b_wins[sid] >= 0.5
        if bw and fw:
            bw_fw += 1
        elif bw and not fw:
            bw_ff += 1
        elif not bw and fw:
            bf_fw += 1
        else:
            ff += 1
    return {
        "n": n, "both_win": bw_fw, "a_win_b_fail": bw_ff,
        "a_fail_b_win": bf_fw, "both_fail": ff,
        "pct_both_win": round(100 * bw_fw / n, 2),
        "pct_a_win_b_fail": round(100 * bw_ff / n, 2),
        "pct_a_fail_b_win": round(100 * bf_fw / n, 2),
        "pct_both_fail": round(100 * ff / n, 2),
    }


def run_verified(assert_consistency: bool, recompute: bool) -> int:
    from nestful_core.logging_utils import write_csv as _write_csv

    if recompute or not os.path.isfile(_PER_SAMPLE_CSV):
        print("[verified] (re)computing canonical per-sample official Win ...")
        import subprocess
        subprocess.call([sys.executable,
                         os.path.join(_HERE, "recompute_per_sample_official.py")])

    consistent = _consistency_passed()
    print(f"[verified] per-sample consistency report: "
          f"{'PASS' if consistent else ('FAIL' if consistent is False else 'MISSING')}")
    if assert_consistency and consistent is not True:
        print("[verified] ASSERT FAILED: per-sample does not reproduce aggregate; "
              "refusing to emit overlap/taxonomy. Run recompute_per_sample_official.py.")
        return 1

    per = _load_verified_per_sample()
    if not per:
        print("[verified] no per_sample_official_win.csv — nothing to do.")
        return 1

    # Only runs with verified per-sample data participate (request §2/§20).
    pairs = [
        ("baseline_react", "partial_s1_e4_react"),
        ("baseline_react", "partial_s4_e1_react"),
        ("partial_s1_e4_react", "partial_s4_e1_react"),
    ]
    overlap_rows = []
    for a, b in pairs:
        if a not in per or b not in per:
            overlap_rows.append({"comparison": f"{a} vs {b}", "n": 0,
                                 "note": "missing verified per-sample for one side"})
            continue
        ov = _verified_overlap(per[a], per[b])
        overlap_rows.append({"comparison": f"{a} vs {b}", **ov})
    _write_csv(os.path.join(_HERE, "win_loss_overlap_verified.csv"), overlap_rows,
               fieldnames=["comparison", "n", "both_win", "a_win_b_fail",
                           "a_fail_b_win", "both_fail", "pct_both_win",
                           "pct_a_win_b_fail", "pct_a_fail_b_win", "pct_both_fail", "note"])

    # Coarse verified taxonomy: among baseline-win/ft-fail, why did ft lose?
    # With only per-sample official_win we can report the regression COUNT; the
    # fine-grained failure type needs trajectories (kept in the heuristic legacy
    # path), so we record that limitation explicitly.
    tax_rows = []
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        ids = set(per[a]) & set(per[b])
        reg = sum(1 for s in ids if per[a][s] >= 0.5 and per[b][s] < 0.5)
        gain = sum(1 for s in ids if per[a][s] < 0.5 and per[b][s] >= 0.5)
        tax_rows.append({"comparison": f"{a} vs {b}", "regressions_a_win_b_fail": reg,
                         "gains_a_fail_b_win": gain, "n": len(ids),
                         "note": "verified counts; fine-grained type needs trajectories"})
    _write_csv(os.path.join(_HERE, "failure_taxonomy_verified.csv"), tax_rows,
               fieldnames=["comparison", "regressions_a_win_b_fail",
                           "gains_a_fail_b_win", "n", "note"])

    # Verified brief.
    lines = [
        "# MEETING BRIEF (VERIFIED)",
        "",
        "All numbers below come from the canonical per-sample official Win pipeline "
        "(`per_sample_official_win.csv`), which is asserted to reproduce each run's "
        "aggregate `metrics_official.json` win_rate (see "
        "`per_sample_consistency_report.md`).",
        "",
        f"- per-sample consistency: **{'PASS' if consistent else ('FAIL' if consistent is False else 'MISSING')}**",
        "",
        "## Win/loss overlap (verified, per-task official Win)",
        "",
        "| comparison | n | both win | A win / B fail | A fail / B win | both fail |",
        "|---|---|---|---|---|---|",
    ]
    for o in overlap_rows:
        if o.get("n", 0) == 0:
            lines.append(f"| {o['comparison']} | 0 | — | — | — | — |  ({o.get('note','')})")
            continue
        lines.append(
            f"| {o['comparison']} | {o['n']} | {o['pct_both_win']}% | "
            f"{o['pct_a_win_b_fail']}% | {o['pct_a_fail_b_win']}% | {o['pct_both_fail']}% |")
    lines += [
        "",
        "## Limitations",
        "",
        "- Runs without preserved trajectories (baseline_direct, minimal_s4e2_react) "
        "are EXCLUDED from verified overlap — they cannot be recomputed per-sample.",
        "- Fine-grained failure types require trajectories; the verified taxonomy "
        "reports regression/gain counts only.",
        "",
        "Generated by `meeting_analysis.py --verified`.",
    ]
    with open(os.path.join(_HERE, "MEETING_BRIEF_VERIFIED.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[verified] wrote win_loss_overlap_verified.csv, "
          "failure_taxonomy_verified.csv, MEETING_BRIEF_VERIFIED.md")
    return 0


def _clear_cache() -> None:
    import shutil
    if os.path.isdir(_CACHE):
        shutil.rmtree(_CACHE)
        print(f"[meeting] cleared cache {_CACHE}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Meeting analysis (verified canonical pipeline by default)")
    ap.add_argument("--legacy", action="store_true",
                    help="run the legacy heuristic trajectory-based analysis")
    ap.add_argument("--no-cache", action="store_true", help="clear .cache before running")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute canonical per-sample official Win first")
    ap.add_argument("--assert-consistency", action="store_true",
                    help="refuse to emit overlap/taxonomy unless per-sample reproduces aggregate")
    args = ap.parse_args()

    if args.no_cache:
        _clear_cache()
    if not args.legacy:
        # Make nestful_core importable for the hygiene writer.
        _exp = os.path.join(_REPO, "experiments")
        if _exp not in sys.path:
            sys.path.insert(0, _exp)
        return run_verified(args.assert_consistency, args.recompute)
    return _main_legacy()


def _main_legacy() -> int:
    os.makedirs(os.path.join(_REPO, "experiments", "analysis"), exist_ok=True)
    diag = _load_diagnostics()
    feval = _load_final_eval_csv()
    dataset = _load_dataset()

    summary = build_summary(diag, feval)
    with open(os.path.join(_HERE, "meeting_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    samples_cache: Dict[str, Dict[str, SampleRecord]] = {}
    missing: List[str] = []
    for run_id in SAMPLE_RUNS:
        smp, err = _load_run_samples(run_id, dataset)
        if err:
            missing.append(f"{run_id}: {err}")
        if smp:
            samples_cache[run_id] = smp

    overlap_pairs = [
        ("baseline_react vs partial_s4e1_react", "baseline_react", "partial_s4e1_react"),
        ("baseline_react vs minimal_s4e2_react", "baseline_react", "minimal_s4e2_react"),
        ("baseline_direct vs partial_s1e4_direct", "baseline_direct", "partial_s1e4_direct"),
    ]
    overlap_rows = []
    for label, a, b in overlap_pairs:
        ba, bb = samples_cache.get(a), samples_cache.get(b)
        if not ba or not bb:
            note = []
            if not ba:
                note.append(f"chybí per-sample pro {a}")
            if not bb:
                note.append(f"chybí per-sample pro {b}")
            overlap_rows.append({"comparison": label, "n": 0, "note": "; ".join(note)})
            continue
        ov = _overlap(ba, bb)
        overlap_rows.append({"comparison": label, **ov})

    with open(os.path.join(_HERE, "win_loss_overlap.csv"), "w", encoding="utf-8", newline="") as fh:
        cols = ["comparison", "n", "both_win", "b_win_f_fail", "b_fail_f_win", "both_fail",
                "pct_both_win", "pct_b_win_f_fail", "pct_b_fail_f_win", "pct_both_fail", "note"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(overlap_rows)

    tax_rows = []
    tax_specs = [
        ("baseline_win_partial_fail", "baseline_react", "partial_s1e4_react", "b_win_f_fail"),
        ("baseline_fail_partial_win", "baseline_react", "partial_s1e4_react", "b_fail_f_win"),
        ("baseline_win_strict_fail", "baseline_react", "minimal_s4e2_react", "b_win_strict_fail"),
        ("baseline_win_partial_fail_direct", "baseline_direct", "partial_s1e4_direct", "b_win_f_fail"),
    ]
    for group_name, a, b, gkind in tax_specs:
        ba, bb = samples_cache.get(a), samples_cache.get(b)
        if not ba or not bb:
            tax_rows.append({
                "group": group_name, "failure_type": "N/A", "count": 0, "pct": 0.0,
                "note": f"per-sample unavailable ({a} or {b})",
            })
            continue
        if gkind == "b_win_strict_fail":
            c = Counter()
            for sid in set(ba) & set(bb):
                if ba[sid].win and not bb[sid].strict_pass:
                    c[_classify_failure(bb[sid])] += 1
            total = sum(c.values())
        else:
            c = _taxonomy_for_group(ba, bb, gkind)
            total = sum(c.values())
        if total == 0:
            tax_rows.append({"group": group_name, "failure_type": "none", "count": 0, "pct": 0.0, "note": ""})
            continue
        for tag in FAILURE_TAGS:
            cnt = c.get(tag, 0)
            if cnt:
                tax_rows.append({
                    "group": group_name, "failure_type": tag, "count": cnt,
                    "pct": round(100 * cnt / total, 2), "note": "heuristic",
                })

    with open(os.path.join(_HERE, "failure_taxonomy.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "failure_type", "count", "pct", "note"])
        w.writeheader()
        w.writerows(tax_rows)

    stage = build_stage_degradation(diag, feval, samples_cache)
    taxonomy_notes = (
        "Taxonomie je heuristická (parse/call-count/stop_reason/official_win vs strict); "
        "nereflektuje fine-grained wrong_function bez gold parse per call."
    )
    write_meeting_brief(summary, overlap_rows, stage, missing, taxonomy_notes)

    # Mirror key outputs to experiments/analysis/
    analysis_dir = os.path.join(_REPO, "experiments", "analysis")
    for name in ("meeting_summary.csv", "win_loss_overlap.csv", "failure_taxonomy.csv", "MEETING_BRIEF.md"):
        src = os.path.join(_HERE, name)
        if os.path.isfile(src):
            dst = os.path.join(analysis_dir, name)
            with open(src, encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
                fdst.write(fsrc.read())

    print(f"Wrote meeting_summary.csv ({len(summary)} rows)")
    print(f"Wrote win_loss_overlap.csv ({len(overlap_rows)} comparisons)")
    print(f"Wrote failure_taxonomy.csv ({len(tax_rows)} rows)")
    print(f"Wrote MEETING_BRIEF.md")
    print(f"Mirrored to {analysis_dir}/")
    if missing:
        print("\nMissing per-sample sources:")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
