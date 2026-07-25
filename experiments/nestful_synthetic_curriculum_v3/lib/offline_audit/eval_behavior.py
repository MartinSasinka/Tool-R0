from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.offline_audit import ARMS
from lib.offline_audit.paths import c0_eval_dir, eval_dir


def _load_eval_rows(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("task_id")
            if sid:
                out[str(sid)] = r
    return out


def _win(rec: Dict) -> bool:
    traj = rec.get("_traj") or {}
    w = traj.get("official_win")
    if w is None:
        w = rec.get("official_win")
    if w is None:
        w = rec.get("win")
    return w in (1, 1.0, True)


def _call_trace(rec: Dict) -> str:
    traj = rec.get("_traj") or rec.get("trajectory") or {}
    turns = traj.get("turns") or []
    parts = []
    for tn in turns:
        c = tn.get("parsed_call")
        if not c:
            continue
        name = c.get("name") or c.get("tool_name")
        args = c.get("arguments") or c.get("args") or {}
        parts.append(f"{name}|{json.dumps(args, sort_keys=True)}")
    return "->".join(parts)


def eval_behavior(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    paths = {arm: eval_dir(runs_root, arm, seed) / "final_eval_trajectories.jsonl" for arm in ARMS}
    paths["C0"] = c0_eval_dir(runs_root, seed) / "final_eval_trajectories.jsonl"
    loaded = {k: _load_eval_rows(p) for k, p in paths.items() if p.is_file()}

    def compare(a: str, b: str) -> Dict[str, Any]:
        if a not in loaded or b not in loaded:
            return {"error": "missing eval"}
        shared = set(loaded[a]) & set(loaded[b])
        win_agree = 0
        trace_agree = 0
        n = 0
        discordant = []
        both_win = both_loss = a_only = b_only = 0
        for tid in shared:
            ra, rb = loaded[a][tid], loaded[b][tid]
            wa, wb = _win(ra), _win(rb)
            n += 1
            if wa == wb:
                win_agree += 1
            if _call_trace(ra) == _call_trace(rb):
                trace_agree += 1
            if wa and wb:
                both_win += 1
            elif not wa and not wb:
                both_loss += 1
            elif wa and not wb:
                a_only += 1
            elif wb and not wa:
                b_only += 1
            if wa != wb or _call_trace(ra) != _call_trace(rb):
                if a == "A0_R0_CURRENT" and b == "A4_GATED_VERIFIABLE" and len(discordant) < 500:
                    discordant.append(
                        {
                            "task_id": tid,
                            "a0_win": wa,
                            "a4_win": wb,
                            "a0_trace": _call_trace(ra)[:500],
                            "a4_trace": _call_trace(rb)[:500],
                        }
                    )
        return {
            "n_tasks": n,
            "win_agreement": win_agree / n if n else None,
            "trace_agreement": trace_agree / n if n else None,
            "both_win": both_win,
            "both_loss": both_loss,
            "a_only_win": a_only,
            "b_only_win": b_only,
            "discordant": discordant,
        }

    pairs = [
        ("A0_R0_CURRENT", "A4_GATED_VERIFIABLE"),
        ("A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST"),
        ("C0", "A0_R0_CURRENT"),
        ("C0", "A4_GATED_VERIFIABLE"),
    ]
    results = {}
    csv_rows = []
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        results[key] = compare(a, b)
        r = results[key]
        if "error" in r:
            continue
        csv_rows.append(
            {
                "pair": key,
                "n_tasks": r["n_tasks"],
                "win_agreement": r["win_agreement"],
                "trace_agreement": r["trace_agreement"],
            }
        )

    a0a4 = results.get("A0_R0_CURRENT_vs_A4_GATED_VERIFIABLE", {})
    if a0a4.get("discordant"):
        with open(reports_dir / "a0_vs_a4_discordant_cases.jsonl", "w", encoding="utf-8") as fh:
            for row in a0a4["discordant"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if csv_rows:
        with open(reports_dir / "eval_behavior_similarity.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    md = ["# Eval behavior similarity (500 diagnostic tasks)", ""]
    for key, r in results.items():
        if "error" in r:
            md.append(f"## {key}: MISSING")
            continue
        md.append(f"## {key}")
        md.append(f"- win agreement: **{r['win_agreement']:.4f}**")
        md.append(f"- trace agreement: **{r['trace_agreement']:.4f}**")
        if "A0" in key and "A4" in key:
            md.append(
                f"- both win / both loss / A0-only / A4-only: "
                f"{r['both_win']} / {r['both_loss']} / {r['a_only_win']} / {r['b_only_win']}"
            )
        md.append("")
    (reports_dir / "EVAL_BEHAVIOR_SIMILARITY.md").write_text("\n".join(md), encoding="utf-8")
    return results
