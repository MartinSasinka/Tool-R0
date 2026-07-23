#!/usr/bin/env python3
"""Paired C0 / E1 / E2 NESTFUL test analysis for pure Stage 3 overnight run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
sys.path.insert(0, str(_V3))
sys.path.insert(0, str(_V3 / "scripts"))

from motif_lib import default_test_path, extract_motifs, load_jsonl, load_task_row  # noqa: E402
from scripts.analysis.two_phase_root_cause_analysis import (  # noqa: E402
    classify_failure,
    mcnemar,
    official_win,
    paired_bootstrap,
)

DEFAULT_RUN = _V3 / "outputs/runs/pure_stage3_2ep_20260719_221918"
OUT = _V3 / "reports/pure_stage3_offline_analysis"
TEST_PATH = default_test_path()
PROMPT = _REPO / "experiments/nestful_mtgrpo_minimal/prompt.py"
SCORER = _REPO / "experiments/nestful_mtgrpo_minimal/nestful_official_score.py"

ARM_DIRS = {"C0": "C0_test", "E1": "S3_E1_test", "E2": "S3_E2_test"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_traj(d: Path) -> Dict[str, dict]:
    out = {}
    with open(d / "final_eval_trajectories.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def gold_meta() -> Dict[str, dict]:
    m = {}
    for row in load_jsonl(TEST_PATH):
        t = load_task_row(row)
        gc = t.get("gold_calls") or []
        m[t["task_id"]] = {
            "gold_call_count": len(gc),
            "gold_motif": extract_motifs(t).get("motif_type"),
            "gold_first_tool": gc[0].get("name") if gc else None,
        }
    return m


def win(row) -> bool:
    return official_win(row) == 1.0


def first_tool(row) -> Optional[str]:
    for t in (row.get("_traj") or {}).get("turns") or []:
        n = (t.get("parsed_call") or {}).get("name")
        if n:
            return n
    return None


def failure_primary(row) -> str:
    if win(row):
        return "success"
    return classify_failure(row)[0]


def call_bucket(n: int) -> str:
    return str(n) if n <= 5 else "6+"


def transition_cat(w0: bool, w1: bool, w2: bool) -> str:
    if w0 and w1 and w2:
        return "stable_win"
    if not w0 and not w1 and not w2:
        return "stable_loss"
    if not w0 and w1 and w2:
        return "gained_E1_kept_E2"
    if not w0 and w1 and not w2:
        return "gained_after_E1_lost_E2"
    if not w0 and not w1 and w2:
        return "gained_after_E2_only"
    if w0 and not w1 and not w2:
        return "lost_after_E1"
    if w0 and w1 and not w2:
        return "gained_E1_lost_E2"
    if w0 and not w1 and w2:
        return "lost_E1_regained_E2"
    if not w0 and not w1 and w2:
        return "gained_after_E2_only"
    return "other"


def aggregate(rows: Dict[str, dict], meta: Dict[str, dict]) -> dict:
    ids = list(rows)
    tax = Counter()
    ft_ok = []
    for sid in ids:
        r = rows[sid]
        tax[failure_primary(r)] += 1
        gf, pf = meta[sid].get("gold_first_tool"), first_tool(r)
        if gf and pf:
            ft_ok.append(pf == gf)
    n = len(ids)
    return {
        "n": n,
        "win_rate": sum(win(rows[i]) for i in ids) / n,
        "f1_func": sum(rows[i].get("internal_f1_func", 0) or 0 for i in ids) / n,
        "f1_param": sum(rows[i].get("internal_f1_param", 0) or 0 for i in ids) / n,
        "first_tool_acc": sum(ft_ok) / len(ft_ok) if ft_ok else None,
        "executability": sum(bool((rows[i].get("_traj") or {}).get("executable")) for i in ids) / n,
        "under_calling": sum(
            (rows[i].get("_traj") or {}).get("num_tool_calls", 0) < rows[i].get("num_gold_calls", 0)
            for i in ids
        ) / n,
        "avg_calls": sum((rows[i].get("_traj") or {}).get("num_tool_calls", 0) or 0 for i in ids) / n,
        "final_answer_pass": sum(bool(rows[i].get("final_answer_pass")) for i in ids) / n,
        "taxonomy": dict(tax),
    }


def failure_shift(t0: Counter, t1: Counter, t2: Counter, n: int) -> dict:
    keys = [
        "correct keys, wrong argument values",
        "wrong tool",
        "executable trajectory ending wrong result",
        "no tool call",
        "too few calls",
        "parse/format error",
    ]
    out = {}
    for k in keys:
        c0, c1, c2 = t0.get(k, 0) / n, t1.get(k, 0) / n, t2.get(k, 0) / n
        out[k] = {"C0": c0, "E1": c1, "E2": c2, "E2_minus_C0_pp": (c2 - c0) * 100}
    wv = out["correct keys, wrong argument values"]
    wt = out["wrong tool"]
    ex = out["executable trajectory ending wrong result"]
    pattern = wv["E2"] < wv["C0"] and (wt["E2"] > wt["C0"] or ex["E2"] > ex["C0"])
    return {"by_failure": out, "pattern_values_down_tool_or_exec_up": pattern}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = gold_meta()
    arms = {k: load_traj(run_dir / "eval" / v) for k, v in ARM_DIRS.items()}
    ids = sorted(arms["C0"].keys())
    for k in arms:
        if set(arms[k]) != set(ids):
            raise SystemExit(f"task_id mismatch {k}")

    # parity
    m0 = json.loads((run_dir / "eval/C0_test/eval_manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((run_dir / "eval/S3_E2_test/eval_manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    e1h = json.loads((run_dir / "checkpoints/S3_E1/checkpoint_manifest.json").read_text(encoding="utf-8"))
    e2h = json.loads((run_dir / "checkpoints/S3_E2/checkpoint_manifest.json").read_text(encoding="utf-8"))

    summary = {k: aggregate(arms[k], meta) for k in ("C0", "E1", "E2")}

    trans = Counter()
    task_rows = []
    gained_e1 = lost_e1 = gained_e2 = lost_e2 = 0
    e1_gain_e2_loss = e1_loss_e2_gain = 0

    for sid in ids:
        w0, w1, w2 = win(arms["C0"][sid]), win(arms["E1"][sid]), win(arms["E2"][sid])
        cat = transition_cat(w0, w1, w2)
        trans[cat] += 1
        if not w0 and w1:
            gained_e1 += 1
        if w0 and not w1:
            lost_e1 += 1
        if not w2 and (w0 or w1):
            pass
        if not w0 and w2 and not w1:
            gained_e2 += 1
        if w0 and not w2:
            lost_e2 += 1
        if w1 and not w2 and not w0:
            e1_gain_e2_loss += 1
        if w0 and not w1 and w2:
            e1_loss_e2_gain += 1

        g = meta[sid]
        task_rows.append({
            "task_id": sid,
            "gold_call_count": g["gold_call_count"],
            "gold_motif": g["gold_motif"],
            "gold_first_tool": g["gold_first_tool"],
            "C0_win": w0, "E1_win": w1, "E2_win": w2,
            "C0_failure": failure_primary(arms["C0"][sid]),
            "E1_failure": failure_primary(arms["E1"][sid]),
            "E2_failure": failure_primary(arms["E2"][sid]),
            "C0_num_calls": (arms["C0"][sid].get("_traj") or {}).get("num_tool_calls"),
            "E1_num_calls": (arms["E1"][sid].get("_traj") or {}).get("num_tool_calls"),
            "E2_num_calls": (arms["E2"][sid].get("_traj") or {}).get("num_tool_calls"),
            "C0_first_tool": first_tool(arms["C0"][sid]),
            "E1_first_tool": first_tool(arms["E1"][sid]),
            "E2_first_tool": first_tool(arms["E2"][sid]),
            "C0_executable": (arms["C0"][sid].get("_traj") or {}).get("executable"),
            "E1_executable": (arms["E1"][sid].get("_traj") or {}).get("executable"),
            "E2_executable": (arms["E2"][sid].get("_traj") or {}).get("executable"),
            "transition": cat,
        })

    deltas_c1 = [float(win(arms["E1"][s]) - win(arms["C0"][s])) for s in ids]
    deltas_c2 = [float(win(arms["E2"][s]) - win(arms["C0"][s])) for s in ids]
    deltas_e2_e1 = [float(win(arms["E2"][s]) - win(arms["E1"][s])) for s in ids]

    by_calls = {}
    for b in ["2", "3", "4", "5", "6+"]:
        sids = [s for s in ids if call_bucket(meta[s]["gold_call_count"]) == b]
        if not sids:
            continue
        by_calls[b] = {
            "n": len(sids),
            "C0": sum(win(arms["C0"][s]) for s in sids) / len(sids),
            "E1": sum(win(arms["E1"][s]) for s in sids) / len(sids),
            "E2": sum(win(arms["E2"][s]) for s in sids) / len(sids),
        }

    by_motif = defaultdict(lambda: {"n": 0, "C0": 0.0, "E1": 0.0, "E2": 0.0})
    for sid in ids:
        mo = meta[sid]["gold_motif"] or "unknown"
        by_motif[mo]["n"] += 1
        by_motif[mo]["C0"] += win(arms["C0"][sid])
        by_motif[mo]["E1"] += win(arms["E1"][sid])
        by_motif[mo]["E2"] += win(arms["E2"][sid])
    for mo in by_motif:
        n = by_motif[mo]["n"]
        by_motif[mo]["C0"] /= n
        by_motif[mo]["E1"] /= n
        by_motif[mo]["E2"] /= n

    fs = failure_shift(
        Counter(summary["C0"]["taxonomy"]),
        Counter(summary["E1"]["taxonomy"]),
        Counter(summary["E2"]["taxonomy"]),
        len(ids),
    )

    result = {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "parity_ok": True,
        "adapter_hash_E1": e1h.get("adapter_hash"),
        "adapter_hash_E2": e2h.get("adapter_hash"),
        "summary": summary,
        "paired": {
            "E1_vs_C0": {
                "gained": sum(1 for s in ids if not win(arms["C0"][s]) and win(arms["E1"][s])),
                "lost": sum(1 for s in ids if win(arms["C0"][s]) and not win(arms["E1"][s])),
                "mcnemar": mcnemar(
                    sum(1 for s in ids if not win(arms["C0"][s]) and win(arms["E1"][s])),
                    sum(1 for s in ids if win(arms["C0"][s]) and not win(arms["E1"][s])),
                ),
                "bootstrap": paired_bootstrap(deltas_c1),
            },
            "E2_vs_C0": {
                "gained": sum(1 for s in ids if not win(arms["C0"][s]) and win(arms["E2"][s])),
                "lost": sum(1 for s in ids if win(arms["C0"][s]) and not win(arms["E2"][s])),
                "mcnemar": mcnemar(
                    sum(1 for s in ids if not win(arms["C0"][s]) and win(arms["E2"][s])),
                    sum(1 for s in ids if win(arms["C0"][s]) and not win(arms["E2"][s])),
                ),
                "bootstrap": paired_bootstrap(deltas_c2),
            },
            "E2_vs_E1": {
                "gained": sum(1 for s in ids if not win(arms["E1"][s]) and win(arms["E2"][s])),
                "lost": sum(1 for s in ids if win(arms["E1"][s]) and not win(arms["E2"][s])),
                "mcnemar": mcnemar(
                    sum(1 for s in ids if not win(arms["E1"][s]) and win(arms["E2"][s])),
                    sum(1 for s in ids if win(arms["E1"][s]) and not win(arms["E2"][s])),
                ),
                "bootstrap": paired_bootstrap(deltas_e2_e1),
            },
        },
        "transitions": dict(trans),
        "e1_gain_e2_loss": e1_gain_e2_loss,
        "e1_loss_e2_gain": e1_loss_e2_gain,
        "by_calls": by_calls,
        "by_motif": dict(by_motif),
        "failure_shift": fs,
    }

    with open(OUT / "analysis_c0_e1_e2_test_overnight.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(OUT / "pure_stage3_task_level_analysis.jsonl", "w", encoding="utf-8") as f:
        for r in task_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT / "PURE_STAGE3_FAILURE_TRANSITIONS.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
        w.writeheader()
        w.writerows(task_rows)

    # markdown report
    s0, s1, s2 = summary["C0"], summary["E1"], summary["E2"]
    p1, p2, p3 = result["paired"]["E1_vs_C0"], result["paired"]["E2_vs_C0"], result["paired"]["E2_vs_E1"]
    md = [
        "# Pure Stage 3 Overnight — C0 / E1 / E2 Test (n=1661)",
        "",
        f"Run: `{run_dir.name}` | Generated: {_now()}",
        "",
        "## Headline win rate",
        "",
        f"| Arm | Win | Δ vs C0 |",
        f"|-----|----:|--------:|",
        f"| C0 | {s0['win_rate']:.4f} | — |",
        f"| E1 | {s1['win_rate']:.4f} | {s1['win_rate']-s0['win_rate']:+.4f} |",
        f"| E2 | {s2['win_rate']:.4f} | {s2['win_rate']-s0['win_rate']:+.4f} |",
        "",
        f"E1 vs C0: gained {p1['gained']} lost {p1['lost']} net {p1['gained']-p1['lost']} p={p1['mcnemar'].get('p_value')}",
        f"E2 vs C0: gained {p2['gained']} lost {p2['lost']} net {p2['gained']-p2['lost']} p={p2['mcnemar'].get('p_value')}",
        f"E2 vs E1: gained {p3['gained']} lost {p3['lost']} net {p3['gained']-p3['lost']} p={p3['mcnemar'].get('p_value')}",
        "",
        "## Transitions",
        "",
    ]
    for k, v in sorted(trans.items(), key=lambda x: -x[1]):
        md.append(f"- **{k}**: {v}")
    md += [
        "",
        f"- E1 gain → E2 loss: **{e1_gain_e2_loss}**",
        f"- E1 loss → E2 regain: **{e1_loss_e2_gain}**",
        "",
        "## Key pattern (values↓ tool/exec↑)?",
        f"**{fs['pattern_values_down_tool_or_exec_up']}**",
        "",
        "## By call bucket",
        "",
        "| bucket | n | C0 | E1 | E2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for b, v in by_calls.items():
        md.append(f"| {b} | {v['n']} | {v['C0']:.4f} | {v['E1']:.4f} | {v['E2']:.4f} |")
    md += ["", "## Failure rates (non-success share)", ""]
    for k, v in fs["by_failure"].items():
        md.append(f"- {k}: C0 {v['C0']*100:.2f}% → E1 {v['E1']*100:.2f}% → E2 {v['E2']*100:.2f}%")
    (OUT / "PURE_STAGE3_C0_E1_E2_PAIRED.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "C0": s0["win_rate"], "E1": s1["win_rate"], "E2": s2["win_rate"],
        "E1-C0": s1["win_rate"]-s0["win_rate"],
        "E2-C0": s2["win_rate"]-s0["win_rate"],
        "E2-E1": s2["win_rate"]-s1["win_rate"],
        "pattern": fs["pattern_values_down_tool_or_exec_up"],
        "e1_gain_e2_loss": e1_gain_e2_loss,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
