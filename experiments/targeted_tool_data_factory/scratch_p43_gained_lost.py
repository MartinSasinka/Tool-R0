"""Paired baseline vs step200 NESTFUL analysis (gained/lost + hard failure types)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(r"c:\Users\Šunka\Downloads\p43_nestful_t0(1)\p43_nestful_t0\baseline\final_eval_trajectories.jsonl")
NEW = Path(r"c:\Users\Šunka\Downloads\p43_nestful_t0_step200\p43_nestful_t0_step200\step200\final_eval_trajectories.jsonl")
OUT = Path(r"c:\Users\Šunka\Downloads\p43_nestful_t0_step200\GAINED_LOST_ANALYSIS")
OUT.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r.get("sample_id")
            traj = r.get("_traj") or {}
            tid = traj.get("task_id") or sid
            official = traj.get("official_win")
            if official is None:
                official = r.get("internal_win_rate")
            win = bool(float(official or 0.0) >= 0.5)
            n_gold = r.get("num_gold_calls")
            if n_gold is None:
                n_gold = traj.get("gold_num_turns")
            bucket = "6+" if (n_gold or 0) >= 6 else str(n_gold or "?")
            turns = traj.get("turns") or []
            n_pred_calls = traj.get("num_tool_calls")
            if n_pred_calls is None:
                n_pred_calls = sum(1 for t in turns if (t.get("tool_calls") or t.get("parsed_calls")))
            # failure / behavior tags
            tags = []
            if not traj.get("parse_valid", True):
                tags.append("parse_invalid")
            if traj.get("prompt_overflow"):
                tags.append("prompt_overflow")
            if traj.get("clipped_any"):
                tags.append("clipped")
            if traj.get("execution_error"):
                tags.append("execution_error")
            if traj.get("executable") is False:
                tags.append("not_executable")
            if (n_pred_calls or 0) == 0:
                tags.append("no_tool_calls")
            if (n_gold or 0) > 0 and (n_pred_calls or 0) > 0 and (n_pred_calls or 0) < (n_gold or 0):
                tags.append("too_few_calls")
            if (n_gold or 0) > 0 and (n_pred_calls or 0) > (n_gold or 0) + 1:
                tags.append("too_many_calls")
            if r.get("correct_answer_but_unsupported_trace"):
                tags.append("answer_ok_trace_bad")
            if r.get("strict_fail_but_solution_equivalent_pass"):
                tags.append("solution_equivalent_only")
            if not r.get("final_answer_pass", True) and win is False:
                tags.append("final_answer_fail")
            stop = traj.get("stop_reason")
            if stop:
                tags.append(f"stop:{stop}")
            mismatch = traj.get("mismatch_reason") or traj.get("mismatch")
            if mismatch:
                tags.append(f"mismatch:{mismatch}" if not str(mismatch).startswith("mismatch:") else str(mismatch))
            out[str(sid)] = {
                "sample_id": sid,
                "task_id": tid,
                "win": win,
                "official_win": float(official or 0.0),
                "num_gold_calls": int(n_gold or 0),
                "call_bucket": bucket,
                "num_pred_calls": int(n_pred_calls or 0),
                "f1_func": float(r.get("internal_f1_func") or 0.0),
                "f1_param": float(r.get("internal_f1_param") or 0.0),
                "partial_seq": float(r.get("internal_partial_sequence_accuracy") or 0.0),
                "full_seq": float(r.get("internal_full_sequence_accuracy") or 0.0),
                "strict_gold": bool(r.get("strict_gold_trace_pass")),
                "sol_eq": bool(r.get("solution_equivalent_pass")),
                "final_answer_pass": bool(r.get("final_answer_pass")),
                "executable": traj.get("executable"),
                "parse_valid": traj.get("parse_valid"),
                "stop_reason": stop,
                "mismatch_reason": traj.get("mismatch_reason"),
                "tags": tags,
                "pred_answer": traj.get("pred_answer"),
            }
    return out


def rate(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def main() -> None:
    base = load(BASE)
    new = load(NEW)
    common = sorted(set(base) & set(new))
    only_b = sorted(set(base) - set(new))
    only_n = sorted(set(new) - set(base))
    print(f"baseline={len(base)} step200={len(new)} common={len(common)} only_base={len(only_b)} only_new={len(only_n)}")

    gained: List[str] = []
    lost: List[str] = []
    both_win: List[str] = []
    both_lose: List[str] = []
    for sid in common:
        bw, nw = base[sid]["win"], new[sid]["win"]
        if (not bw) and nw:
            gained.append(sid)
        elif bw and (not nw):
            lost.append(sid)
        elif bw and nw:
            both_win.append(sid)
        else:
            both_lose.append(sid)

    b_wins = sum(1 for s in common if base[s]["win"])
    n_wins = sum(1 for s in common if new[s]["win"])
    print(f"paired win baseline={b_wins}/{len(common)}={rate(b_wins,len(common)):.4f}")
    print(f"paired win step200={n_wins}/{len(common)}={rate(n_wins,len(common)):.4f}")
    print(f"gained={len(gained)} lost={len(lost)} net={len(gained)-len(lost)}")
    print(f"both_win={len(both_win)} both_lose={len(both_lose)}")

    def bucket_table(sids: List[str], which: str) -> List[Tuple[str, int, int, float]]:
        rows = []
        for bucket in ["2", "3", "4", "5", "6+", "?"]:
            ids = [s for s in common if base[s]["call_bucket"] == bucket]
            if which == "all":
                bw = sum(1 for s in ids if base[s]["win"])
                nw = sum(1 for s in ids if new[s]["win"])
                rows.append((bucket, len(ids), nw - bw, rate(nw, len(ids)) - rate(bw, len(ids))))
            elif which == "gained":
                g = sum(1 for s in ids if s in set(gained))
                rows.append((bucket, len(ids), g, rate(g, len(ids))))
            elif which == "lost":
                l = sum(1 for s in ids if s in set(lost))
                rows.append((bucket, len(ids), l, rate(l, len(ids))))
            elif which == "hard":
                # still failing on step200
                fail = [s for s in ids if not new[s]["win"]]
                rows.append((bucket, len(ids), len(fail), rate(len(fail), len(ids))))
        return rows

    print("\n=== Win by call bucket ===")
    print("bucket | n | d_wins | d_pp")
    for b, n, dw, dpp in bucket_table(common, "all"):
        print(f"{b:>6} | {n:4d} | {dw:+4d} | {100*dpp:+.1f}pp")

    print("\n=== Gained / Lost by call bucket ===")
    gset, lset = set(gained), set(lost)
    for bucket in ["2", "3", "4", "5", "6+"]:
        ids = [s for s in common if base[s]["call_bucket"] == bucket]
        g = sum(1 for s in ids if s in gset)
        l = sum(1 for s in ids if s in lset)
        print(f"{bucket}: gained={g} lost={l} net={g-l} (n={len(ids)})")

    # Hard residual failures on step200
    hard = [s for s in common if not new[s]["win"]]
    print(f"\n=== Still failing on step200: {len(hard)} / {len(common)} ===")

    tag_c = Counter()
    stop_c = Counter()
    mismatch_c = Counter()
    pred_calls_c = Counter()
    for s in hard:
        r = new[s]
        for t in r["tags"]:
            tag_c[t] += 1
        stop_c[str(r.get("stop_reason"))] += 1
        mismatch_c[str(r.get("mismatch_reason"))] += 1
        pred_calls_c[r["call_bucket"] + f"|pred={r['num_pred_calls']}"] += 1

    print("top failure tags:")
    for k, v in tag_c.most_common(25):
        print(f"  {k}: {v} ({100*v/len(hard):.1f}%)")
    print("stop_reason:")
    for k, v in stop_c.most_common(15):
        print(f"  {k}: {v}")
    print("mismatch_reason:")
    for k, v in mismatch_c.most_common(15):
        print(f"  {k}: {v}")

    # Behavioral contrast gained vs lost
    def behavior(sids: List[str], arm: Dict[str, Dict[str, Any]], label: str) -> None:
        if not sids:
            return
        avg = lambda key: sum(arm[s][key] for s in sids) / len(sids)
        print(
            f"{label}: n={len(sids)} avg_gold={avg('num_gold_calls'):.2f} "
            f"avg_pred={avg('num_pred_calls'):.2f} f1_func={avg('f1_func'):.3f} "
            f"f1_param={avg('f1_param'):.3f} partial={avg('partial_seq'):.3f}"
        )
        tc = Counter()
        for s in sids:
            for t in arm[s]["tags"]:
                tc[t] += 1
        print("  tags:", ", ".join(f"{k}:{v}" for k, v in tc.most_common(12)))

    print("\n=== Gained tasks (baseline fail -> step200 win) ===")
    behavior(gained, new, "gained@step200")
    behavior(gained, base, "gained@baseline")
    print("\n=== Lost tasks (baseline win -> step200 fail) ===")
    behavior(lost, new, "lost@step200")
    behavior(lost, base, "lost@baseline")
    print("\n=== Persistent fails ===")
    behavior(both_lose, new, "both_lose@step200")

    # Call-count error pattern among hard fails
    print("\n=== Hard fails: gold vs pred call counts ===")
    gp = Counter()
    for s in hard:
        g = new[s]["num_gold_calls"]
        p = new[s]["num_pred_calls"]
        rel = "eq" if p == g else ("under" if p < g else "over")
        bucket = new[s]["call_bucket"]
        gp[(bucket, rel)] += 1
    for bucket in ["2", "3", "4", "5", "6+"]:
        parts = []
        for rel in ["under", "eq", "over"]:
            parts.append(f"{rel}={gp[(bucket, rel)]}")
        print(f"  {bucket}: " + ", ".join(parts))

    # Export CSVs
    def write_csv(name: str, sids: List[str]) -> None:
        path = OUT / name
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "task_id",
                    "call_bucket",
                    "num_gold_calls",
                    "baseline_win",
                    "step200_win",
                    "baseline_pred_calls",
                    "step200_pred_calls",
                    "baseline_f1_func",
                    "step200_f1_func",
                    "baseline_f1_param",
                    "step200_f1_param",
                    "step200_partial_seq",
                    "step200_full_seq",
                    "step200_strict_gold",
                    "step200_sol_eq",
                    "step200_final_answer_pass",
                    "step200_stop_reason",
                    "step200_mismatch_reason",
                    "step200_tags",
                ],
            )
            w.writeheader()
            for s in sids:
                b, n = base[s], new[s]
                w.writerow(
                    {
                        "sample_id": s,
                        "task_id": n["task_id"],
                        "call_bucket": n["call_bucket"],
                        "num_gold_calls": n["num_gold_calls"],
                        "baseline_win": int(b["win"]),
                        "step200_win": int(n["win"]),
                        "baseline_pred_calls": b["num_pred_calls"],
                        "step200_pred_calls": n["num_pred_calls"],
                        "baseline_f1_func": round(b["f1_func"], 4),
                        "step200_f1_func": round(n["f1_func"], 4),
                        "baseline_f1_param": round(b["f1_param"], 4),
                        "step200_f1_param": round(n["f1_param"], 4),
                        "step200_partial_seq": round(n["partial_seq"], 4),
                        "step200_full_seq": round(n["full_seq"], 4),
                        "step200_strict_gold": int(n["strict_gold"]),
                        "step200_sol_eq": int(n["sol_eq"]),
                        "step200_final_answer_pass": int(n["final_answer_pass"]),
                        "step200_stop_reason": n.get("stop_reason"),
                        "step200_mismatch_reason": n.get("mismatch_reason"),
                        "step200_tags": "|".join(n["tags"]),
                    }
                )
        print("wrote", path)

    write_csv("gained.csv", gained)
    write_csv("lost.csv", lost)
    write_csv("persistent_fail.csv", both_lose)
    write_csv("all_paired.csv", common)

    # Markdown summary
    md = OUT / "GAINED_LOST_REPORT.md"
    lines = [
        "# P43 step200 vs baseline — gained/lost analysis",
        "",
        f"- Common tasks: **{len(common)}**",
        f"- Baseline win: **{rate(b_wins,len(common)):.4f}** ({b_wins})",
        f"- Step200 win: **{rate(n_wins,len(common)):.4f}** ({n_wins})",
        f"- Gained: **{len(gained)}** | Lost: **{len(lost)}** | Net: **{len(gained)-len(lost)}**",
        f"- Persistent fails: **{len(both_lose)}**",
        "",
        "## Win delta by gold call-count",
        "",
        "| Bucket | n | d wins | d pp |",
        "| --- | ---: | ---: | ---: |",
    ]
    for b, n, dw, dpp in bucket_table(common, "all"):
        lines.append(f"| {b} | {n} | {dw:+d} | {100*dpp:+.1f} |")
    lines += [
        "",
        "## Gained / lost by call-count",
        "",
        "| Bucket | Gained | Lost | Net |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in ["2", "3", "4", "5", "6+"]:
        ids = [s for s in common if base[s]["call_bucket"] == bucket]
        g = sum(1 for s in ids if s in gset)
        l = sum(1 for s in ids if s in lset)
        lines.append(f"| {bucket} | {g} | {l} | {g-l:+d} |")

    lines += [
        "",
        "## What step200 still fails at",
        "",
        f"Residual failures: **{len(hard)}** ({100*rate(len(hard),len(common)):.1f}%).",
        "",
        "Top tags:",
        "",
    ]
    for k, v in tag_c.most_common(20):
        lines.append(f"- `{k}`: {v} ({100*v/len(hard):.1f}%)")
    lines += [
        "",
        "Gold vs predicted calls on residual fails:",
        "",
    ]
    for bucket in ["2", "3", "4", "5", "6+"]:
        parts = [f"{rel}={gp[(bucket, rel)]}" for rel in ["under", "eq", "over"]]
        lines.append(f"- **{bucket}**: " + ", ".join(parts))
    lines += [
        "",
        "## Files",
        "",
        "- `gained.csv` — baseline fail -> step200 win",
        "- `lost.csv` — baseline win -> step200 fail",
        "- `persistent_fail.csv` — fail on both",
        "- `all_paired.csv` — full paired table",
        "",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", md)


if __name__ == "__main__":
    main()
