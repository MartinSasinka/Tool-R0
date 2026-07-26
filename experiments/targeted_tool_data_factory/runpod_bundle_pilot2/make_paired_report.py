#!/usr/bin/env python3
"""Paired report for D0 vs D1 or C0 vs D1.

Reads the per-task trajectories written by the trainer eval path and produces a
markdown report (+ optional JSON) with, for every eval set:

  Win Rate, Function F1, Parameter F1, executability, gained/lost task lists,
  a paired bootstrap 95% CI on the win-rate delta, an exact McNemar test on the
  discordant pairs, and a failure taxonomy.

Everything is PAIRED on sample_id: only tasks present in both compared arms are
scored, so the delta is a within-task contrast, not two independent means. The
held-out report is additionally broken down by track (G = generalization,
A = adaptation).

`--baseline` / `--treatment` select the contrast:
  default D0 vs D1 (full experiment)
  C0 vs D1         (--c0-vs-d1 mode; no D0 checkpoint required)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260726


# ───────────────────────────────────────────────────────────── loading ────

def load_rows(out_dir: Path) -> dict[str, dict]:
    path = out_dir / "final_eval_trajectories.jsonl"
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[str(r.get("sample_id"))] = r
    return rows


def _traj(row: dict) -> dict:
    return row.get("_traj") or {}


def win(row: dict):
    v = _traj(row).get("official_win")
    if v is not None:
        return float(bool(v))
    v = row.get("internal_win_rate")
    return None if v is None else float(v)


def metric(row: dict, official: str, internal: str):
    v = _traj(row).get(official)
    if v is None:
        v = row.get(internal)
    return None if v is None else float(v)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f} %" if pct else f"{v:.3f}"


# ─────────────────────────────────────────────────────────── statistics ────

def paired_bootstrap(a: list[float], b: list[float]):
    """95% CI on mean(b) - mean(a) resampling TASKS (paired)."""
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = [b[i] - a[i] for i in range(n)]
    boots = []
    for _ in range(BOOTSTRAP_ITERS):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return (boots[int(0.025 * BOOTSTRAP_ITERS)], boots[int(0.975 * BOOTSTRAP_ITERS) - 1])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (baseline only)
    and c (treatment only). Under H0 each discordant pair is a fair coin flip."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def failure_class(row: dict) -> str:
    t = _traj(row)
    for key in ("failure_class", "failure_taxonomy", "error_class"):
        v = t.get(key) or row.get(key)
        if v:
            return str(v)
    if win(row) == 1.0:
        return "success"
    if (t.get("executability") or row.get("internal_executability") or 0) == 0:
        return "not_executable"
    pred = t.get("num_predicted_calls")
    gold = t.get("num_gold_calls")
    if isinstance(pred, int) and isinstance(gold, int):
        if pred < gold:
            return "under_calling"
        if pred > gold:
            return "over_calling"
    return "wrong_answer"


# ────────────────────────────────────────────────────────────── report ────

METRICS: list[tuple[str, Callable[[dict], Any]]] = [
    ("Win Rate", lambda r: win(r)),
    ("Function F1", lambda r: metric(r, "official_function_f1", "internal_function_f1")),
    ("Parameter F1", lambda r: metric(r, "official_parameter_f1", "internal_parameter_f1")),
    ("Executability", lambda r: metric(r, "executability", "internal_executability")),
]


def section(title: str, arms: dict[str, dict[str, dict]], meta: dict[str, dict],
            baseline: str, treatment: str) -> tuple[list[str], dict]:
    """arms: {arm -> {sample_id -> row}}; meta: {sample_id -> {"track": ...}}"""
    out = [f"## {title}", ""]
    payload: dict[str, Any] = {"title": title, "complete": False}
    present = {a: r for a, r in arms.items() if r}
    if baseline not in present or treatment not in present:
        msg = (f"_missing {baseline} or {treatment} trajectories — "
               "evaluation incomplete._")
        return out + [msg, ""], payload

    # Pair only on the two contrasted arms so a missing optional third arm
    # (e.g. D0 in C0-vs-D1 mode) cannot empty the intersection.
    ids = sorted(set(present[baseline]) & set(present[treatment]))
    extras = [a for a in present if a not in (baseline, treatment)]
    out.append(
        f"Paired on **{len(ids)}** tasks present in `{baseline}` and "
        f"`{treatment}` "
        f"({', '.join(f'{a}: {len(r)}' for a, r in present.items())} rows loaded).")
    out.append("")

    delta_label = f"{treatment} - {baseline}"
    col_arms = [baseline, treatment] + extras
    out += ["| Metric | " + " | ".join(col_arms) + f" | {delta_label} |",
            "|---|" + "---|" * (len(col_arms) + 1)]
    metric_vals: dict[str, dict[str, Optional[float]]] = {}
    for name, fn in METRICS:
        vals = {a: mean([fn(present[a][i]) for i in ids]) for a in col_arms}
        d = (None if vals.get(treatment) is None or vals.get(baseline) is None
             else vals[treatment] - vals[baseline])
        metric_vals[name] = {**vals, "delta": d}
        out.append(f"| {name} | " + " | ".join(fmt(vals[a]) for a in col_arms) +
                   f" | {'n/a' if d is None else f'{100 * d:+.1f} pp'} |")
    out.append("")

    w_base = [win(present[baseline][i]) or 0.0 for i in ids]
    w_treat = [win(present[treatment][i]) or 0.0 for i in ids]
    gained = [i for i, a, b in zip(ids, w_base, w_treat) if b > a]
    lost = [i for i, a, b in zip(ids, w_base, w_treat) if b < a]
    ci = paired_bootstrap(w_base, w_treat)
    p = mcnemar_exact(len(lost), len(gained))

    out += [f"- gained ({treatment} wins, {baseline} loses): **{len(gained)}**",
            f"- lost ({baseline} wins, {treatment} loses): **{len(lost)}**",
            f"- paired bootstrap 95% CI on Win Rate delta: "
            f"{'n/a' if ci is None else f'[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp'}",
            f"- exact McNemar on {len(gained) + len(lost)} discordant pairs: p = {p:.4f}"
            f" ({'significant' if p < 0.05 else 'not significant'} at alpha=0.05)",
            ""]

    track_rows = []
    tracks = defaultdict(list)
    for i in ids:
        t = (meta.get(i) or {}).get("track")
        if t:
            tracks[t].append(i)
    if tracks:
        out += ["### By track", "",
                f"| Track | n | {baseline} Win | {treatment} Win | delta |",
                "|---|---|---|---|---|"]
        for t, tid in sorted(tracks.items()):
            a = mean([win(present[baseline][i]) for i in tid])
            b = mean([win(present[treatment][i]) for i in tid])
            d = None if a is None or b is None else b - a
            out.append(f"| {t} | {len(tid)} | {fmt(a)} | {fmt(b)} | "
                       f"{'n/a' if d is None else f'{100 * d:+.1f} pp'} |")
            track_rows.append({"track": t, "n": len(tid),
                               f"{baseline}_win": a, f"{treatment}_win": b,
                               "delta": d})
        out.append("")

    out += ["### Failure taxonomy", "",
            "| Class | " + " | ".join(col_arms) + " |",
            "|---|" + "---|" * len(col_arms)]
    counts = {a: Counter(failure_class(present[a][i]) for i in ids) for a in col_arms}
    classes = sorted({c for cc in counts.values() for c in cc})
    for cls in classes:
        out.append(f"| {cls} | " + " | ".join(str(counts[a][cls]) for a in col_arms) + " |")
    out += ["", "<details><summary>gained / lost task ids</summary>", "",
            f"gained: `{', '.join(gained) if gained else '-'}`", "",
            f"lost: `{', '.join(lost) if lost else '-'}`", "", "</details>", ""]

    payload = {
        "title": title,
        "complete": True,
        "n_paired": len(ids),
        "arms_loaded": {a: len(r) for a, r in present.items()},
        "metrics": {k: v for k, v in metric_vals.items()},
        "gained": gained,
        "lost": lost,
        "n_gained": len(gained),
        "n_lost": len(lost),
        "bootstrap_ci_95": None if ci is None else {"lo": ci[0], "hi": ci[1]},
        "mcnemar_p": p,
        "by_track": track_rows,
        "failure_taxonomy": {a: dict(counts[a]) for a in col_arms},
    }
    return out, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="optional machine-readable twin of the markdown report")
    ap.add_argument("--baseline", default="D0",
                    help="baseline arm for the paired contrast (default D0)")
    ap.add_argument("--treatment", default="D1",
                    help="treatment arm for the paired contrast (default D1)")
    ap.add_argument("--title", default=None,
                    help="report title override (default: '<baseline> vs <treatment>')")
    ap.add_argument("--heldout-meta", type=Path, default=None,
                    help="canonical pilot2 held-out JSONL, used for the track split")
    args = ap.parse_args()

    baseline, treatment = args.baseline, args.treatment
    title = args.title or f"{baseline} vs {treatment}"
    # Load every arm that has trajectories on disk; the contrast itself only
    # requires baseline + treatment. Extra arms (e.g. D0 leftover dirs) are
    # shown as additional columns when present.
    all_arms = ("C0", "D0", "D1")

    meta: dict[str, dict] = {}
    default_meta = Path(__file__).resolve().parent / "data" / "heldout_canonical_pilot2.jsonl"
    for path in (args.heldout_meta, default_meta):
        if path and Path(path).is_file():
            with Path(path).open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        meta[str(r.get("task_id") or r.get("sample_id"))] = {
                            "track": r.get("track"),
                            "motif": r.get("motif"),
                            "answer_kind": r.get("answer_kind"),
                        }
            break

    if baseline == "C0" and treatment == "D1":
        blurb = (
            "`C0` = base checkpoint (no adapter). `D1` = 160 pilot2 factory "
            "tasks trained with A4_GATED_VERIFIABLE, 8 rollouts, GPU0 learner / "
            "GPU1-3 rollout workers. D0 training was skipped (`--c0-vs-d1`)."
        )
    else:
        blurb = (
            "`D0` = 160 old Stage-3 tasks. `D1` = 160 pilot2 factory tasks. "
            "Same C0, seed, reward (A4_GATED_VERIFIABLE), LR/KL/LoRA/optimizer/"
            "credit assignment/decoding, 8 rollouts, same optimizer-step budget. "
            "The training dataset is the only manipulated variable."
        )

    lines = [f"# {title} — paired report", "", blurb, ""]
    sections_json: list[dict] = []

    for sect_title, sub in (("Structural held-out (pilot2, 80 tasks)", "heldout80"),
                            ("NESTFUL diagnostic-500 (frozen)", "nestful500")):
        arms = {a: load_rows(args.results / "eval" / f"{a}_{sub}") for a in all_arms}
        md, payload = section(sect_title, arms,
                              meta if sub == "heldout80" else {},
                              baseline, treatment)
        lines += md
        sections_json.append({"eval_set": sub, **payload})

    lines += ["## Reading this report", "",
              "The held-out section measures whether the model learned the *data* "
              "(in-domain). The NESTFUL section measures whether that learning "
              "*transferred*. A treatment gain on held-out with no NESTFUL gain is "
              "an in-domain-learning result, not a transfer result, and must be "
              "reported as such.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {args.out}")

    json_path = args.json_out or args.out.with_suffix(".json")
    report = {
        "title": title,
        "baseline": baseline,
        "treatment": treatment,
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sections": sections_json,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] wrote {json_path}")
    print("\n".join(lines)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
