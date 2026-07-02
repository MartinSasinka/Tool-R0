"""Compute strict_gold_trace_pass / final_answer_pass / zero_tool_calls
from old curriculum prediction JSONL files, so we can compare apples-to-apples
with the new MT-GRPO run.
"""
import json, sys


def norm_val(v):
    try:
        return float(v)
    except Exception:
        return str(v).lower().strip()


def calls_match_strict(pred, gold):
    """Name + argument match.
    Gold args that are var-refs ($var_N...) are skipped because the old executor
    substitutes them at runtime – we cannot compare substituted vs symbolic values.
    """
    if len(pred) != len(gold):
        return False
    for p, g in zip(pred, gold):
        if p.get("name") != g.get("name"):
            return False
        pa = p.get("arguments", {})
        ga = g.get("arguments", {})
        if set(pa.keys()) != set(ga.keys()):
            return False
        for k in pa:
            gv = str(ga.get(k, ""))
            if gv.startswith("$var"):
                continue  # runtime variable ref – skip
            if norm_val(pa[k]) != norm_val(ga[k]):
                return False
    return True


def calls_match_names(pred, gold):
    """Only function name sequence must match (looser check)."""
    if len(pred) != len(gold):
        return False
    return all(p.get("name") == g.get("name") for p, g in zip(pred, gold))


def compute(path, label, n_filter=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if n_filter is not None:
        rows = [r for r in rows if len(r.get("gold_calls", [])) == n_filter]
    total = len(rows)
    if not total:
        print(f"{label} (n_calls={n_filter}): no rows")
        return

    final = sum(1 for r in rows if r.get("verdict") == "pass")
    zero = sum(
        1
        for r in rows
        if (r.get("num_tool_calls") or len(r.get("predicted_calls", []))) == 0
    )
    strict = sum(
        1
        for r in rows
        if calls_match_strict(r.get("predicted_calls", []), r.get("gold_calls", []))
    )
    name_seq = sum(
        1
        for r in rows
        if calls_match_names(r.get("predicted_calls", []), r.get("gold_calls", []))
    )

    print(f"\n{label}  (n_calls={n_filter}, {total} rows)")
    print(f"  final_answer_pass    : {final/total*100:.2f}%  ({final}/{total})")
    print(f"  zero_tool_calls      : {zero/total*100:.2f}%  ({zero}/{total})")
    print(f"  name_seq_match       : {name_seq/total*100:.2f}%  (same fn-names, same count)")
    print(f"  strict_gold_trace    : {strict/total*100:.2f}%  (names+args, var-refs skipped)")


if __name__ == "__main__":
    base = "curricullum/evaluation/results1/curriculum_baseline_multiturn_predictions.jsonl"
    s3e1 = "curricullum/evaluation/results_v2_20260617/curriculum_stage_3_epoch1_multiturn_predictions.jsonl"
    s5e2 = "curricullum/evaluation/results_v2_20260617/curriculum_stage_5_epoch2_multiturn_predictions.jsonl"

    print("=" * 60)
    print("OLD BASELINE  (results1)")
    print("=" * 60)
    for n in [None, 1, 2, 3]:
        compute(base, "Baseline", n_filter=n)

    print()
    print("=" * 60)
    print("OLD CURRICULUM  Stage3-e1  (results_v2_20260617)")
    print("=" * 60)
    for n in [None, 1, 2, 3]:
        compute(s3e1, "Stage3-e1", n_filter=n)

    print()
    print("=" * 60)
    print("OLD CURRICULUM  Stage5-e2  (results_v2_20260617)")
    print("=" * 60)
    for n in [None, 1, 2, 3]:
        compute(s5e2, "Stage5-e2", n_filter=n)
