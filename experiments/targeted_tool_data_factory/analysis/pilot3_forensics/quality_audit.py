"""Train data quality / anti-shortcut audit (no generation)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from .io import short_hash
from .surface_features import normalize_tool_name


def _skeleton(question: str) -> str:
    q = question.lower()
    q = re.sub(r"-?\d+(?:\.\d+)?", "#", q)
    q = re.sub(r"\$var_?\d+", "$var", q)
    q = re.sub(r"[A-Za-z]+_\d+", "NAME", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _ngrams(s: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def audit_train_quality(rows: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    flags_rows = []
    skeletons = Counter()
    tool_combos = Counter()
    prog_hashes = Counter()
    gold_positions = []

    for row in rows:
        q = str(row.get("question") or "")
        calls = row.get("gold_calls") or []
        tools = row.get("tools") or []
        ans = row.get("gold_answer")
        sk = _skeleton(q)
        skeletons[sk] += 1
        names = tuple(str(c.get("name") or "") for c in calls)
        tool_combos[names] += 1
        # program family: topology-ish + tool names
        prog = short_hash({"tools": names, "n": len(calls)})
        prog_hashes[prog] += 1

        leak = False
        if ans is not None and str(ans) and str(ans) in q:
            leak = True
        # constants in order
        consts = []
        for c in calls:
            args = c.get("arguments") or {}
            if isinstance(args, dict):
                for v in args.values():
                    if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v or "")):
                        consts.append(str(v))
        const_order = " ".join(consts)
        const_in_q = bool(consts) and all(c in q for c in consts[: min(3, len(consts))])

        offered_names = [str(t.get("name") or "") for t in tools if isinstance(t, dict)]
        positions = []
        for n in names:
            if n in offered_names:
                positions.append(offered_names.index(n))
        if positions:
            gold_positions.append(sum(positions) / len(positions))

        explicit_ops = len(re.findall(r"\b(sum|average|multiply|divide|convert|concatenate|filter)\b", q.lower()))

        flags_rows.append({
            "sample_id": str(row.get("sample_id")),
            "answer_leak_in_question": int(leak),
            "constants_mentioned_in_question": int(const_in_q),
            "explicit_op_word_count": explicit_ops,
            "question_skeleton": sk[:200],
            "tool_combo": "|".join(names),
            "program_hash": prog,
            "n_offered": len(offered_names),
            "mean_gold_offered_position": (sum(positions) / len(positions)) if positions else None,
            "generation_cell": (row.get("provenance") or {}).get("generation_cell_id") if isinstance(row.get("provenance"), dict) else "",
        })

    # near duplicates via skeleton
    template_rows = []
    for sk, cnt in skeletons.most_common(50):
        template_rows.append({
            "skeleton": sk[:300],
            "count": cnt,
            "share": cnt / max(1, len(rows)),
        })

    # offered position correlation proxy
    mean_pos = sum(gold_positions) / len(gold_positions) if gold_positions else None

    summary = {
        "n": len(rows),
        "n_unique_skeletons": len(skeletons),
        "top1_skeleton_share": skeletons.most_common(1)[0][1] / max(1, len(rows)) if skeletons else 0.0,
        "n_unique_tool_combos": len(tool_combos),
        "top1_tool_combo_share": tool_combos.most_common(1)[0][1] / max(1, len(rows)) if tool_combos else 0.0,
        "n_unique_program_hashes": len(prog_hashes),
        "answer_leak_rate": sum(r["answer_leak_in_question"] for r in flags_rows) / max(1, len(rows)),
        "constants_in_question_rate": sum(r["constants_mentioned_in_question"] for r in flags_rows) / max(1, len(rows)),
        "mean_gold_tool_offered_position": mean_pos,
        "note": "Lexical/template proxies only; not proof of reward hacking.",
    }

    md = [
        "# ANTI_SHORTCUT_AUDIT",
        "",
        f"- n={summary['n']}",
        f"- top1 skeleton share={summary['top1_skeleton_share']:.3f}",
        f"- top1 tool combo share={summary['top1_tool_combo_share']:.3f}",
        f"- answer leak rate={summary['answer_leak_rate']:.3f}",
        f"- constants-in-question rate={summary['constants_in_question_rate']:.3f}",
        f"- mean gold tool offered position={summary['mean_gold_tool_offered_position']}",
        "",
        "Flags are proxies. Do not treat correlation as causal evidence of shortcut use at train time.",
        "",
    ]
    return flags_rows, template_rows, summary, "\n".join(md)
