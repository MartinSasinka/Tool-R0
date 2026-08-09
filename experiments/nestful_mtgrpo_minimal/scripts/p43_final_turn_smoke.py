#!/usr/bin/env python3
"""Final-turn opportunity smoke (mock stack by default; --gpu for real vLLM)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "targeted_tool_data_factory" / "outputs" / "pilot4_3_nestful_profile_1000"
sys.path.insert(0, str(ROOT))

from interaction_loop import derive_interaction_budget, run_tool_agent_loop  # noqa: E402
from rollout import Trajectory  # noqa: E402


class _FakeExec:
    mode = "synthetic"

    def __init__(self):
        self.n = 0

    def execute(self, call):
        self.n += 1
        from types import SimpleNamespace
        return SimpleNamespace(observation=f"obs_{self.n}", error=None)


def mock_smoke(n_prompts: int = 10):
    # Stratified gold lengths
    lengths = [2, 2, 2, 3, 3, 5, 5, 6, 8, 10][:n_prompts]
    rows = []
    for i, n in enumerate(lengths):
        scripts = [
            f'<tool_call_answer>[{{"name":"tool_{j}","arguments":{{"x":{j}}}}}]'
            f'</tool_call_answer>'
            for j in range(1, n + 1)
        ] + ['<tool_call_answer>[]</tool_call_answer>']
        state = {"i": 0}

        def gen(messages, max_new, _s=state, _sc=scripts):
            t = _sc[_s["i"]] if _s["i"] < len(_sc) else '<tool_call_answer>[]</tool_call_answer>'
            _s["i"] += 1
            return {"text": t, "clipped": False, "prompt_overflow": False,
                    "prompt_tokens": 1, "completion_tokens": 1}

        cfg = {"interaction": {"reserve_final_answer_turn": True},
               "train": {"max_extra_turns_train": 0}}
        task = {"task_id": f"p{i}", "num_calls": n, "gold_calls": [{}] * n,
                "gold_answer": "x", "question": "q", "tools": []}
        b = derive_interaction_budget(n, cfg, mode="train")
        traj = Trajectory(task["task_id"], n, n, executor_mode="synthetic")
        meta = run_tool_agent_loop(
            task=task, config=cfg, executor=_FakeExec(), traj=traj,
            history=[], generate_fn=gen, max_new_tokens=32, budget=b,
        )
        rows.append({**meta.as_dict(), "gold_calls": n, "task_id": task["task_id"]})

    n = len(rows)
    completed_tools = [r for r in rows if r["tool_calls_executed"] >= r["tool_budget"]]
    payload = {
        "mode": "mock_interaction_loop",
        "n_rollouts": n,
        "n_with_tool_calls": sum(1 for r in rows if r["tool_calls_executed"] > 0),
        "n_reaching_tool_budget": len(completed_tools),
        "n_final_response_turn_attempted": sum(
            1 for r in rows if r["final_response_turn_attempted"] or r["final_answer_present"]),
        "n_final_answer_present": sum(1 for r in rows if r["final_answer_present"]),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "fraction_with_final_response_opportunity": (
            sum(1 for r in completed_tools
                if r["final_response_turn_attempted"] or r["final_answer_present"])
            / max(len(completed_tools), 1)
        ),
        "rows": rows,
        "note": "Mock stack validates interaction invariant. Re-run with --gpu on pod "
                "for real vLLM DP parity.",
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    if args.gpu:
        print("GPU smoke not implemented in this environment helper; "
              "use training DP pool on pod with CANARY_TRAJ_LOG=1", file=sys.stderr)
        sys.exit(3)
    payload = mock_smoke(args.n_prompts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / "P43_FINAL_TURN_SMOKE.json"
    mp = args.out_dir / "P43_FINAL_TURN_SMOKE.md"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mp.write_text(
        "# P43 Final Turn Smoke\n\n"
        f"- mode: `{payload['mode']}`\n"
        f"- n_rollouts: {payload['n_rollouts']}\n"
        f"- fraction_with_final_response_opportunity: "
        f"**{payload['fraction_with_final_response_opportunity']:.0%}**\n"
        f"- stop_reasons: `{payload['stop_reasons']}`\n"
        f"- n_final_answer_present: {payload['n_final_answer_present']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": [str(jp), str(mp)],
                      "fraction": payload["fraction_with_final_response_opportunity"]},
                     indent=2))
    if payload["fraction_with_final_response_opportunity"] < 1.0 - 1e-9:
        sys.exit(2)


if __name__ == "__main__":
    main()
