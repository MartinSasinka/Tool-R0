import json, os, sys, csv
mini = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nestful_mtgrpo_minimal"))
part = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nestful_mtgrpo_partial"))
sys.path.insert(0, mini)
sys.path.insert(0, part)

from parser import parse_tool_call, parse_tool_calls_all
from rollout import Trajectory, Turn
import reward as strict_mod
import partial_reward as partial_mod
import execution_reward as exec_mod

OUT = os.path.dirname(__file__)

# ---------------- PARSER CASES ----------------
parser_cases = [
    ("valid_single", '<tool_call_answer>[{"name":"add","arguments":{"arg_0":1,"arg_1":2}}]</tool_call_answer>', False),
    ("valid_multi_first_taken", '<tool_call_answer>[{"name":"add","arguments":{"a":1}},{"name":"mul","arguments":{"b":2}}]</tool_call_answer>', False),
    ("invalid_json", '<tool_call_answer>[{"name":}]</tool_call_answer>', False),
    ("missing_name", '<tool_call_answer>[{"arguments":{"arg_0":1}}]</tool_call_answer>', False),
    ("missing_args", '<tool_call_answer>[{"name":"add"}]</tool_call_answer>', False),
    ("no_tag", 'I will add 1 and 2 to get 3.', False),
    ("terminal_empty", '<tool_call_answer>[]</tool_call_answer>', False),
    ("multiple_tags_strict", '<tool_call_answer>[{"name":"a","arguments":{}}]</tool_call_answer><tool_call_answer>[{"name":"b","arguments":{}}]</tool_call_answer>', False),
    ("mangled_close_lenient", '<tool_call_answer>[{"name":"add","arguments":{"arg_0":1}}]</tool_call_call>', True),
    ("bare_array_lenient", 'Here: [{"name":"add","arguments":{"arg_0":1}}]', True),
]
prows = []
for name, text, lenient in parser_cases:
    pr = parse_tool_call(text, lenient=lenient)
    rt = ""
    if pr.ok and pr.call is not None:
        # round-trip: serialize then re-parse
        ser = '<tool_call_answer>[' + json.dumps(pr.call) + ']</tool_call_answer>'
        pr2 = parse_tool_call(ser, lenient=False)
        rt = "ok" if (pr2.ok and pr2.call and pr2.call.get("name") == pr.call.get("name")) else "FAIL"
    prows.append({"case": name, "lenient": lenient, "ok": pr.ok, "is_terminal": pr.is_terminal,
                  "reason": pr.reason, "name": (pr.call or {}).get("name") if pr.call else None,
                  "round_trip": rt})
    print(f"PARSER {name:24s} lenient={lenient} ok={pr.ok} term={pr.is_terminal} reason={pr.reason} rt={rt}")

with open(os.path.join(OUT, "parser_executor_audit_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
    w.writeheader()
    w.writerows(prows)

# ---------------- REWARD CASES ----------------
GOLD = [
    {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
    {"name": "multiply", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
]
GOLD_ANSWER = 9

def mk_task(gold=GOLD, ans=GOLD_ANSWER):
    return {"task_id": "t", "gold_calls": gold, "gold_answer": ans, "num_calls": len(gold), "tools": []}

def mk_traj(turns_spec, final_obs, stop_reason, clipped=False):
    """turns_spec: list of (parsed_call_or_None, fail_reason_or_None, observation)."""
    tr = Trajectory(task_id="t", stage=2, gold_num_turns=2, executor_mode="full")
    tr.clipped_any = clipped
    tr.stop_reason = stop_reason
    tr.final_observation = final_obs
    for i, (call, fail, obs) in enumerate(turns_spec):
        t = Turn(turn_idx=i, model_text="")
        t.parsed_call = call
        t.fail_reason = fail
        t.observation = obs
        if call is None and fail is None and obs is None:
            t.is_terminal = True
        tr.turns.append(t)
    return tr

# define cases: (name, traj, expectations note)
cases = []
# 1 perfect gold trace
cases.append(("perfect_gold_trace", mk_traj(
    [(GOLD[0], None, 3), (GOLD[1], None, 9)], 9, "max_turns")))
# 2 correct final via alternative path (different tool names)
cases.append(("correct_answer_alt_path", mk_traj(
    [({"name": "sum", "arguments": {"x": 1, "y": 2}, "label": "$var1"}, None, 3),
     ({"name": "prod", "arguments": {"x": "$var1.result$", "y": 3}, "label": "$var2"}, None, 9)], 9, "max_turns")))
# 3 no_tool_call (terminal first)
cases.append(("no_tool_call", mk_traj([(None, None, None)], None, "terminal")))
# 4 too_few_calls + wrong answer (1 of 2 calls, wrong final)
cases.append(("too_few_calls_wrong", mk_traj(
    [(GOLD[0], None, 3)], 3, "terminal")))
# 5 valid executable trajectory + wrong answer
cases.append(("executable_wrong_answer", mk_traj(
    [(GOLD[0], None, 3), (GOLD[1], None, 99)], 99, "max_turns")))
# 6 invalid reference (ref to non-existent var)
cases.append(("invalid_reference", mk_traj(
    [(GOLD[0], None, 3),
     ({"name": "multiply", "arguments": {"arg_0": "$var9.result$", "arg_1": 3}, "label": "$var2"}, None, 9)], 9, "max_turns")))
# 7 wrong tool name
cases.append(("wrong_tool_name", mk_traj(
    [({"name": "subtract", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"}, None, 3), (GOLD[1], None, 9)], 9, "max_turns")))
# 8 wrong argument keys
cases.append(("wrong_arg_keys", mk_traj(
    [({"name": "add", "arguments": {"x": 1, "y": 2}, "label": "$var1"}, None, 3), (GOLD[1], None, 9)], 9, "max_turns")))
# 9 extra calls (3 vs gold 2)
cases.append(("extra_calls", mk_traj(
    [(GOLD[0], None, 3), (GOLD[1], None, 9), ({"name": "noop", "arguments": {}, "label": "$var3"}, None, 9)], 9, "max_turns")))
# 10 missing calls (1 correct of 2, but ends)
cases.append(("missing_calls", mk_traj(
    [(GOLD[0], None, 3)], 3, "max_turns")))
# 11 parse error
cases.append(("parse_error", mk_traj(
    [(GOLD[0], None, 3), (None, "parse:invalid_json", None)], 3, "parse_fail")))
# 12 clipped rollout
cases.append(("clipped_rollout", mk_traj(
    [(GOLD[0], None, 3), (None, "clipped_completion", None)], None, "clipped", clipped=True)))
# 13 executor error (call errored)
cases.append(("executor_error", mk_traj(
    [(GOLD[0], None, 3), (GOLD[1], "exec:boom", None)], 3, "executor_error")))

task = mk_task()
rrows = []
for name, tr in cases:
    s = strict_mod.strict_gold_trace_reward(tr, task, None).reward
    p = partial_mod.partial_gold_trace_reward(tr, task, None).reward
    er = exec_mod.execution_aware_reward(tr, task, None)
    e = er.reward
    cap = er.diagnostics.get("cap_applied")
    rrows.append({"case": name, "strict": round(s, 3), "partial": round(p, 3),
                  "execution": round(e, 3), "exec_cap": cap})
    print(f"REWARD {name:26s} strict={s:.2f} partial={p:.2f} exec={e:.2f} cap={cap}")

with open(os.path.join(OUT, "reward_audit_cases.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rrows[0].keys()))
    w.writeheader()
    w.writerows(rrows)
print("WROTE reward_audit_cases.csv and parser_executor_audit_summary.csv")
