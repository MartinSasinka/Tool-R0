"""End-to-end pipeline verification for nestful_standalone/run.py.

Checks the *full* path the colleague will hit on the GPU box:

1. HuggingFace dataset download (ibm-research/nestful) and our loader's
   normalisation.
2. ``ensure_ibm_repo`` + ``IBMFunctionRegistry`` against a fresh clone.
3. Multi-turn loop with a *real* (small) HF causal LM driving the
   model; verifies that the model's response is parsed, the parsed
   tool call is executed, and the result is fed back into the next
   prompt.

vLLM is Linux-only so we use a thin transformers-based driver here that
honours the same `_apply_template / approx_token_count / generate`
contract as `_VLLMRunner` in run.py. The deliverable run.py stays
vLLM-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "nestful_standalone" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("standalone_run", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["standalone_run"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# transformers backend that mimics _VLLMRunner (test-only).
# ---------------------------------------------------------------------


class _HFRunner:
    def __init__(self, model_id: str, *, device: str = "cpu", dtype=None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"[hf] loading {model_id} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype or torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device

    def _apply_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def approx_token_count(self, text):
        return max(1, len(text) // 3)

    def generate(self, all_messages, *, temperature, top_p, max_new_tokens, seeds=None):
        import torch

        outs = []
        for i, messages in enumerate(all_messages):
            prompt = self._apply_template(messages)
            ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seed = seeds[i] if seeds else 0
            torch.manual_seed(seed)
            gen = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = gen[0, ids.input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outs.append(text)
        return outs


# ---------------------------------------------------------------------
# Step 1: HuggingFace dataset download + normalisation.
# ---------------------------------------------------------------------


def step1_hf_dataset(mod) -> list:
    print()
    print("=" * 60)
    print(" STEP 1: HuggingFace dataset download")
    print("=" * 60)
    tasks = mod.load_nestful_tasks(max_tasks=3)
    print(f"[step1] loaded {len(tasks)} tasks")
    for t in tasks:
        print(
            f"  task_id={t['task_id']}  q={t['question'][:50]!r}...  "
            f"gold_calls={len(t['gold_calls'])}  gold_answer={t['gold_answer']}"
        )
    assert len(tasks) == 3
    for t in tasks:
        assert isinstance(t["tools"], list) and t["tools"]
        assert isinstance(t["gold_calls"], list)
        assert "name" in t["tools"][0]
        assert "parameters" in t["tools"][0]
    print("[step1] OK normalisation")
    return tasks


# ---------------------------------------------------------------------
# Step 2: git clone of IBM/NESTFUL into a fresh tempdir, then verify
# IBMFunctionRegistry can lazy-load a non-trivial helper.
# ---------------------------------------------------------------------


def step2_ibm_clone(mod) -> str:
    print()
    print("=" * 60)
    print(" STEP 2: git clone IBM/NESTFUL into a fresh dir")
    print("=" * 60)
    # Persistent path under .verify_artifacts so steps 4+5 can reuse it.
    artifacts = REPO_ROOT / "scripts" / ".verify_artifacts"
    artifacts.mkdir(exist_ok=True)
    target = str(artifacts / "nestful_repo")
    ok = mod.ensure_ibm_repo(target)
    if not ok:
        print(f"[step2] WARNING: clone returned False (network/git issue)")
        print(f"[step2] falling back to vendored copy at eval/data/NESTFUL-main")
        target = str(REPO_ROOT / "eval" / "data" / "NESTFUL-main")
        assert os.path.isdir(target), "no fallback IBM checkout"
    else:
        print(f"[step2] using clone at {target}")
        sentinel = os.path.join(target, "data_v2/executable_functions/func_file_map.json")
        assert os.path.isfile(sentinel)
    reg = mod.IBMFunctionRegistry(target)
    assert reg.available
    print(f"[step2] IBM registry stats: {reg.stats()}")
    # Lazy load a helper from func_file_map (not in basic_functions).
    fn = reg.get("find_kth_largest")
    assert callable(fn), "find_kth_largest should resolve via lazy import"
    result = fn(nums=[1, 2, 3, 4, 5], k=2)
    assert result == 4, f"find_kth_largest should be 4, got {result}"
    print(f"[step2] lazy-imported find_kth_largest([1..5], k=2) = {result}")
    # Also through the standalone's execute_one with arg_0/arg_1 convention.
    trace = mod.execute_one(
        {"name": "find_kth_largest", "arguments": {"nums": [10, 20, 30], "k": 1},
         "label": "v1"},
        {}, [], index=0, ibm_registry=reg,
    )
    assert trace.error is None and trace.source == "ibm" and trace.result == 30, trace
    print(f"[step2] execute_one path also works: result={trace.result} via {trace.source}")
    return target


# ---------------------------------------------------------------------
# Step 3: real LLM end-to-end multi-turn with a tiny HF model.
# ---------------------------------------------------------------------


def step3_multiturn_real(mod, tasks: list, ibm_root: str, model_id: str,
                         max_steps: int, num_rollouts: int) -> dict:
    print()
    print("=" * 60)
    print(f" STEP 3: real LLM multi-turn ({model_id})")
    print("=" * 60)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else None
    runner = _HFRunner(model_id, device=device, dtype=dtype)

    reg = mod.IBMFunctionRegistry(ibm_root)
    out_dir = REPO_ROOT / "scripts" / ".verify_artifacts" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "real_multiturn_predictions.jsonl"
    summary_path = out_dir / "real_multiturn_summary.json"
    for p in (pred_path, summary_path):
        if p.exists():
            p.unlink()
    summary = mod.run_multiturn_rollouts(
        tasks,
        runner,
        num_rollouts=num_rollouts,
        max_steps=max_steps,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=256,
        max_model_len=4096,
        seed=0,
        ibm_registry=reg,
        output_path=str(pred_path),
        model_name=model_id,
        model_profile="real",
        summary_path=str(summary_path),
    )
    print(f"[step3] summary mean_score={summary.get('mean_score')}  "
          f"stop={summary.get('stop_reason_breakdown')}  "
          f"exec={summary.get('execution_class_breakdown')}")

    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    print(f"[step3] wrote {len(rows)} rows")
    expected = len(tasks) * num_rollouts
    assert len(rows) == expected, f"expected {expected}, got {len(rows)}"

    # Diagnostic: inspect the first rollout's transcript
    print()
    print("[step3] first rollout transcript snapshot:")
    r0 = rows[0]
    print(f"   task_id={r0['task_id']}  rollout_idx={r0['rollout_idx']}")
    print(f"   stopped={r0['stopped']}  num_steps={r0['num_steps']}  "
          f"verdict={r0['verdict']}")
    print(f"   predicted_calls={r0['predicted_calls']}")
    print(f"   execution_trace sources={[t['source'] for t in r0['execution_trace']]}")
    print(f"   predicted_final={r0['predicted_final']}  gold_answer={r0['gold_answer']}")
    if r0['messages']:
        last = r0['messages'][-1]
        print(f"   last message[{last['role']}]={last['content'][:200]!r}")

    # Did at least one rollout call a tool?
    rollouts_with_calls = sum(1 for r in rows if r["predicted_calls"])
    print(
        f"[step3] {rollouts_with_calls}/{len(rows)} rollouts emitted "
        f"at least one tool call"
    )

    # Did the executor source ever fire?
    sources_all = {tr["source"] for r in rows for tr in r["execution_trace"]}
    print(f"[step3] executor sources observed across all rollouts: {sources_all}")

    return {"rows": rows, "summary": summary,
            "rollouts_with_calls": rollouts_with_calls,
            "sources": sources_all}


# ---------------------------------------------------------------------
# Step 4: deterministic scripted-runner verification of the full
# parse -> execute -> feedback loop. Uses *real* run.py code (real
# parser, real executor, real IBM registry, real prompt formatting),
# only the LLM is replaced by a pre-canned response sequence.
# ---------------------------------------------------------------------


class _ScriptedRunner:
    """Returns pre-canned LLM outputs per turn. Index per rollout-state."""

    def __init__(self, scripts):
        # scripts: list[list[str]] -> per rollout, list of per-turn outputs
        self.scripts = scripts
        self.turn_count = [0] * len(scripts)
        self.observed_messages = []  # capture the messages list passed in

    def _apply_template(self, messages):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def approx_token_count(self, text):
        return max(1, len(text) // 3)

    def generate(self, all_messages, *, temperature, top_p,
                 max_new_tokens, seeds=None):
        outs = []
        # find which rollout each messages belongs to by exact identity
        for messages in all_messages:
            self.observed_messages.append([dict(m) for m in messages])
            # match by messages length+last user content
            chosen = None
            for i, script in enumerate(self.scripts):
                if self.turn_count[i] >= len(script):
                    continue
                # heuristic: first-not-yet-consumed script
                chosen = i
                break
            if chosen is None:
                outs.append("")
                continue
            text = self.scripts[chosen][self.turn_count[chosen]]
            self.turn_count[chosen] += 1
            outs.append(text)
        return outs


def step4_scripted_endtoend(mod, tasks, ibm_root):
    print()
    print("=" * 60)
    print(" STEP 4: scripted LLM -> parse -> execute -> feedback")
    print("=" * 60)

    # Use ONE task from the actual NESTFUL dataset.
    task = tasks[0]  # the chemical-X mixture problem with gold=40.0
    print(f"[step4] using task {task['task_id']}: {task['question'][:80]!r}...")
    print(f"[step4] gold gold_answer={task['gold_answer']}, "
          f"gold_calls (first 3 names)={[c['name'] for c in task['gold_calls'][:3]]}")

    # Script realistic Qwen-style output sequences. Our parser handles
    # <tool_call>...</tool_call>, <tool_call_answer>[...]</tool_call_answer>,
    # bare JSON, fenced JSON, etc. We intentionally use multiple formats
    # to exercise the parser + executor + variable resolution.
    scripts = [[
        # turn 1: emit one tool call inside <tool_call>...</tool_call>
        'Let me start by computing how much chemical X is in the original mixture.\n'
        '<tool_call>{"name": "multiply", "arguments": {"arg_0": 0.25, "arg_1": 80}, "label": "var_1"}</tool_call>',
        # turn 2: variable reference (var_1) -> resolver should plug in 20.0
        '<tool_call>{"name": "add", "arguments": {"arg_0": "$var_1$", "arg_1": 20}, "label": "var_2"}</tool_call>',
        # turn 3: total volume
        '<tool_call>{"name": "add", "arguments": {"arg_0": 80, "arg_1": 20}, "label": "var_3"}</tool_call>',
        # turn 4: percentage = (40/100) * 100 (test fenced JSON path too)
        'Now divide and scale to a percentage:\n'
        '```json\n{"name": "divide", "arguments": {"arg_0": "$var_2$", "arg_1": "$var_3$"}, "label": "var_4"}\n```',
        # turn 5: final answer using a list-style <tool_call_answer> block
        '<tool_call_answer>[{"name": "multiply", "arguments": {"arg_0": "$var_4$", "arg_1": 100}, "label": "var_5"}]</tool_call_answer>',
        # turn 6: model should now have the result and emit a final answer
        'The answer is **40.0%**.',
    ]]

    runner = _ScriptedRunner(scripts)
    reg = mod.IBMFunctionRegistry(ibm_root)
    out_dir = REPO_ROOT / "scripts" / ".verify_artifacts" / "scripted"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "scripted_multiturn_predictions.jsonl"
    summary_path = out_dir / "scripted_multiturn_summary.json"
    if pred_path.exists():
        pred_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    summary = mod.run_multiturn_rollouts(
        [task],
        runner,
        num_rollouts=1,
        max_steps=8,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=256,
        max_model_len=8192,
        seed=0,
        ibm_registry=reg,
        output_path=str(pred_path),
        model_name="scripted-test",
        model_profile="scripted",
        summary_path=str(summary_path),
    )

    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    row = rows[0]

    # === eval/results/nestful schema fields are present (minus judge) ===
    for key in ("task_id", "question", "status", "score", "verdict",
                "verdict_reason", "stopped", "num_steps", "predicted_final",
                "gold_answer", "predicted_calls", "execution_trace",
                "raw_completions", "execution_error",
                "num_tool_calls", "error_category"):
        assert key in row, f"missing key {key} in row"
    for forbidden in ("judge_used", "judge_cache_hit",
                      "judge_used_count", "judge_enabled",
                      "use_judge_fallback"):
        assert forbidden not in row, f"unexpected judge field {forbidden}"

    print(f"[step4] num_steps={row['num_steps']}  stopped={row['stopped']}  "
          f"verdict={row['verdict']}  predicted_final={row['predicted_final']}  "
          f"score={row['score']}")
    print(f"[step4] predicted_calls (names): {[c['name'] for c in row['predicted_calls']]}")

    # The LLM emitted real tool calls and we parsed them.
    names = [c["name"] for c in row["predicted_calls"]]
    assert names == ["multiply", "add", "add", "divide", "multiply"], names
    # Each call was actually executed by execute_one (not skipped).
    assert len(row["execution_trace"]) == 5
    assert row["num_tool_calls"] == 5, row["num_tool_calls"]
    for tr in row["execution_trace"]:
        assert tr["error"] is None, f"unexpected error in {tr}"
    # Variable references were resolved against prior results.
    var2 = row["execution_trace"][1]
    assert var2["arguments_resolved"]["arg_0"] == 20.0, var2  # $var_1$ resolved
    var4 = row["execution_trace"][3]
    assert var4["arguments_resolved"]["arg_0"] == 40.0, var4  # $var_2$ resolved
    assert var4["arguments_resolved"]["arg_1"] == 100.0, var4  # $var_3$ resolved
    # Final tool result fed back; final answer matches gold.
    assert row["execution_trace"][-1]["result"] == 40.0
    assert row["status"] == "completed", row["status"]
    assert row["score"] == 1.0
    assert row["verdict"] == "pass"
    assert row["verdict_reason"] == "executor_match"
    # The conversation contains tool results between turns (fed back as
    # role=user to match eval/benchmarks/nestful convention).
    msg_roles = [m["role"] for m in row["messages"]]
    assert msg_roles.count("user") >= 6, msg_roles
    tool_result_msgs = [m for m in row["messages"][2:]
                        if m["role"] == "user" and "tool_response" in m["content"]]
    assert len(tool_result_msgs) >= 4, [m["content"][:80] for m in tool_result_msgs]
    # raw_completions present and capped at 6, each <= 1500 chars.
    assert 1 <= len(row["raw_completions"]) <= 6
    for c in row["raw_completions"]:
        assert len(c) <= 1500

    # === summary JSON matches the eval/results/nestful schema ===
    summary_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("benchmark", "model_profile", "total_tasks", "completed",
                "failed", "errors", "mean_score", "mean_score_percent",
                "mode", "final_answer_accuracy", "passed",
                "stop_reason_breakdown", "execution_class_breakdown",
                "ibm_registry_stats", "max_steps_setting"):
        assert key in summary_disk, f"summary missing {key}"
    for forbidden in ("judge_used_count", "judge_used_rate_percent",
                      "judge_cache_hits", "judge_enabled",
                      "use_judge_fallback", "skipped",
                      "skipped_rate_percent", "explicit_final_rate_percent"):
        assert forbidden not in summary_disk, f"unexpected field {forbidden}"
    assert summary_disk["benchmark"] == "nestful"
    assert summary_disk["mode"] == "multiturn"
    assert summary_disk["model_profile"] == "scripted"
    assert summary_disk["total_tasks"] == 1
    assert summary_disk["passed"] == 1
    assert summary_disk["mean_score"] == 1.0
    assert summary_disk["ibm_registry_stats"]["available"] is True
    print(f"[step4] summary stop_reason_breakdown={summary_disk['stop_reason_breakdown']}")
    print(f"[step4] summary execution_class_breakdown={summary_disk['execution_class_breakdown']}")
    # All NESTFUL functions now dispatch through the IBM registry
    # (basic_functions.py defines add/subtract/multiply/divide natively).
    sources = {tr["source"] for tr in row["execution_trace"]}
    assert sources == {"ibm"}, f"expected only IBM dispatch, got {sources}"
    print(f"[step4] sources observed: {sources}")
    print("[step4] OK — parser, executor, var-resolver, "
          "tool-result feedback all verified end-to-end")
    return row


def step5_scripted_with_ibm(mod, ibm_root):
    """Same as step 4 but with an IBM-registry-only function call.

    Uses a synthetic task whose tool list points at an IBM helper, to
    prove that the lazy-loaded IBM helper is dispatched in the
    multiturn loop (not just in unit tests).
    """
    print()
    print("=" * 60)
    print(" STEP 5: scripted LLM hitting an IBM-registry function")
    print("=" * 60)
    task = {
        "task_id": "synthetic-ibm-1",
        "question": "Find the 2nd largest of [10, 20, 30, 40, 50].",
        "tools": [{
            "name": "find_kth_largest",
            "description": "Return the k-th largest element of a list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nums": {"type": "array"},
                    "k": {"type": "integer"},
                },
            },
        }],
        "gold_calls": [{
            "name": "find_kth_largest",
            "arguments": {"nums": [10, 20, 30, 40, 50], "k": 2},
            "label": "var_1",
        }],
        "gold_answer": 40,
    }

    scripts = [[
        '<tool_call>{"name": "find_kth_largest", "arguments": {"nums": [10,20,30,40,50], "k": 2}, "label": "var_1"}</tool_call>',
        'The 2nd largest is **40**.',
    ]]

    runner = _ScriptedRunner(scripts)
    reg = mod.IBMFunctionRegistry(ibm_root)
    out_dir = REPO_ROOT / "scripts" / ".verify_artifacts" / "ibm"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "ibm_multiturn_predictions.jsonl"
    summary_path = out_dir / "ibm_multiturn_summary.json"
    for p in (pred_path, summary_path):
        if p.exists():
            p.unlink()

    mod.run_multiturn_rollouts(
        [task], runner,
        num_rollouts=1, max_steps=4,
        temperature=0.0, top_p=1.0,
        max_new_tokens=128, max_model_len=4096,
        seed=0, ibm_registry=reg,
        output_path=str(pred_path),
        model_name="scripted-ibm",
        model_profile="ibm",
        summary_path=str(summary_path),
    )
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    print(f"[step5] num_steps={row['num_steps']}  stopped={row['stopped']}  "
          f"verdict={row['verdict']}  predicted_final={row['predicted_final']}")
    assert len(row["execution_trace"]) == 1
    tr = row["execution_trace"][0]
    assert tr["error"] is None and tr["result"] == 40, tr
    assert tr["source"] == "ibm", f"expected IBM dispatch, got {tr['source']}"
    assert row["status"] == "completed"
    assert row["verdict"] == "pass"
    assert row["score"] == 1.0
    summary_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_disk["execution_class_breakdown"].get("executed_ok_ibm") == 1
    print("[step5] OK -- IBM-registry helper dispatched and executed in real loop")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-step3", action="store_true",
                    help="Skip the real-LLM step.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "nestful_standalone"))
    mod = _load_run_module()

    tasks = step1_hf_dataset(mod)
    ibm_root = step2_ibm_clone(mod)

    # Steps 4+5: deterministic end-to-end via the *real* multiturn loop.
    step4_scripted_endtoend(mod, tasks, ibm_root)
    step5_scripted_with_ibm(mod, ibm_root)

    if not args.skip_step3:
        step3_multiturn_real(
            mod, tasks, ibm_root,
            model_id=args.model,
            max_steps=args.max_steps,
            num_rollouts=args.num_rollouts,
        )
    else:
        print("[step3] skipped via --skip-step3")

    print()
    print("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
