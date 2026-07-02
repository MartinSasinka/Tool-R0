import hashlib, json, os, datetime
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mini = os.path.join(root, "experiments", "nestful_mtgrpo_minimal")
def md5(p):
    if not os.path.exists(p): return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()
def sz(p):
    return os.path.getsize(p) if os.path.exists(p) else None

files = {
    "nestful_eval": os.path.join(mini, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"),
    "scorer": os.path.join(mini, "data", "NESTFUL-main", "src", "scorer.py"),
    "reward_strict": os.path.join(mini, "reward.py"),
    "reward_partial": os.path.join(root, "experiments", "nestful_mtgrpo_partial", "partial_reward.py"),
    "reward_execution": os.path.join(root, "experiments", "nestful_mtgrpo_partial", "execution_reward.py"),
    "parser": os.path.join(mini, "parser.py"),
    "executor": os.path.join(mini, "executor.py"),
    "rollout": os.path.join(mini, "rollout.py"),
    "prompt": os.path.join(mini, "prompt.py"),
    "grpo_train": os.path.join(mini, "grpo_train.py"),
}
for n in range(1, 7):
    files[f"train_stage{n}"] = os.path.join(mini, "data", "filtered_toolr0_synthetic", f"epoch_{n}_{n}call.jsonl")

manifest = {
    "audit_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_commit": "fd222c7ee35821fb769719dba4cc49329456bb0c",
    "git_commit_note": "upstream 'initial code added'; experiments/ tree is untracked working state (not in this commit)",
    "model": {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "finetuning": "qlora r=16 alpha=32 dropout=0.05"},
    "datasets": {k: {"path": os.path.relpath(v, root), "md5": md5(v), "bytes": sz(v)} for k, v in files.items() if "stage" in k or k == "nestful_eval"},
    "code_versions_md5": {k: md5(v) for k, v in files.items() if "stage" not in k and k != "nestful_eval"},
    "generation": {"temperature": 0.7, "top_p": 0.95, "num_generations": 4,
                   "max_turns_train": "gold_n", "max_turns_eval": "gold_n + 1 (cap gold_n+4)",
                   "max_new_tokens_train": 2048, "max_new_tokens_eval": 2560, "seed": 42},
    "training": {"learning_rate": 1e-6, "kl_beta": 0.02, "grad_accum": 4, "per_device_bs": 1,
                 "max_grad_norm": 1.0, "epochs_per_stage_max": 4, "mt_grpo_mode": "turn_level_minimal",
                 "gamma": 1.0, "lambda_episode": 1.0, "normalize_advantage": True,
                 "mask_clipped_from_update": True},
    "curriculum": {"train_stages": "1 2 3 4 (synthetic, N-call each, 400 tasks/stage)",
                   "advance_threshold_strict_pass": 0.50, "plateau_patience": 2,
                   "carry_forward_selection": "best strict_gold_trace_pass per stage (NOT Win)"},
    "reward_train_policy": {"minimal": "strict_gold_trace", "partial": "partial_gold_trace",
                            "execution_aware": "available, not yet used in a full run"},
    "eval": {"dataset": "NESTFUL nestful_data.jsonl n=1861", "paradigms": ["react", "direct"],
             "scorer": "official NESTFUL src/scorer.py (Win via IBM executable_functions)",
             "eval_prompt": "SYSTEM_PROMPT + _EVAL_HARDENING", "eval_parser": "lenient",
             "train_prompt": "SYSTEM_PROMPT only", "train_parser": "strict"},
    "checkpoints_final_eval": ["partial_s1_e4", "partial_s2_e2", "partial_s3_e2", "partial_s4_e1",
                               "minimal_s1_e4", "minimal_s2_e4", "minimal_s4e2", "baseline"],
}
out = os.path.join(os.path.dirname(__file__), "audit_manifest.json")
json.dump(manifest, open(out, "w", encoding="utf-8"), indent=2)
print("wrote", out)
print(json.dumps(manifest["datasets"], indent=2))
