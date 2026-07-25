"""A01 — Config/parity diff C0 + A0-A4 (from raw config_used.json / manifests / state)."""
from __future__ import annotations

from typing import Any, Dict

from common import ARMS, load_json, run_dir, write_json

# fields that legitimately differ between arms (identity, not treatment)
IDENTITY_KEYS = {
    "reward_id", "description", "wandb.run_name", "wandb.extra_tags",
    "reward.train_policy", "data.train_dataset", "data.eval_dataset_ids",
}


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> Dict[str, Any]:
    configs = {}
    manifests = {}
    states = {}
    for arm in ARMS:
        rd = run_dir(arm)
        configs[arm] = _flatten(load_json(rd / "config_used.json"))
        manifests[arm] = load_json(rd / "run_manifest.json")
        states[arm] = load_json(rd / "ablation_run_state.json")

    # cross-arm config diff
    all_keys = sorted({k for c in configs.values() for k in c})
    diffs = []
    for k in all_keys:
        vals = {arm: configs[arm].get(k) for arm in ARMS}
        uniq = {repr(v) for v in vals.values()}
        if len(uniq) > 1:
            diffs.append({"key": k, "values": vals,
                          "identity_field": k in IDENTITY_KEYS
                          or any(k.startswith(p) for p in ("wandb.", "description", "reward_id"))})
    unexpected = [d for d in diffs if not d["identity_field"]]

    # manifest hash parity
    hash_keys = ["dataset_hash", "eval_subset_hash", "reward_spec_hash",
                 "executor_hash", "registry_version"]
    hash_parity = {}
    for hk in hash_keys:
        vals = {arm: (manifests[arm].get("hashes") or {}).get(hk) for arm in ARMS}
        hash_parity[hk] = {"values": vals, "identical": len(set(vals.values())) == 1}

    # config_hash SHOULD differ (encodes train_policy) — check
    ch = {arm: (manifests[arm].get("hashes") or {}).get("config_hash") for arm in ARMS}

    # seeds / git
    seeds = {arm: manifests[arm].get("seed") for arm in ARMS}
    commits = {arm: manifests[arm].get("git_commit") for arm in ARMS}

    # declared vs manifest reward policy
    declared = {arm: manifests[arm].get("reward_train_policy") for arm in ARMS}

    # run state steps
    step_names = {arm: sorted((states[arm].get("steps") or {}).keys()) for arm in ARMS}

    payload = {
        "n_config_keys": len(all_keys),
        "config_diff_keys": diffs,
        "unexpected_config_diffs": unexpected,
        "unexpected_config_diff_count": len(unexpected),
        "hash_parity": hash_parity,
        "config_hash_per_arm": ch,
        "seeds": seeds,
        "git_commits": commits,
        "declared_reward_policy": declared,
        "run_state_steps": step_names,
        "verdict": {
            "configs_identical_up_to_identity": len(unexpected) == 0,
            "same_data_and_executor": all(v["identical"] for k, v in hash_parity.items()),
            "same_seed": len(set(seeds.values())) == 1,
            "same_commit": len(set(commits.values())) == 1,
        },
    }
    write_json("a01_config_parity.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    print("unexpected diffs:", r["unexpected_config_diff_count"])
    print("verdict:", r["verdict"])
