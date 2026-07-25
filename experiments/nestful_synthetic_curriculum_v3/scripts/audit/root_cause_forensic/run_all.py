"""Run all forensic analyses; per-module isolation; summary to stdout."""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    "a01_config_parity",
    "a02_reward_dispatch",
    "a03_adapter_audit",
    "a04_update_strength",
    "a05_eval_audit",
    "a06_data_transfer",
    "a07_counterfactual_rescore",
    "a08_credit_audit",
]


def main() -> int:
    results = {}
    failed = []
    for name in MODULES:
        print(f"=== {name} ===", flush=True)
        try:
            mod = importlib.import_module(name)
            results[name] = mod.main()
            print(f"[ok] {name}", flush=True)
        except Exception:
            failed.append(name)
            traceback.print_exc()
    from common import write_json
    write_json("_run_status.json", {"ok": [m for m in MODULES if m not in failed],
                                    "failed": failed})
    print(json.dumps({"ok": len(MODULES) - len(failed), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
