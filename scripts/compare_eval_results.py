#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two ToolAlpaca eval summary JSON files.")
    ap.add_argument("--base_result", type=str, required=True, help="Path to base model eval summary JSON.")
    ap.add_argument("--trained_result", type=str, required=True, help="Path to trained model eval summary JSON.")
    ap.add_argument("--output_path", type=str, default=None, help="Optional path to write comparison JSON.")
    ap.add_argument("--table_path", type=str, default=None, help="Optional path to write markdown comparison table.")
    return ap.parse_args()


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def main() -> None:
    args = parse_args()
    base = load_summary(args.base_result)
    trained = load_summary(args.trained_result)

    base_ast = 100.0 * base.get("mean_soft_score", 0.0)
    trained_ast = 100.0 * trained.get("mean_soft_score", 0.0)
    base_exact = base.get("exact_match_accuracy_percent", 100.0 * base.get("final_accuracy", 0.0))
    trained_exact = trained.get("exact_match_accuracy_percent", 100.0 * trained.get("final_accuracy", 0.0))

    comparison = {
        "benchmark": "ToolAlpaca",
        "dataset_path_base": base.get("dataset_path"),
        "dataset_path_trained": trained.get("dataset_path"),
        "split_name_base": base.get("split_name"),
        "split_name_trained": trained.get("split_name"),
        "split_role_base": base.get("split_role"),
        "split_role_trained": trained.get("split_role"),
        "base_model_path": base["model_path"],
        "trained_model_path": trained["model_path"],
        "base_ast_accuracy_percent": base_ast,
        "trained_ast_accuracy_percent": trained_ast,
        "delta_ast_accuracy_percent_points": trained_ast - base_ast,
        "base_exact_match_percent": base_exact,
        "trained_exact_match_percent": trained_exact,
        "delta_exact_match_percent_points": trained_exact - base_exact,
        "base_parseable_predictions": base["parseable_predictions"],
        "trained_parseable_predictions": trained["parseable_predictions"],
    }
    table = "\n".join(
        [
            "| metric | base | trained | delta |",
            "|---|---:|---:|---:|",
            f"| split_name | {comparison['split_name_base'] or '-'} | {comparison['split_name_trained'] or '-'} | - |",
            f"| **ast_accuracy_percent** | **{base_ast:.2f}** | **{trained_ast:.2f}** | **{trained_ast - base_ast:+.2f}** |",
            f"| exact_match_percent | {base_exact:.2f} | {trained_exact:.2f} | {trained_exact - base_exact:+.2f} |",
            f"| parseable_predictions | {comparison['base_parseable_predictions']} | {comparison['trained_parseable_predictions']} | {comparison['trained_parseable_predictions'] - comparison['base_parseable_predictions']} |",
            f"| base_model | `{comparison['base_model_path']}` |  |  |",
            f"| trained_model |  | `{comparison['trained_model_path']}` |  |",
        ]
    )

    if args.output_path:
        parent = os.path.dirname(args.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
    if args.table_path:
        parent = os.path.dirname(args.table_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.table_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")

    print("ToolAlpaca comparison")
    print(table)
    if args.output_path:
        print(f"  comparison json:        {args.output_path}")
    if args.table_path:
        print(f"  comparison table:       {args.table_path}")


if __name__ == "__main__":
    main()

