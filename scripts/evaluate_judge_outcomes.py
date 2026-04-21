"""
Aggregate judge outcomes from benchmark run directories.

Usage:
    python scripts/evaluate_judge_outcomes.py run_artifacts/benchmark_runs/20260421_012556
    python scripts/evaluate_judge_outcomes.py run_artifacts/benchmark_runs/20260421_012556 run_artifacts/benchmark_runs/20260421_013010
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any


def _passes_gate(side_scores: dict[str, Any], require_overall: bool) -> bool:
    correctness = int(side_scores.get("correctness_1_5", 0))
    methodology = int(side_scores.get("methodology_ml_1_5", 0))
    leakage = int(side_scores.get("leakage_and_validation_1_5", 0))
    overall = int(side_scores.get("overall_1_5", 0))
    if require_overall and overall < 4:
        return False
    return correctness >= 4 and methodology >= 4 and leakage >= 4


def _classify(single_pass: bool, multi_pass: bool) -> str:
    if single_pass and multi_pass:
        return "both_passed"
    if single_pass and not multi_pass:
        return "single_passed_multi_failed"
    if not single_pass and multi_pass:
        return "single_failed_multi_passed"
    return "both_failed"


def _iter_judge_files(run_dir: str) -> list[str]:
    pattern = os.path.join(run_dir, "judge_*.json")
    files = [p for p in glob.glob(pattern) if not p.endswith("judge_aggregate.json")]
    return sorted(files)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate benchmark judge outcomes.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more benchmark run directories containing judge_*.json files.",
    )
    parser.add_argument(
        "--require-overall",
        action="store_true",
        help="Also require overall_1_5 >= 4 for pass classification.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    counts = {
        "single_failed_multi_passed": 0,
        "single_passed_multi_failed": 0,
        "both_passed": 0,
        "both_failed": 0,
    }
    total = 0
    single_pass_total = 0
    multi_pass_total = 0

    for run_dir in args.run_dirs:
        run_abs = os.path.abspath(run_dir)
        if not os.path.isdir(run_abs):
            print(f"[skip] not a directory: {run_abs}", file=sys.stderr)
            continue
        for judge_path in _iter_judge_files(run_abs):
            data = _load_json(judge_path)
            scores = data.get("scores", {})
            single_scores = scores.get("single", {})
            multi_scores = scores.get("multi", {})
            single_pass = _passes_gate(single_scores, args.require_overall)
            multi_pass = _passes_gate(multi_scores, args.require_overall)
            label = _classify(single_pass, multi_pass)
            counts[label] += 1
            total += 1
            single_pass_total += int(single_pass)
            multi_pass_total += int(multi_pass)
            rows.append(
                {
                    "run_dir": run_abs,
                    "dataset_id": str(data.get("dataset_id", "")),
                    "task_id": str(data.get("task_id", "")),
                    "label": label,
                    "winner": str(scores.get("comparison_winner", "")),
                }
            )

    if total == 0:
        print("No judge files found.")
        return 1

    print("=== Aggregate Judge Outcome ===")
    print(f"pairs_evaluated: {total}")
    print(f"single_pass_rate: {single_pass_total / total:.3f}")
    print(f"multi_pass_rate: {multi_pass_total / total:.3f}")
    for k in (
        "single_failed_multi_passed",
        "single_passed_multi_failed",
        "both_passed",
        "both_failed",
    ):
        print(f"{k}: {counts[k]} ({counts[k] / total:.3f})")

    print("\n=== Per-task Outcomes ===")
    for row in rows:
        print(
            f"{row['run_dir']} | {row['dataset_id']} {row['task_id']} | "
            f"{row['label']} | winner={row['winner']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
