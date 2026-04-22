"""
Post-hoc Ollama judge over an existing benchmark run folder (meta.json).

Usage:

    python scripts/judge_benchmark_run.py run_artifacts/benchmark_runs/20260101_120000
    python scripts/judge_benchmark_run.py run_artifacts/benchmark_runs/20260101_120000 --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.benchmark.judge import judge_run_pair_from_disk, RunPairPaths


def _sanitize_from_meta(dataset_id: str, task_id: str) -> str:
    return f"{dataset_id}__{task_id}".replace("/", "_").replace(" ", "_")


async def judge_run_dir(run_dir: str, force: bool) -> int:
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"No meta.json in {run_dir}", file=sys.stderr)
        return 1
    with open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)

    results: list[dict] = []
    for entry in meta.get("tasks", []):
        runs = entry.get("runs") or {}
        if "single" not in runs or "multi" not in runs:
            continue
        base = _sanitize_from_meta(entry["dataset_id"], entry["task_id"])
        jp = os.path.join(run_dir, f"judge_{base}.json")
        if os.path.isfile(jp) and not force:
            continue
        sp = runs["single"]["trail_path"]
        mp = runs["multi"]["trail_path"]
        sm = runs["single"]["manifest_path"]
        mm = runs["multi"]["manifest_path"]
        out = await judge_run_pair_from_disk(
            {
                "task": entry.get("task", ""),
                "expected_output": entry.get("expected_output", ""),
                "reference_metrics_hint": entry.get("reference_metrics_hint"),
            },
            RunPairPaths(
                task_id=entry["task_id"],
                dataset_id=entry["dataset_id"],
                single_trail=sp,
                multi_trail=mp,
                single_manifest=sm,
                multi_manifest=mm,
            ),
        )
        with open(jp, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2)
        results.append(out)

    agg = os.path.join(run_dir, "judge_aggregate.json")
    with open(agg, "w", encoding="utf-8") as fp:
        json.dump({"count": len(results), "results": results}, fp, indent=2)
    print(f"Wrote {len(results)} judge file(s) and {agg}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Path to a benchmark run directory containing meta.json")
    p.add_argument("--force", action="store_true", help="Overwrite existing judge_*.json files.")
    args = p.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    return asyncio.run(judge_run_dir(run_dir, args.force))


if __name__ == "__main__":
    raise SystemExit(main())
