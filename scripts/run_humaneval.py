"""Section 4.1 benchmark harness.

Runs the single-agent baseline and the multi-agent pipeline over every
HumanEval-style task defined in ``humaneval/tasks/`` and reports:
  * pass@1 (fraction of tasks whose completion passes all unit tests),
  * mean LLM latency, execution latency, and total latency,
  * mean interaction count per task.

Per-task rows and raw LLM outputs are saved under
``run_artifacts/humaneval/<timestamp>/`` so results are reproducible and
auditable.

Usage (from the project root):

    # Full benchmark: single + multi on all 10 tasks
    python scripts/run_humaneval.py

    # Smoke: no Ollama needed, uses canonical solutions
    python scripts/run_humaneval.py --canonical

    # Subset of pipelines / tasks
    python scripts/run_humaneval.py --pipelines single --tasks AmazonDA/01,AmazonDA/02
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from humaneval import load_tasks
from humaneval.executor import execute_candidate
from humaneval.runners.canonical import run_canonical


def _run_pipeline(pipeline: str, task: dict):
    if pipeline == "single":
        from humaneval.runners.single import run_single

        return run_single(task)
    if pipeline == "multi":
        from humaneval.runners.multi import run_multi

        return run_multi(task)
    if pipeline == "canonical":
        return run_canonical(task)
    raise ValueError(f"Unknown pipeline: {pipeline}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HumanEval-style benchmark for Section 4.1.",
    )
    parser.add_argument(
        "--pipelines",
        default="single,multi",
        help="Comma-separated pipelines to run: single, multi, canonical.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task IDs to run; default is all.",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Shortcut for --pipelines canonical (no LLM needed).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-task execution timeout seconds (default: 30).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(project_root, "run_artifacts", "humaneval"),
        help="Output directory root.",
    )
    args = parser.parse_args()

    pipelines = (
        ["canonical"]
        if args.canonical
        else [p.strip() for p in args.pipelines.split(",") if p.strip()]
    )

    tasks = load_tasks()
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t["task_id"] in wanted]
        if not tasks:
            print("No tasks matched --tasks filter.", file=sys.stderr)
            return 1

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, run_id)
    raw_dir = os.path.join(run_dir, "raw")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    aggregate: dict[str, dict] = {}

    for pipeline in pipelines:
        Path(os.path.join(raw_dir, pipeline)).mkdir(parents=True, exist_ok=True)
        passed = 0
        total_llm_latency = 0.0
        total_exec_latency = 0.0
        total_interactions = 0

        for task in tasks:
            print(f"[{pipeline}] {task['task_id']} -- {task['entry_point']}")
            pipeline_error: str | None = None
            completion = ""
            raw = ""
            llm_latency = 0.0
            interactions = 0
            run_result = None

            try:
                run_result = _run_pipeline(pipeline, task)
                completion = run_result.completion
                raw = run_result.raw_response
                llm_latency = run_result.llm_latency_sec
                interactions = run_result.interaction_count
            except Exception as exc:
                pipeline_error = f"{type(exc).__name__}: {exc}"

            exec_result = execute_candidate(task, completion, timeout_sec=args.timeout)

            total_llm_latency += llm_latency
            total_exec_latency += exec_result.exec_latency_sec
            total_interactions += interactions
            if exec_result.passed and pipeline_error is None:
                passed += 1

            raw_path = os.path.join(
                raw_dir,
                pipeline,
                task["task_id"].replace("/", "_") + ".txt",
            )
            with open(raw_path, "w", encoding="utf-8") as fp:
                # Multi-agent extras: planner output, critic turns, approval status.
                # These exist only on MultiRunResult; emitting them first makes the
                # file easier to audit end-to-end.
                if run_result is not None and hasattr(run_result, "plan"):
                    fp.write("=== planner ===\n")
                    fp.write((run_result.plan or "") + "\n\n")
                # Researcher transcripts so we can see whether ResearchAgent
                # actually ran and what it produced.
                if run_result is not None and hasattr(run_result, "researcher_outputs"):
                    researcher_outputs = run_result.researcher_outputs or []
                    for idx, txt in enumerate(researcher_outputs, 1):
                        fp.write(f"=== researcher turn {idx} ===\n")
                        fp.write((txt or "") + "\n\n")
                    if not researcher_outputs:
                        fp.write("=== researcher (NO OUTPUT) ===\n\n")
                # Coder transcripts: essential for diagnosing silent
                # DataScientist failures where the model emits no text
                # (selector-routing bug or reflection hiccup).
                if run_result is not None and hasattr(run_result, "coder_outputs"):
                    coder_outputs = run_result.coder_outputs or []
                    for idx, txt in enumerate(coder_outputs, 1):
                        fp.write(f"=== coder turn {idx} ===\n")
                        fp.write((txt or "") + "\n\n")
                    if not coder_outputs:
                        fp.write("=== coder (NO OUTPUT) ===\n\n")
                if run_result is not None and hasattr(run_result, "critiques"):
                    critiques = run_result.critiques or []
                    for idx, crit in enumerate(critiques, 1):
                        fp.write(f"=== critic turn {idx} ===\n")
                        fp.write((crit or "") + "\n\n")
                    if hasattr(run_result, "approved"):
                        fp.write(
                            f"=== critic verdict ===\napproved={run_result.approved} "
                            f"turns_used={getattr(run_result, 'turns_used', '?')}\n\n"
                        )
                fp.write("=== raw response ===\n")
                fp.write(raw + "\n\n")
                fp.write("=== extracted completion ===\n")
                fp.write(completion + "\n\n")
                fp.write("=== diagnostic ===\n")
                fp.write(exec_result.diagnostic + "\n")
                if pipeline_error:
                    fp.write(f"\n=== pipeline error ===\n{pipeline_error}\n")

            rows.append(
                {
                    "pipeline": pipeline,
                    "task_id": task["task_id"],
                    "entry_point": task["entry_point"],
                    "passed": int(exec_result.passed and pipeline_error is None),
                    "pipeline_error": pipeline_error or "",
                    "exec_exit_code": exec_result.exit_code,
                    "llm_latency_sec": round(llm_latency, 4),
                    "exec_latency_sec": round(exec_result.exec_latency_sec, 4),
                    "total_latency_sec": round(
                        llm_latency + exec_result.exec_latency_sec, 4
                    ),
                    "interactions": interactions,
                    "diagnostic": exec_result.diagnostic,
                }
            )

        total = len(tasks)
        aggregate[pipeline] = {
            "tasks": total,
            "passed": passed,
            "pass_at_1": (passed / total) if total else 0.0,
            "mean_llm_latency_sec": (total_llm_latency / total) if total else 0.0,
            "mean_exec_latency_sec": (total_exec_latency / total) if total else 0.0,
            "mean_total_latency_sec": ((total_llm_latency + total_exec_latency) / total)
            if total
            else 0.0,
            "mean_interactions": (total_interactions / total) if total else 0.0,
        }

    per_task_path = os.path.join(run_dir, "per_task.csv")
    with open(per_task_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    agg_path = os.path.join(run_dir, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as fp:
        json.dump(aggregate, fp, indent=2)

    print("\n=== Aggregate ===")
    for pipeline, stats in aggregate.items():
        print(
            f"{pipeline:<10} pass@1={stats['pass_at_1']:.2%} "
            f"({stats['passed']}/{stats['tasks']})  "
            f"mean_llm={stats['mean_llm_latency_sec']:.2f}s  "
            f"mean_exec={stats['mean_exec_latency_sec']:.3f}s  "
            f"mean_interactions={stats['mean_interactions']:.2f}"
        )
    print(f"\nResults written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
