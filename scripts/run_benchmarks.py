"""
Multi-dataset benchmark harness: single vs multi pipelines, JSONL trails, manifests.

Usage (from project root):

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --datasets retail_sample,hr_sample
    python scripts/run_benchmarks.py --tasks retail/01,retail/02
    python scripts/run_benchmarks.py --pipelines single
    python scripts/run_benchmarks.py --with-judge   # runs Ollama judge after each task pair
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.agents.multi_agent import run_multi_agent_pipeline
from app.agents.single_agent import run_single_agent_pipeline
from app.agents.ml_agent import run_multi_agent_ml_pipeline
from app.benchmark.judge import judge_run_pair_from_disk, RunPairPaths
from app.benchmark.registry import BenchmarkRegistry, load_registry, resolve_dataset_path
from app.core.config import PROJECT_ROOT


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(key: str) -> str:
    return key.replace("/", "_").replace(" ", "_")


def _message_payload(msg: Any) -> dict[str, Any]:
    source = getattr(msg, "source", "") or ""
    typ = type(msg).__name__
    content = getattr(msg, "content", msg)
    if not isinstance(content, str):
        try:
            content = str(content)
        except Exception:
            content = repr(content)
    created = str(getattr(msg, "created_at", "") or "")
    return {
        "source": source,
        "message_type": typ,
        "content": content[:200000],
        "created_at": created,
    }


async def _consume_pipeline(
    gen: AsyncIterator[Any],
    trail_fp,
    seq_holder: list[int],
) -> None:
    async for msg in gen:
        seq_holder[0] += 1
        if type(msg).__name__ == "TaskResult":
            rec = {
                "seq": seq_holder[0],
                "ts": _utc_now_iso(),
                "event": "task_result",
                "message_type": "TaskResult",
                "content": "",
            }
            trail_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trail_fp.flush()
            continue
        rec = {
            "seq": seq_holder[0],
            "ts": _utc_now_iso(),
            "event": "agent_message",
            **_message_payload(msg),
        }
        trail_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        trail_fp.flush()


def _artifact_manifest(artifact_dir: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if os.path.isdir(artifact_dir):
        for root, _, names in os.walk(artifact_dir):
            for n in sorted(names):
                p = os.path.join(root, n)
                rel = os.path.relpath(p, artifact_dir)
                try:
                    st = os.stat(p)
                    files.append(
                        {
                            "path": p,
                            "rel": rel,
                            "size_bytes": st.st_size,
                            "mtime": st.st_mtime,
                        }
                    )
                except OSError:
                    continue
    return {"artifact_dir": artifact_dir, "files": files}


def _read_last_agent_result(trail_path: str) -> str:
    """Return a short final assistant message excerpt from a trail file."""
    last_content = ""
    try:
        with open(trail_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("event") != "agent_message":
                    continue
                content = obj.get("content")
                if isinstance(content, str) and content.strip():
                    last_content = content.strip()
    except OSError:
        return ""
    # Keep console output and CSV manageable.
    return " ".join(last_content.split())[:500]


async def _run_one_pipeline(
    *,
    pipeline: str,
    task_type: str,
    task_text: str,
    dataset_abs: str,
    artifact_dir: str,
    schema_hint: str | None,
    trail_path: str,
    task_timeout_sec: float,
) -> tuple[float, str | None]:
    os.makedirs(os.path.dirname(trail_path) or ".", exist_ok=True)
    seq = [0]
    t0 = time.perf_counter()
    err: str | None = None
    with open(trail_path, "w", encoding="utf-8") as trail_fp:
        trail_fp.write(
            json.dumps(
                {
                    "seq": 0,
                    "ts": _utc_now_iso(),
                    "event": "run_start",
                    "pipeline": pipeline,
                    "task_type": task_type,
                    "dataset_path": dataset_abs,
                    "artifact_dir": artifact_dir,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        try:
            if task_type == "ml":
                if pipeline == "single":
                    gen = run_single_agent_pipeline(
                        task_text, dataset_abs, artifact_dir=artifact_dir
                    )
                else:
                    gen = run_multi_agent_ml_pipeline(
                        task_text,
                        dataset_path=dataset_abs,
                        artifact_dir=artifact_dir,
                        schema_hint=schema_hint,
                    )
            else:
                if pipeline == "single":
                    gen = run_single_agent_pipeline(
                        task_text, dataset_abs, artifact_dir=artifact_dir
                    )
                else:
                    gen = run_multi_agent_pipeline(
                        task_text, dataset_abs, artifact_dir=artifact_dir
                    )
            await asyncio.wait_for(
                _consume_pipeline(gen, trail_fp, seq),
                timeout=task_timeout_sec,
            )
        except asyncio.TimeoutError:
            err = (
                f"TimeoutError: pipeline '{pipeline}' exceeded "
                f"{task_timeout_sec:.1f}s for task '{task_text[:120]}'"
            )
            trail_fp.write(
                json.dumps(
                    {
                        "seq": seq[0] + 1,
                        "ts": _utc_now_iso(),
                        "event": "error",
                        "content": err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            trail_fp.write(
                json.dumps(
                    {
                        "seq": seq[0] + 1,
                        "ts": _utc_now_iso(),
                        "event": "error",
                        "content": err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        trail_fp.write(
            json.dumps(
                {
                    "seq": seq[0] + 2,
                    "ts": _utc_now_iso(),
                    "event": "run_end",
                    "pipeline": pipeline,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    elapsed = time.perf_counter() - t0
    return elapsed, err


async def run_benchmark_main(args: argparse.Namespace) -> int:
    registry: BenchmarkRegistry = load_registry(args.registry)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, args.outdir, run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    wanted_ds = None
    if args.datasets:
        wanted_ds = {x.strip() for x in args.datasets.split(",") if x.strip()}
    wanted_tasks = None
    if args.tasks:
        wanted_tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    rows: list[dict[str, Any]] = []
    meta_tasks: list[dict[str, Any]] = []

    for ds in registry.datasets:
        if wanted_ds and ds.id not in wanted_ds:
            continue
        dataset_abs = resolve_dataset_path(ds.dataset_path)
        if not os.path.isfile(dataset_abs):
            print(f"[skip] dataset file missing: {dataset_abs}", file=sys.stderr)
            continue
        for task in ds.tasks:
            if wanted_tasks and task.id not in wanted_tasks:
                continue
            base = _sanitize(f"{ds.id}__{task.id}")
            art_root = os.path.join(run_dir, "artifacts")
            Path(art_root).mkdir(parents=True, exist_ok=True)

            entry: dict[str, Any] = {
                "dataset_id": ds.id,
                "domain": ds.domain,
                "task_id": task.id,
                "task_type": task.task_type,
                "task": task.task,
                "expected_output": task.expected_output,
                "reference_metrics_hint": task.reference_metrics_hint,
                "dataset_path_abs": dataset_abs,
            }

            for pipeline in pipelines:
                subdir = os.path.join(art_root, f"{pipeline}_{base}")
                os.makedirs(subdir, exist_ok=True)
                trail_path = os.path.join(run_dir, f"{pipeline}_{base}.jsonl")
                lat, err = await _run_one_pipeline(
                    pipeline=pipeline,
                    task_type=task.task_type,
                    task_text=task.task,
                    dataset_abs=dataset_abs,
                    artifact_dir=subdir,
                    schema_hint=ds.ml_schema_hint,
                    trail_path=trail_path,
                    task_timeout_sec=args.task_timeout_sec,
                )
                manifest = _artifact_manifest(subdir)
                man_path = os.path.join(run_dir, f"{pipeline}_{base}_manifest.json")
                with open(man_path, "w", encoding="utf-8") as fp:
                    json.dump(manifest, fp, indent=2)

                rows.append(
                    {
                        "dataset_id": ds.id,
                        "task_id": task.id,
                        "task_type": task.task_type,
                        "pipeline": pipeline,
                        "latency_sec": round(lat, 4),
                        "error": err or "",
                        "trail_path": trail_path,
                        "manifest_path": man_path,
                        "artifact_dir": subdir,
                        "result_excerpt": _read_last_agent_result(trail_path),
                    }
                )
                result_excerpt = _read_last_agent_result(trail_path)
                entry.setdefault("runs", {})[pipeline] = {
                    "latency_sec": round(lat, 4),
                    "error": err or "",
                    "trail_path": trail_path,
                    "manifest_path": man_path,
                    "artifact_dir": subdir,
                    "result_excerpt": result_excerpt,
                }
                print(
                    f"[result] {ds.id} {task.id} | pipeline={pipeline} | "
                    f"error={bool(err)} | {result_excerpt or '<no assistant output>'}"
                )

            meta_tasks.append(entry)

            if args.with_judge and "single" in pipelines and "multi" in pipelines:
                sp = os.path.join(run_dir, f"single_{base}.jsonl")
                mp = os.path.join(run_dir, f"multi_{base}.jsonl")
                sm = os.path.join(run_dir, f"single_{base}_manifest.json")
                mm = os.path.join(run_dir, f"multi_{base}_manifest.json")
                if os.path.isfile(sp) and os.path.isfile(mp):
                    try:
                        judge_out = await judge_run_pair_from_disk(
                            {
                                "task": task.task,
                                "expected_output": task.expected_output,
                                "reference_metrics_hint": task.reference_metrics_hint,
                            },
                            RunPairPaths(
                                task_id=task.id,
                                dataset_id=ds.id,
                                single_trail=sp,
                                multi_trail=mp,
                                single_manifest=sm,
                                multi_manifest=mm,
                            ),
                        )
                        jp = os.path.join(run_dir, f"judge_{base}.json")
                        with open(jp, "w", encoding="utf-8") as fp:
                            json.dump(judge_out, fp, indent=2)
                        entry["judge_path"] = jp
                    except Exception as exc:
                        print(f"[judge-fail] {ds.id} {task.id}: {exc}", file=sys.stderr)

    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "run_id": run_id,
                "registry": args.registry,
                "created": _utc_now_iso(),
                "tasks": meta_tasks,
            },
            fp,
            indent=2,
        )

    if rows:
        csv_path = os.path.join(run_dir, "summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"Benchmark run complete: {run_dir}")
    if rows and any((r.get("error") or "").strip() for r in rows):
        print(
            "One or more pipelines reported errors; see summary.csv and trail .jsonl files.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-dataset benchmark harness.")
    parser.add_argument(
        "--registry",
        default=os.path.join("benchmarks", "registry.json"),
        help="Path to registry JSON (relative to project root or absolute).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("run_artifacts", "benchmark_runs"),
        help="Output root under project (default run_artifacts/benchmark_runs).",
    )
    parser.add_argument("--run-id", default=None, help="Optional run folder name.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset ids.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids.")
    parser.add_argument(
        "--pipelines",
        default="single,multi",
        help="Comma-separated: single, multi.",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="After each task, call Ollama judge (requires single and multi in --pipelines).",
    )
    parser.add_argument(
        "--task-timeout-sec",
        type=float,
        default=300.0,
        help="Per-pipeline timeout in seconds for each task (default: 300).",
    )
    args = parser.parse_args()
    return asyncio.run(run_benchmark_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
