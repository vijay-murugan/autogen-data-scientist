"""Load ``benchmarks/registry.json`` for multi-dataset benchmark runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from app.core.config import PROJECT_ROOT


@dataclass
class BenchmarkTask:
    id: str
    task_type: str
    task: str
    expected_output: str
    reference_metrics_hint: str | None = None


@dataclass
class BenchmarkDataset:
    id: str
    domain: str
    dataset_path: str
    ml_schema_hint: str | None
    tasks: list[BenchmarkTask]


@dataclass
class BenchmarkRegistry:
    version: int
    datasets: list[BenchmarkDataset]


def _parse_task(obj: dict[str, Any]) -> BenchmarkTask:
    return BenchmarkTask(
        id=str(obj["id"]),
        task_type=str(obj.get("task_type", "analytics")),
        task=str(obj["task"]),
        expected_output=str(obj.get("expected_output", "")),
        reference_metrics_hint=obj.get("reference_metrics_hint"),
    )


def _parse_dataset(obj: dict[str, Any]) -> BenchmarkDataset:
    tasks = [_parse_task(t) for t in obj.get("tasks", [])]
    return BenchmarkDataset(
        id=str(obj["id"]),
        domain=str(obj.get("domain", "")),
        dataset_path=str(obj["dataset_path"]),
        ml_schema_hint=obj.get("ml_schema_hint"),
        tasks=tasks,
    )


def load_registry(path: str | None = None) -> BenchmarkRegistry:
    """Load registry JSON. *path* is absolute or relative to project root."""
    root = PROJECT_ROOT
    reg_path = path or os.path.join(root, "benchmarks", "registry.json")
    if not os.path.isabs(reg_path):
        reg_path = os.path.join(root, reg_path)
    with open(reg_path, encoding="utf-8") as fp:
        raw = json.load(fp)
    return BenchmarkRegistry(
        version=int(raw.get("version", 1)),
        datasets=[_parse_dataset(d) for d in raw.get("datasets", [])],
    )


def resolve_dataset_path(dataset_path: str) -> str:
    """Resolve a registry dataset_path (often relative to project root) to absolute."""
    if os.path.isabs(dataset_path):
        return dataset_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, dataset_path))
