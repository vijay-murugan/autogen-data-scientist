"""Deterministic runner that returns the task's canonical solution.

Used by the smoke test to exercise the full runner -> executor -> metrics
pipeline without requiring an Ollama server. Also useful as an upper-bound
oracle when interpreting benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanonicalRunResult:
    completion: str
    raw_response: str
    llm_latency_sec: float
    interaction_count: int


def run_canonical(task: dict) -> CanonicalRunResult:
    solution = task["canonical_solution"]
    return CanonicalRunResult(
        completion=solution,
        raw_response=solution,
        llm_latency_sec=0.0,
        interaction_count=0,
    )
