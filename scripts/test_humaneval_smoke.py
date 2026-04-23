"""Smoke test for the Section 4.1 HumanEval harness.

Exercises the full runner -> executor -> metrics path on every task using
the canonical-solution runner (no Ollama needed). Also exercises the code
extractor on a few synthetic LLM-style outputs, and does a static import /
dataclass-shape check on the two AutoGen-backed runners so the benchmark
harness does not crash when it reaches for their result fields.

Exit code 0 means all plumbing works end-to-end.

Usage:
    python scripts/test_humaneval_smoke.py
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from humaneval import load_tasks
from humaneval.code_extraction import extract_function_body
from humaneval.executor import execute_candidate
from humaneval.runners.canonical import run_canonical


def test_canonical_round_trip() -> list[str]:
    """Canonical solution + executor should pass every task."""
    failures: list[str] = []
    for task in load_tasks():
        run = run_canonical(task)
        result = execute_candidate(task, run.completion, timeout_sec=15)
        mark = "PASS" if result.passed else "FAIL"
        tail = f" -- {result.diagnostic}" if not result.passed else ""
        print(f"{mark} canonical round-trip -- {task['task_id']} {task['entry_point']}{tail}")
        if not result.passed:
            failures.append(task["task_id"])
    return failures


def test_code_extraction() -> list[str]:
    """Verify the extractor handles fenced / unindented / full-function outputs."""
    failures: list[str] = []
    entry = "add_two"
    prompt = (
        "from __future__ import annotations\n\n"
        "def add_two(a: int, b: int) -> int:\n"
        '    """Return a + b."""\n'
    )
    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(-1, 1) == 0\n"
    )
    task = {"prompt": prompt, "entry_point": entry, "test": test}

    cases = {
        "body_indented":      "    return a + b\n",
        "body_unindented":    "return a + b\n",
        "fenced_body":        "```python\n    return a + b\n```",
        "fenced_unindented":  "```python\nreturn a + b\n```",
        "full_function":      "def add_two(a, b):\n    return a + b\n",
        "fenced_full":        "```python\ndef add_two(a, b):\n    return a + b\n```",
        "with_docstring":     'def add_two(a, b):\n    """docstring"""\n    return a + b\n',
        "full_typed_annotated": (
            "def add_two(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
        "multiline_signature": (
            "def add_two(\n"
            "    a: int,\n"
            "    b: int,\n"
            ") -> int:\n"
            "    return a + b\n"
        ),
        "single_line_function": "def add_two(a, b): return a + b\n",
        "two_space_indent_body": "  return a + b\n",
        "tab_indent_body":      "\treturn a + b\n",
        "preamble_then_def": (
            "Here is the function:\n\n"
            "def add_two(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
        "fenced_multiline_sig": (
            "```python\n"
            "def add_two(\n"
            "    a: int,\n"
            "    b: int,\n"
            ") -> int:\n"
            "    return a + b\n"
            "```"
        ),
        "nested_block_body": (
            "    if a > 0:\n"
            "        return a + b\n"
            "    else:\n"
            "        return a + b\n"
        ),
        "full_with_nested_block": (
            "def add_two(a, b):\n"
            "    if a > 0:\n"
            "        return a + b\n"
            "    else:\n"
            "        return a + b\n"
        ),
        "first_line_flush_rest_indented": (
            "s = a + b\n"
            "    return s\n"
        ),
        "first_line_flush_rest_indented_multi": (
            "x = a\n"
            "    y = b\n"
            "    return x + y\n"
        ),
    }
    for label, raw in cases.items():
        completion = extract_function_body(raw, entry)
        result = execute_candidate(task, completion, timeout_sec=10)
        mark = "PASS" if result.passed else "FAIL"
        tail = f" -- {result.diagnostic}" if not result.passed else ""
        print(f"{mark} extractor -- {label}{tail}")
        if not result.passed:
            failures.append(f"extractor/{label}")
    return failures


# (label, module, entry_fn, required_fields)
RUNNER_SPECS = [
    (
        "single",
        "humaneval.runners.single",
        "run_single",
        ("completion", "raw_response", "llm_latency_sec", "interaction_count"),
    ),
    (
        "multi",
        "humaneval.runners.multi",
        "run_multi",
        (
            "completion", "raw_response", "plan", "critiques",
            "llm_latency_sec", "interaction_count", "turns_used",
            "approved", "messages",
        ),
    ),
]


def test_runner_import_and_shape() -> list[str]:
    """Import the AutoGen-backed runners and confirm their result dataclasses
    expose the fields ``scripts/run_humaneval.py`` reads. Does NOT spin up
    Ollama -- this is a static compatibility check.
    """
    failures: list[str] = []

    for pipeline, module_path, func_name, required_fields in RUNNER_SPECS:
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            print(f"FAIL runner-import -- {pipeline}: {type(exc).__name__}: {exc}")
            failures.append(f"import/{pipeline}")
            continue

        func = getattr(mod, func_name, None)
        if not callable(func):
            print(f"FAIL runner-import -- {pipeline}: missing {func_name}()")
            failures.append(f"entrypoint/{pipeline}")
            continue

        result_cls = None
        for name, obj in vars(mod).items():
            if name.endswith("RunResult") and dataclasses.is_dataclass(obj):
                result_cls = obj
                break

        if result_cls is None:
            print(f"FAIL runner-import -- {pipeline}: no *RunResult dataclass")
            failures.append(f"dataclass/{pipeline}")
            continue

        field_names = set(f.name for f in dataclasses.fields(result_cls))
        missing = set(required_fields) - field_names
        if missing:
            print(
                "FAIL runner-import -- "
                + pipeline + ": " + result_cls.__name__
                + " missing fields " + str(sorted(missing))
            )
            failures.append(f"fields/{pipeline}")
            continue

        print(f"PASS runner-import -- {pipeline} ({result_cls.__name__})")

    return failures


def main() -> int:
    print("[1/3] Canonical round-trip\n")
    c_failures = test_canonical_round_trip()
    print("\n[2/3] Code-extractor cases\n")
    e_failures = test_code_extraction()
    print("\n[3/3] Runner import / dataclass shape\n")
    r_failures = test_runner_import_and_shape()

    failures = c_failures + e_failures + r_failures
    print()
    if failures:
        print(f"SMOKE FAIL: {failures}")
        return 1
    print("SMOKE PASS: runner + executor + extractor all work end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
