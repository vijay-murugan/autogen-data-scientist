"""Verify that every HumanEval-style task in this benchmark is self-consistent.

For each task under ``humaneval/tasks`` this script:
  1. Constructs the full program: ``PROMPT + CANONICAL_SOLUTION + TEST + check(entry_point)``.
  2. Writes it to a temp file.
  3. Runs it with the current Python interpreter under a subprocess timeout.
  4. Reports pass / fail and a one-line diagnostic for any failures.

Usage (from the project root, with the ``humaneval`` package on sys.path):

    python -m humaneval.verify_canonical

Exit code 0 means every canonical solution passed its tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# Allow running as a script without installing the package.
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from humaneval import load_tasks  # noqa: E402

PER_TASK_TIMEOUT_SEC = 30


def _full_program(task: dict) -> str:
    return (
        task["prompt"]
        + task["canonical_solution"]
        + "\n\n"
        + task["test"]
        + f"\ncheck({task['entry_point']})\n"
        + "print('CANONICAL_OK')\n"
    )


def run_task(task: dict) -> tuple[bool, str]:
    program = _full_program(task)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as fp:
        fp.write(program)
        tmp_path = fp.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=PER_TASK_TIMEOUT_SEC,
        )
    finally:
        os.unlink(tmp_path)

    if proc.returncode == 0 and "CANONICAL_OK" in proc.stdout:
        return True, ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    diag = tail[-1] if tail else f"return code {proc.returncode}"
    return False, diag


def main() -> int:
    tasks = load_tasks()
    failures: list[tuple[str, str]] = []

    print(f"Verifying {len(tasks)} canonical solutions...\n")
    for task in tasks:
        ok, diag = run_task(task)
        mark = "PASS" if ok else "FAIL"
        tail = f"  --  {diag}" if not ok else ""
        print(f"{mark}  {task['task_id']:<12} {task['entry_point']}{tail}")
        if not ok:
            failures.append((task["task_id"], diag))

    print()
    if failures:
        print(f"{len(failures)}/{len(tasks)} tasks FAILED.")
        return 1
    print(f"All {len(tasks)} canonical solutions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
