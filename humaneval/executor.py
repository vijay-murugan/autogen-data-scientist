"""Sandboxed subprocess executor for HumanEval-style task completions.

Given a task dict (with ``prompt``, ``entry_point``, ``test``) and a candidate
``completion`` (the function body produced by one of the runners), this module
assembles a runnable Python program, executes it with a subprocess timeout,
and reports pass/fail plus timing.

Section 4.1 metrics recorded here:
  * ``passed``            -- did the candidate's body satisfy all asserts in TEST
  * ``exit_code``         -- subprocess return code (syntax errors -> nonzero)
  * ``exec_latency_sec``  -- wall-clock time spent in the subprocess
  * ``diagnostic``        -- last non-empty line of stderr/stdout on failure
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

SUCCESS_SENTINEL = "HUMANEVAL_CHECK_OK"


@dataclass
class ExecutionResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    diagnostic: str
    exec_latency_sec: float
    program: str


def assemble_program(prompt: str, entry_point: str, completion: str, test: str) -> str:
    """Concatenate PROMPT + completion + TEST + check(entry_point) into a program."""
    return (
        prompt
        + completion
        + "\n\n"
        + test
        + f"\ncheck({entry_point})\n"
        + f"print({SUCCESS_SENTINEL!r})\n"
    )


def execute_candidate(
    task: dict, completion: str, timeout_sec: int = 30
) -> ExecutionResult:
    """Run ``completion`` against ``task``'s unit tests in a subprocess."""
    program = assemble_program(
        prompt=task["prompt"],
        entry_point=task["entry_point"],
        completion=completion,
        test=task["test"],
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as fp:
        fp.write(program)
        tmp_path = fp.name

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        exec_latency_sec = time.perf_counter() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
        passed = exit_code == 0 and SUCCESS_SENTINEL in stdout
        if passed:
            diag = ""
        else:
            tail = (stderr or stdout).strip().splitlines()
            diag = tail[-1] if tail else f"return code {exit_code}"
    except subprocess.TimeoutExpired as exc:
        exec_latency_sec = time.perf_counter() - start
        stdout = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        )
        exit_code = -1
        passed = False
        diag = f"timeout after {timeout_sec}s"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return ExecutionResult(
        passed=passed,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        diagnostic=diag,
        exec_latency_sec=exec_latency_sec,
        program=program,
    )
