"""Consolidate the individual task modules into a single HumanEval-style JSONL.

Run (from this directory or the project root):

    python -m humaneval.build_jsonl

It writes ``humaneval/tasks.jsonl`` with one JSON record per task, using the
same schema as the original OpenAI HumanEval benchmark:

    {"task_id": ..., "entry_point": ..., "prompt": ...,
     "canonical_solution": ..., "test": ...}
"""

from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from humaneval import load_tasks  # noqa: E402


def main() -> int:
    tasks = load_tasks()
    out_path = os.path.join(THIS_DIR, "tasks.jsonl")
    with open(out_path, "w", encoding="utf-8") as fp:
        for task in tasks:
            fp.write(json.dumps(task, ensure_ascii=False))
            fp.write("\n")
    print(f"Wrote {len(tasks)} tasks to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
