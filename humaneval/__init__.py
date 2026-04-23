"""HumanEval-style benchmark for Section 4.1 of the Multi-Agent Intelligent
Data Analytics Framework proposal.

Each task under ``humaneval/tasks`` is a Python module exposing four
string constants following the OpenAI HumanEval convention:

    PROMPT              -- imports + function signature + docstring
    ENTRY_POINT         -- the function name to call in the test
    CANONICAL_SOLUTION  -- reference implementation (body only)
    TEST                -- defines ``check(candidate)`` with asserts

A runnable program for a candidate completion is constructed as:

    PROMPT + candidate_body + "\\n\\n" + TEST + f"\\ncheck({ENTRY_POINT})\\n"

The runner (to be added in a follow-up) will substitute the
``candidate_body`` with output from the single-agent or multi-agent
pipelines and run the concatenated program in a subprocess sandbox.
"""

from importlib import import_module

TASK_MODULES = [
    "humaneval.tasks.task_01_dataset_shape",
    "humaneval.tasks.task_02_top_n_categories",
    "humaneval.tasks.task_03_avg_price_max_original",
    "humaneval.tasks.task_04_rating_distribution",
    "humaneval.tasks.task_05_missing_percentage",
    "humaneval.tasks.task_06_best_seller_discount",
    "humaneval.tasks.task_07_electronics_scatter",
    "humaneval.tasks.task_08_top_purchase_categories",
    "humaneval.tasks.task_09_sponsored_price_comparison",
    "humaneval.tasks.task_10_bestseller_coupon_count",
]


def load_tasks():
    """Return a list of dicts, one per task, with HumanEval-style fields."""
    records = []
    for mod_path in TASK_MODULES:
        mod = import_module(mod_path)
        records.append(
            {
                "task_id": mod.TASK_ID,
                "entry_point": mod.ENTRY_POINT,
                "prompt": mod.PROMPT,
                "canonical_solution": mod.CANONICAL_SOLUTION,
                "test": mod.TEST,
            }
        )
    return records
