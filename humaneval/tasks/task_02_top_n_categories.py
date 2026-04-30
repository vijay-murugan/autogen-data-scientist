"""AmazonDA/02 -- Top-N product categories by count.

Maps to tasks.json task 2:
    "Find the top 5 product categories by count ..."
We return the raw counts so a wrong aggregation fails a unit test,
independent of how the chart is drawn.
"""

TASK_ID = "AmazonDA/02"
ENTRY_POINT = "top_n_categories"

PROMPT = '''from __future__ import annotations

import pandas as pd


def top_n_categories(df: pd.DataFrame, n: int = 5) -> list[tuple[str, int]]:
    """Return the top ``n`` values of ``product_category`` by row count.

    The output is a list of ``(category, count)`` tuples, sorted by count
    in descending order. Ties on count are broken alphabetically by
    category name (ascending). If the DataFrame has fewer than ``n``
    distinct categories, all of them are returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``product_category`` column.
    n : int, default 5
        Maximum number of categories to return.

    Returns
    -------
    list[tuple[str, int]]
        Ordered list of ``(category, count)`` tuples.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"product_category": ["A", "A", "B", "B", "C"]})
    >>> top_n_categories(df, n=2)
    [('A', 2), ('B', 2)]
    """
'''

CANONICAL_SOLUTION = """    counts = df["product_category"].value_counts()
    pairs = [(str(k), int(v)) for k, v in counts.items()]
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs[:n]
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({
        "product_category": ["A", "A", "A", "B", "B", "C", "D", "D", "E"],
    })
    # Counts: A=3, B=2, D=2, C=1, E=1. Tie B/D -> alphabetical (B before D).
    expected_all = [("A", 3), ("B", 2), ("D", 2), ("C", 1), ("E", 1)]
    assert candidate(df, n=5) == expected_all
    assert candidate(df, n=3) == [("A", 3), ("B", 2), ("D", 2)]

    # Default n = 5
    assert candidate(df) == expected_all

    # Fewer categories than n -> return all of them
    assert candidate(df, n=99) == expected_all

    # Empty DataFrame
    empty = pd.DataFrame({"product_category": []})
    assert candidate(empty) == []

    # Case sensitivity: "a" and "A" are distinct
    df_case = pd.DataFrame({"product_category": ["a", "a", "A"]})
    assert candidate(df_case, n=2) == [("a", 2), ("A", 1)]
"""
