"""AmazonDA/08 -- Top purchase categories by mean units sold last month.

Maps to tasks.json task 8:
    "Identify the top 3 categories with the highest average
    'purchased_last_month'."
"""

TASK_ID = "AmazonDA/08"
ENTRY_POINT = "top_purchase_categories"

PROMPT = '''from __future__ import annotations

import pandas as pd


def top_purchase_categories(df: pd.DataFrame, n: int = 3) -> list[str]:
    """Top ``n`` categories ranked by mean ``purchased_last_month``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns ``product_category`` and
        ``purchased_last_month``.
    n : int, default 3
        Number of categories to return.

    Returns
    -------
    list[str]
        Category names in descending order of the group mean. Ties on
        the mean are broken alphabetically by category name. Categories
        for which every ``purchased_last_month`` value is NaN are
        excluded. If the DataFrame has fewer than ``n`` eligible
        categories, all eligible categories are returned.
    """
'''

CANONICAL_SOLUTION = """    grouped = df.groupby("product_category")["purchased_last_month"].mean().dropna()
    pairs = [(str(k), float(v)) for k, v in grouped.items()]
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return [kv[0] for kv in pairs[:n]]
"""

TEST = """def check(candidate):
    import pandas as pd

    # A: mean 150, B: mean 50, C: mean 300, D: mean 150
    # Desc by mean, ties (A,D) broken alphabetically -> C, A, D, B
    df = pd.DataFrame({
        "product_category":     ["A", "A", "B", "B", "C", "C", "D"],
        "purchased_last_month": [100, 200, 50,  50,  300, 300, 150],
    })
    assert candidate(df, n=3) == ["C", "A", "D"]
    assert candidate(df, n=1) == ["C"]
    assert candidate(df, n=10) == ["C", "A", "D", "B"]
    assert candidate(df) == ["C", "A", "D"]

    # Category with only NaN values is excluded
    df2 = pd.DataFrame({
        "product_category":     ["A", "A", "B"],
        "purchased_last_month": [float("nan"), float("nan"), 10],
    })
    assert candidate(df2, n=5) == ["B"]

    # Empty DataFrame
    empty = pd.DataFrame({"product_category": [], "purchased_last_month": []})
    assert candidate(empty, n=3) == []
"""
