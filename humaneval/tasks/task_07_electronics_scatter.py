"""AmazonDA/07 -- Data for Electronics scatter plot.

Maps to tasks.json task 7:
    "Plot a scatter plot of 'total_reviews' vs 'product_rating' for products
    in the 'Electronics' category."

We test the underlying filtered DataFrame rather than the plot.
"""

TASK_ID = "AmazonDA/07"
ENTRY_POINT = "electronics_reviews_vs_rating"

PROMPT = '''from __future__ import annotations

import pandas as pd


def electronics_reviews_vs_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Return the data needed to scatter-plot reviews vs rating for Electronics.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns ``product_category``, ``total_reviews``,
        and ``product_rating``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with exactly the columns ``total_reviews`` and
        ``product_rating`` (in that order), containing only rows where
        ``product_category == "Electronics"``. Rows with NaN in either
        column are dropped. The returned DataFrame uses a fresh
        ``RangeIndex`` starting at 0.
    """
'''

CANONICAL_SOLUTION = """    subset = df[df["product_category"] == "Electronics"]
    out = subset[["total_reviews", "product_rating"]].dropna().reset_index(drop=True)
    return out
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({
        "product_category": ["Electronics", "Books", "Electronics", "Electronics", "Toys"],
        "total_reviews":    [100, 200, 300, float("nan"), 400],
        "product_rating":   [4.0, 3.5, 4.5, 3.0, 5.0],
    })
    result = candidate(df)

    assert list(result.columns) == ["total_reviews", "product_rating"]
    assert len(result) == 2
    assert result["total_reviews"].tolist() == [100.0, 300.0]
    assert result["product_rating"].tolist() == [4.0, 4.5]
    assert list(result.index) == [0, 1]

    # No Electronics rows
    no_el = pd.DataFrame({
        "product_category": ["Books", "Toys"],
        "total_reviews":    [1, 2],
        "product_rating":   [1.0, 2.0],
    })
    r2 = candidate(no_el)
    assert list(r2.columns) == ["total_reviews", "product_rating"]
    assert len(r2) == 0

    # Drops rows where product_rating is NaN too
    df3 = pd.DataFrame({
        "product_category": ["Electronics", "Electronics"],
        "total_reviews":    [10, 20],
        "product_rating":   [float("nan"), 4.0],
    })
    r3 = candidate(df3)
    assert len(r3) == 1
    assert r3["total_reviews"].tolist() == [20.0]
    assert r3["product_rating"].tolist() == [4.0]
"""
