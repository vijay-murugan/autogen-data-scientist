"""AmazonDA/06 -- Average discount for best sellers.

Maps to tasks.json task 6:
    "Find the average 'discount_percentage' for products that are
    'is_best_seller'."
"""

TASK_ID = "AmazonDA/06"
ENTRY_POINT = "avg_discount_for_best_sellers"

PROMPT = '''from __future__ import annotations

import pandas as pd


def avg_discount_for_best_sellers(df: pd.DataFrame) -> float:
    """Mean ``discount_percentage`` over rows where ``is_best_seller`` is True.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns ``is_best_seller`` (boolean) and
        ``discount_percentage`` (numeric).

    Returns
    -------
    float
        The arithmetic mean of ``discount_percentage`` over best-seller
        rows. If there are no best-seller rows, or every such row has a
        missing ``discount_percentage``, return ``0.0``. NaN values are
        excluded from the mean.
    """
'''

CANONICAL_SOLUTION = """    subset = df[df["is_best_seller"] == True]
    if len(subset) == 0:
        return 0.0
    vals = subset["discount_percentage"].dropna()
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())
"""

TEST = """def check(candidate):
    import math

    import pandas as pd

    df = pd.DataFrame({
        "is_best_seller":      [True, True, False, True, False],
        "discount_percentage": [10.0, 20.0, 50.0, 30.0, 40.0],
    })
    # Best-seller values: 10, 20, 30 -> mean 20.0
    assert math.isclose(candidate(df), 20.0, rel_tol=1e-9)

    # No best sellers
    none_df = pd.DataFrame({
        "is_best_seller":      [False, False],
        "discount_percentage": [10.0, 20.0],
    })
    assert candidate(none_df) == 0.0

    # All NaN among best sellers
    nan_df = pd.DataFrame({
        "is_best_seller":      [True, True],
        "discount_percentage": [float("nan"), float("nan")],
    })
    assert candidate(nan_df) == 0.0

    # Mixed NaN: NaN ignored in mean
    mixed = pd.DataFrame({
        "is_best_seller":      [True, True, True],
        "discount_percentage": [10.0, float("nan"), 30.0],
    })
    assert math.isclose(candidate(mixed), 20.0, rel_tol=1e-9)

    # Empty DataFrame
    empty = pd.DataFrame({"is_best_seller": [], "discount_percentage": []})
    assert candidate(empty) == 0.0
"""
