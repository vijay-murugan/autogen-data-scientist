"""AmazonDA/10 -- Count of best-sellers that also have a coupon.

Maps to tasks.json task 10:
    "Identify the number of products that have both 'is_best_seller' and
    'has_coupon' as True."
"""

TASK_ID = "AmazonDA/10"
ENTRY_POINT = "count_bestseller_with_coupon"

PROMPT = '''from __future__ import annotations

import pandas as pd


def count_bestseller_with_coupon(df: pd.DataFrame) -> int:
    """Count rows where ``is_best_seller`` and ``has_coupon`` are both True.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain boolean columns ``is_best_seller`` and ``has_coupon``.

    Returns
    -------
    int
        The number of rows that satisfy both conditions.
    """
'''

CANONICAL_SOLUTION = """    mask = (df["is_best_seller"] == True) & (df["has_coupon"] == True)
    return int(mask.sum())
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({
        "is_best_seller": [True, True,  False, True, False, False],
        "has_coupon":     [True, False, True,  True, False, True],
    })
    # Rows 0 and 3 satisfy both -> count 2
    assert candidate(df) == 2

    # No matches
    none_df = pd.DataFrame({
        "is_best_seller": [False, False, True],
        "has_coupon":     [False, True,  False],
    })
    assert candidate(none_df) == 0

    # All match
    all_df = pd.DataFrame({
        "is_best_seller": [True, True, True],
        "has_coupon":     [True, True, True],
    })
    assert candidate(all_df) == 3

    # Empty DataFrame
    empty = pd.DataFrame({"is_best_seller": [], "has_coupon": []})
    assert candidate(empty) == 0
"""
