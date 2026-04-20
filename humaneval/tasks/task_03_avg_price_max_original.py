"""AmazonDA/03 -- Average discounted price and most expensive product.

Maps to tasks.json task 3:
    "Calculate the average 'discounted_price' across all products and
    identify the product with the highest 'original_price'."
"""

TASK_ID = "AmazonDA/03"
ENTRY_POINT = "avg_price_and_highest_original"

PROMPT = '''from __future__ import annotations

import pandas as pd


def avg_price_and_highest_original(df: pd.DataFrame) -> tuple[float, str]:
    """Average discounted price and name of the product with the highest original price.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns ``discounted_price`` (numeric),
        ``original_price`` (numeric), and ``product_name`` (string).

    Returns
    -------
    tuple[float, str]
        ``(mean_of_discounted_price, product_name_with_max_original_price)``.

    Rules
    -----
    * NaN values in ``discounted_price`` are excluded from the mean.
    * If several rows share the maximum ``original_price``, return the
      lexicographically smallest ``product_name`` among them.
    """
'''

CANONICAL_SOLUTION = """    avg = float(df["discounted_price"].mean())
    max_val = df["original_price"].max()
    tied = df[df["original_price"] == max_val]
    name = sorted(str(n) for n in tied["product_name"].tolist())[0]
    return (avg, name)
"""

TEST = """def check(candidate):
    import math

    import pandas as pd

    df = pd.DataFrame({
        "product_name": ["alpha", "beta", "gamma", "delta"],
        "discounted_price": [10.0, 20.0, 30.0, 40.0],
        "original_price":   [100.0, 200.0, 200.0, 50.0],
    })
    avg, name = candidate(df)
    assert math.isclose(avg, 25.0, rel_tol=1e-9)
    # Tie on original_price between beta and gamma -> alphabetical -> beta
    assert name == "beta"

    # NaN in discounted_price is ignored
    df2 = pd.DataFrame({
        "product_name": ["x", "y"],
        "discounted_price": [10.0, float("nan")],
        "original_price":   [50.0, 60.0],
    })
    avg2, name2 = candidate(df2)
    assert math.isclose(avg2, 10.0, rel_tol=1e-9)
    assert name2 == "y"

    # Unique maximum
    df3 = pd.DataFrame({
        "product_name": ["solo", "mid", "low"],
        "discounted_price": [1.0, 2.0, 3.0],
        "original_price":   [9.0, 5.0, 1.0],
    })
    avg3, name3 = candidate(df3)
    assert math.isclose(avg3, 2.0, rel_tol=1e-9)
    assert name3 == "solo"
"""
