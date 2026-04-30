"""AmazonDA/09 -- Sponsored vs non-sponsored average price.

Maps to tasks.json task 9:
    "Compare the average 'discounted_price' of 'is_sponsored' products vs
    non-sponsored products."
"""

TASK_ID = "AmazonDA/09"
ENTRY_POINT = "compare_sponsored_prices"

PROMPT = '''from __future__ import annotations

import pandas as pd


def compare_sponsored_prices(df: pd.DataFrame) -> dict[str, float]:
    """Mean ``discounted_price`` grouped by ``is_sponsored``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns ``is_sponsored`` (boolean) and
        ``discounted_price`` (numeric).

    Returns
    -------
    dict[str, float]
        A dict with exactly two keys:
          * ``"sponsored"`` -- mean ``discounted_price`` where
            ``is_sponsored`` is True.
          * ``"non_sponsored"`` -- mean ``discounted_price`` where
            ``is_sponsored`` is False.
        NaN values in ``discounted_price`` are excluded from each mean.
        If a group has no valid values, the corresponding entry is
        ``float('nan')``.
    """
'''

CANONICAL_SOLUTION = """    sp = df[df["is_sponsored"] == True]["discounted_price"].dropna()
    ns = df[df["is_sponsored"] == False]["discounted_price"].dropna()
    s_mean = float(sp.mean()) if len(sp) else float("nan")
    n_mean = float(ns.mean()) if len(ns) else float("nan")
    return {"sponsored": s_mean, "non_sponsored": n_mean}
"""

TEST = """def check(candidate):
    import math

    import pandas as pd

    df = pd.DataFrame({
        "is_sponsored":     [True, True, False, False, False],
        "discounted_price": [10.0, 20.0, 100.0, 200.0, 300.0],
    })
    result = candidate(df)
    assert set(result.keys()) == {"sponsored", "non_sponsored"}
    assert math.isclose(result["sponsored"], 15.0, rel_tol=1e-9)
    assert math.isclose(result["non_sponsored"], 200.0, rel_tol=1e-9)

    # Empty sponsored group -> NaN for sponsored
    no_sp = pd.DataFrame({
        "is_sponsored":     [False, False],
        "discounted_price": [10.0, 20.0],
    })
    r2 = candidate(no_sp)
    assert math.isnan(r2["sponsored"])
    assert math.isclose(r2["non_sponsored"], 15.0, rel_tol=1e-9)

    # Empty non-sponsored group -> NaN for non-sponsored
    no_ns = pd.DataFrame({
        "is_sponsored":     [True, True],
        "discounted_price": [5.0, 15.0],
    })
    r3 = candidate(no_ns)
    assert math.isclose(r3["sponsored"], 10.0, rel_tol=1e-9)
    assert math.isnan(r3["non_sponsored"])

    # NaN in price is excluded group-wise
    nan_df = pd.DataFrame({
        "is_sponsored":     [True, True, False],
        "discounted_price": [10.0, float("nan"), 100.0],
    })
    r4 = candidate(nan_df)
    assert math.isclose(r4["sponsored"], 10.0, rel_tol=1e-9)
    assert math.isclose(r4["non_sponsored"], 100.0, rel_tol=1e-9)
"""
