"""AmazonDA/04 -- Rating distribution (histogram bin counts).

Maps to tasks.json task 4:
    "Analyze the distribution of 'product_rating' and plot a histogram of ratings."

We check the underlying bin counts rather than the plot, so unit tests
are deterministic.
"""

TASK_ID = "AmazonDA/04"
ENTRY_POINT = "rating_distribution"

PROMPT = '''from __future__ import annotations

import pandas as pd


def rating_distribution(df: pd.DataFrame, bins: list[float]) -> list[int]:
    """Compute histogram counts of ``product_rating`` over the given bin edges.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``product_rating`` column.
    bins : list[float]
        Monotonically increasing bin edges of length ``n + 1`` for ``n`` bins.
        Bin ``i`` covers ``[bins[i], bins[i+1])`` for all bins except the
        last, which is the closed interval ``[bins[-2], bins[-1]]``.

    Returns
    -------
    list[int]
        A list of length ``len(bins) - 1`` with per-bin integer counts.

    Rules
    -----
    * NaN values in ``product_rating`` are excluded before counting.
    * Use the same semantics as ``numpy.histogram``.
    """
'''

CANONICAL_SOLUTION = """    import numpy as np

    ratings = df["product_rating"].dropna().to_numpy()
    counts, _ = np.histogram(ratings, bins=bins)
    return [int(c) for c in counts]
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({
        "product_rating": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    })
    # Bins [1,2), [2,3), [3,4), [4,5]
    assert candidate(df, [1.0, 2.0, 3.0, 4.0, 5.0]) == [2, 2, 2, 3]

    # NaNs are excluded
    df2 = pd.DataFrame({"product_rating": [1.0, 2.0, float("nan"), 3.0]})
    assert candidate(df2, [1.0, 2.0, 3.0, 4.0]) == [1, 1, 1]

    # Single bin covering everything
    df3 = pd.DataFrame({"product_rating": [0.0, 2.5, 5.0]})
    assert candidate(df3, [0.0, 5.0]) == [3]

    # Empty ratings
    empty = pd.DataFrame({"product_rating": []})
    assert candidate(empty, [0.0, 1.0, 2.0]) == [0, 0]

    # Out-of-range values are dropped by numpy.histogram
    df_oor = pd.DataFrame({"product_rating": [-1.0, 0.5, 10.0]})
    assert candidate(df_oor, [0.0, 1.0, 2.0]) == [1, 0]
"""
