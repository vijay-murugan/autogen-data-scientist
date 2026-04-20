"""AmazonDA/01 -- Dataset shape.

Maps to tasks.json task 1:
    "Load the dataset and report the total number of products and the
    list of columns available."
"""

TASK_ID = "AmazonDA/01"
ENTRY_POINT = "get_dataset_shape"

PROMPT = '''from __future__ import annotations

import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return the shape of the Amazon products DataFrame as ``(n_rows, n_cols)``.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame loaded from the Amazon Products Sales 2025 dataset.

    Returns
    -------
    tuple[int, int]
        ``(number_of_rows, number_of_columns)``.

    Examples
    --------
    >>> import pandas as pd
    >>> get_dataset_shape(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    (2, 2)
    >>> get_dataset_shape(pd.DataFrame(columns=["x", "y", "z"]))
    (0, 3)
    """
'''

CANONICAL_SOLUTION = """    return (len(df), len(df.columns))
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert candidate(df) == (3, 2)

    empty = pd.DataFrame(columns=["x", "y", "z"])
    assert candidate(empty) == (0, 3)

    single = pd.DataFrame({"only": [42]})
    assert candidate(single) == (1, 1)

    realistic = pd.DataFrame({
        "product_name": ["n1", "n2"],
        "product_category": ["Electronics", "Books"],
        "discounted_price": [10.0, 20.0],
    })
    assert candidate(realistic) == (2, 3)
"""
