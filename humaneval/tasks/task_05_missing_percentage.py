"""AmazonDA/05 -- Missing-data percentage per column.

Maps to tasks.json task 5:
    "Check for missing values in the 'discounted_price' and 'product_category'
    columns and report the percentage of missing data for each."
Generalised to any list of column names.
"""

TASK_ID = "AmazonDA/05"
ENTRY_POINT = "missing_data_percentage"

PROMPT = '''from __future__ import annotations

import pandas as pd


def missing_data_percentage(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Percentage of missing values in each of the requested columns.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list[str]
        Column names to inspect. A missing column must raise ``KeyError``.

    Returns
    -------
    dict[str, float]
        Maps each column name to the percentage (0.0 to 100.0) of rows
        where the value is missing (NaN / NaT / None). The percentage
        should be rounded to 4 decimal places. For an empty DataFrame,
        return 0.0 for every column.
    """
'''

CANONICAL_SOLUTION = """    result: dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(col)
        total = len(df)
        pct = 0.0 if total == 0 else float(df[col].isna().mean() * 100.0)
        result[col] = round(pct, 4)
    return result
"""

TEST = """def check(candidate):
    import pandas as pd

    df = pd.DataFrame({
        "a": [1.0, None, 3.0, None],       # 2 / 4 = 50.0%
        "b": [1, 2, 3, 4],                 # 0 / 4 = 0.0%
        "c": [None, None, None, None],     # 4 / 4 = 100.0%
    })
    assert candidate(df, ["a", "b", "c"]) == {"a": 50.0, "b": 0.0, "c": 100.0}

    # Subset
    assert candidate(df, ["b"]) == {"b": 0.0}

    # Empty DataFrame -> 0.0 for each requested column
    empty = pd.DataFrame({"x": [], "y": []})
    assert candidate(empty, ["x", "y"]) == {"x": 0.0, "y": 0.0}

    # Missing column raises KeyError
    raised = False
    try:
        candidate(df, ["nonexistent"])
    except KeyError:
        raised = True
    assert raised, "Expected KeyError for column not in DataFrame"

    # Non-round percentages are rounded to 4 decimals
    df_odd = pd.DataFrame({"v": [1, None, None]})  # 2/3 = 66.6666...%
    r = candidate(df_odd, ["v"])
    assert r == {"v": 66.6667}
"""
