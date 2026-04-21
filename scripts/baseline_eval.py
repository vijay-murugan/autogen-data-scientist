"""
Baseline reference evaluation (no LLM).

Runs five deterministic checks on the configured CSV and prints a summary table.
Use these numbers as ground truth when judging whether the single-agent baseline
produces correct answers for the same tasks.

Usage (from project root):
  .venv\\Scripts\\python.exe scripts/baseline_eval.py
  .venv\\Scripts\\python.exe scripts/baseline_eval.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)


def _load_df():
    import pandas as pd

    from app.core.config import DATASET_PATH

    path = os.path.abspath(DATASET_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path), path


def _clean_price(series):
    import pandas as pd

    if series.dtype == object:
        s = series.astype(str).str.replace(r"[,$₹\s]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def run_evaluations() -> dict:
    import pandas as pd

    df, path = _load_df()
    rows = []
    n, p = len(df), len(df.columns)

    # Task 1 — shape
    ok1 = n > 0 and p > 0
    rows.append(
        {
            "id": 1,
            "name": "Dataset shape",
            "detail": f"rows={n}, cols={p}",
            "pass": ok1,
        }
    )

    # Task 2 — required columns exist
    required = {
        "product_category",
        "discounted_price",
        "product_rating",
    }
    missing_cols = sorted(required - set(df.columns))
    ok2 = len(missing_cols) == 0
    rows.append(
        {
            "id": 2,
            "name": "Required columns",
            "detail": "all present" if ok2 else f"missing: {missing_cols}",
            "pass": ok2,
        }
    )

    # Task 3 — mean discounted_price (after cleaning)
    dp = _clean_price(df["discounted_price"]) if "discounted_price" in df.columns else pd.Series(dtype=float)
    mean_dp = float(dp.mean()) if len(dp) else float("nan")
    ok3 = ok2 and dp.notna().any() and pd.notna(mean_dp)
    rows.append(
        {
            "id": 3,
            "name": "Mean discounted_price",
            "detail": f"{mean_dp:.6g}" if ok3 else "n/a",
            "pass": ok3,
        }
    )

    # Task 4 — top 5 product_category counts
    if "product_category" in df.columns:
        vc = df["product_category"].astype(str).value_counts().head(5)
        top5 = vc.to_dict()
        ok4 = len(top5) > 0
        detail = ", ".join(f"{k[:24]}:{v}" for k, v in list(top5.items())[:3])
        if len(top5) > 3:
            detail += ", ..."
    else:
        top5 = {}
        ok4 = False
        detail = "n/a"
    rows.append(
        {
            "id": 4,
            "name": "Top-5 product_category",
            "detail": detail,
            "pass": ok4,
            "extra": top5,
        }
    )

    # Task 5 — missing % for discounted_price & product_category
    if ok2:
        miss_dp = float(dp.isna().mean() * 100)
        miss_cat = float(df["product_category"].isna().mean() * 100)
        ok5 = True
        detail = f"missing discounted_price={miss_dp:.2f}%, product_category={miss_cat:.2f}%"
    else:
        miss_dp = miss_cat = float("nan")
        ok5 = False
        detail = "n/a"
    rows.append(
        {
            "id": 5,
            "name": "Missing data rates",
            "detail": detail,
            "pass": ok5,
        }
    )

    for r in rows:
        r.setdefault("extra", None)
    return {
        "dataset_path": path,
        "shape_rows": n,
        "shape_cols": p,
        "column_names": list(df.columns),
        "mean_discounted_price": mean_dp if ok3 else None,
        "top5_product_category": top5 if ok4 else None,
        "pct_missing_discounted_price": miss_dp if ok2 else None,
        "pct_missing_product_category": miss_cat if ok2 else None,
        "tasks": rows,
        "all_pass": all(t["pass"] for t in rows),
    }


def _print_table(result: dict) -> None:
    print(f"Dataset: {result['dataset_path']}")
    print(f"Shape: {result['shape_rows']} rows x {result['shape_cols']} cols\n")
    print(f"{'ID':<4} {'Task':<28} {'PASS':<6} Detail")
    print("-" * 90)
    for t in result["tasks"]:
        mark = "yes" if t["pass"] else "no"
        print(f"{t['id']:<4} {t['name']:<28} {mark:<6} {t['detail']}")
    print("-" * 90)
    print(f"Overall: {'PASS' if result['all_pass'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline reference eval (deterministic, no LLM).")
    parser.add_argument("--json", action="store_true", help="Print full JSON including reference stats.")
    args = parser.parse_args()

    try:
        result = run_evaluations()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Drop non-serializable extras in tasks for clean JSON
        out = {k: v for k, v in result.items() if k != "tasks"}
        out["tasks"] = [
            {k: v for k, v in t.items() if k != "extra" or v is not None}
            for t in result["tasks"]
        ]
        print(json.dumps(out, indent=2))
    else:
        _print_table(result)

    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
