#!/usr/bin/env python3
"""
Download real Kaggle datasets listed in benchmarks/kaggle_sources.json into
data/benchmark/kaggle/ as normalized comma-separated UTF-8 CSV files.

Requires: pip install kagglehub (already in project requirements).
Optional: set KAGGLE_USERNAME and KAGGLE_KEY for datasets that return 403 without auth.

Usage (from project root):

    python scripts/download_kaggle_benchmark_data.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import kagglehub  # noqa: E402


def main() -> int:
    spec_path = os.path.join(project_root, "benchmarks", "kaggle_sources.json")
    with open(spec_path, encoding="utf-8") as fp:
        spec = json.load(fp)

    out_dir = os.path.join(project_root, "data", "benchmark", "kaggle")
    os.makedirs(out_dir, exist_ok=True)

    manifest: list[dict] = []
    for entry in spec["datasets"]:
        slug = entry["kaggle_slug"]
        src_name = entry["source_filename"]
        dest_name = entry["dest_csv"]
        opts = dict(entry.get("read_csv_options") or {})

        print(f"Downloading {slug} …")
        root = kagglehub.dataset_download(slug)
        src_path = None
        for walk_root, _, files in os.walk(root):
            if src_name in files:
                src_path = os.path.join(walk_root, src_name)
                break
        if not src_path:
            print(f"ERROR: {src_name} not found under {root}", file=sys.stderr)
            return 1

        df = pd.read_csv(src_path, **opts)
        dest_path = os.path.join(out_dir, dest_name)
        df.to_csv(dest_path, index=False, encoding="utf-8")
        print(f"  Wrote {dest_path} ({len(df)} rows, {len(df.columns)} cols)")
        manifest.append(
            {
                "id": entry["id"],
                "kaggle_slug": slug,
                "kaggle_url": entry.get("kaggle_url", ""),
                "source_cache": src_path,
                "dest_csv": os.path.relpath(dest_path, project_root),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

    man_path = os.path.join(out_dir, "download_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fp:
        json.dump({"spec": spec.get("attribution", ""), "files": manifest}, fp, indent=2)
    print(f"Manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
