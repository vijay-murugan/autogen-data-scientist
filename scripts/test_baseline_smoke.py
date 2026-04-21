"""
Smoke test: same LocalCommandLineCodeExecutor + dataset path as the Baseline agent.
Does not call Ollama — validates pandas/matplotlib + executor + CSV.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from app.core.config import DATASET_PATH, WORKING_DIR


async def main() -> None:
    dataset_abs = os.path.abspath(DATASET_PATH)
    if not os.path.isfile(dataset_abs):
        print("FAIL: dataset missing at", dataset_abs)
        sys.exit(1)

    executor = LocalCommandLineCodeExecutor(work_dir=WORKING_DIR)
    path_literal = json.dumps(dataset_abs)
    code = f"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = {path_literal}
df = pd.read_csv(p)
print("SMOKE_OK rows=", len(df), "cols=", len(df.columns))
if "product_rating" in df.columns:
    plt.figure(figsize=(4, 3))
    df["product_rating"].dropna().head(5000).hist(bins=20)
    plt.tight_layout()
    plt.savefig("baseline_smoke_hist.png")
    print("SMOKE_OK saved baseline_smoke_hist.png")
else:
    print("SMOKE_OK no product_rating column")
"""
    token = CancellationToken()
    result = await executor.execute_code_blocks(
        [CodeBlock(code=code, language="python")],
        token,
    )
    print("exit_code:", result.exit_code)
    print("output:\n", result.output)
    hist_path = os.path.join(WORKING_DIR, "baseline_smoke_hist.png")
    if result.exit_code != 0:
        sys.exit(1)
    if not os.path.isfile(hist_path):
        print("WARN: expected chart not found at", hist_path)
    else:
        print("PASS: chart exists", hist_path)


if __name__ == "__main__":
    asyncio.run(main())
