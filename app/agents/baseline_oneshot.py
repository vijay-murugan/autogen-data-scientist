import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List

from ollama import AsyncClient

from app.core.config import (
    DATASET_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    WORKING_DIR,
    ollama_async_client_kwargs,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _extract_python_code(raw_text: str) -> str:
    fenced = re.findall(r"```python\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    generic = re.findall(r"```\s*(.*?)```", raw_text, flags=re.DOTALL)
    if generic:
        return generic[0].strip()
    return raw_text.strip()


def _contains_any(text: str, patterns: List[str]) -> bool:
    lower = text.lower()
    return any(p in lower for p in patterns)


def _required_step_flags(task: str, code: str, stdout: str, image_count: int) -> Dict[str, Any]:
    modeling_keywords = [
        "model",
        "regression",
        "classification",
        "predict",
        "forecast",
        "clustering",
        "train",
    ]
    modeling_requested = _contains_any(task, modeling_keywords)
    modeling_present = _contains_any(
        code,
        ["sklearn", "statsmodels", "xgboost", "lightgbm", "fit(", "predict("],
    )

    flags = {
        "data_loading": _contains_any(code, ["read_csv(", "pd.read_csv"]),
        "data_cleaning": _contains_any(
            code, ["dropna(", "fillna(", "drop_duplicates(", "astype(", "to_numeric(", "isna("]
        ),
        "eda": _contains_any(
            code,
            ["describe(", "value_counts(", "groupby(", "corr(", "head(", "info(", "hist(", "boxplot("],
        ),
        "modeling_if_relevant": (not modeling_requested) or modeling_present,
        "at_least_two_visualizations": image_count >= 2,
        "final_printed_insights": "final insights" in stdout.lower(),
    }
    flags["modeling_requested"] = modeling_requested
    return flags


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


async def run_oneshot_baseline(task: str) -> Dict[str, Any]:
    if not task.strip():
        raise ValueError("Task cannot be empty.")

    dataset_abs = os.path.abspath(DATASET_PATH)
    if not os.path.isfile(dataset_abs):
        raise FileNotFoundError(f"Dataset not found at {dataset_abs}")

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(WORKING_DIR, "baseline_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    system_prompt = (
        "You are a single-agent data analyst baseline. "
        "Return exactly one complete Python script and nothing else.\n\n"
        "Constraints:\n"
        "- One-shot only: generate the full script in one response.\n"
        "- Use pandas, numpy, and matplotlib only (do not use seaborn).\n"
        "- First load the dataset, then print: columns list, dtypes, and head(5).\n"
        "- Only reference columns that exist in df.columns.\n"
        "- Choose visualizations based on available columns.\n"
        "- If expected columns are missing, fall back to valid EDA and plots using existing numeric/categorical columns.\n"
        "- After inspecting df.columns and dtypes, identify available numeric and categorical columns.\n"
        "- Always create at least two plots using existing columns and save them as image files.\n"
        "- Do not hardcode column names like 'price' or 'category'; choose from the detected column lists.\n"
        "- Create two plots unconditionally when possible: first numeric column histogram + second numeric column boxplot; "
        "if only one numeric column exists, use one numeric plot plus one categorical count/bar plot.\n"
        "- Save explicitly as plot_1.png and plot_2.png in the current working directory.\n"
        "- Use safe fallbacks when needed: histogram of a numeric column, boxplot of a numeric column, "
        "bar chart of top categories from a categorical column, or pandas/matplotlib count-style bar plot.\n"
        "- The script must: load data, clean data, perform EDA, do modeling if task requires it, "
        "create at least 2 visualizations, and print final insights.\n"
        "- Before plotting, inspect available columns and only use columns that actually exist.\n"
        "- Save visualizations as image files in the current working directory.\n"
        "- Print a section header exactly: FINAL INSIGHTS\n"
        "- Script must be executable as-is.\n"
    )
    user_prompt = (
        f"Dataset CSV path: {dataset_abs}\n"
        f"Analytics task: {task}\n\n"
        "Write the full Python code now."
    )

    client = AsyncClient(**ollama_async_client_kwargs(host=OLLAMA_BASE_URL))

    llm_start = time.perf_counter()
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    llm_latency_sec = time.perf_counter() - llm_start

    raw_response = response.message.content or ""
    code = _extract_python_code(raw_response)

    code_path = os.path.abspath(os.path.join(run_dir, "generated_code.py"))
    raw_path = os.path.abspath(os.path.join(run_dir, "llm_raw_response.txt"))
    _write_text(code_path, code)
    _write_text(raw_path, raw_response)

    before_files = set(os.listdir(run_dir))
    exec_start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, code_path],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    execution_latency_sec = time.perf_counter() - exec_start

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    _write_text(os.path.join(run_dir, "stdout.log"), stdout)
    _write_text(os.path.join(run_dir, "stderr.log"), stderr)

    after_files = set(os.listdir(run_dir))
    new_files = sorted(list(after_files - before_files))
    image_files = [f for f in new_files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

    step_flags = _required_step_flags(task, code, stdout, len(image_files))
    execution_success = proc.returncode == 0
    required_steps_completed = all(
        [
            step_flags["data_loading"],
            step_flags["data_cleaning"],
            step_flags["eda"],
            step_flags["modeling_if_relevant"],
            step_flags["at_least_two_visualizations"],
            step_flags["final_printed_insights"],
        ]
    )

    result = {
        "run_id": run_id,
        "run_dir": os.path.abspath(run_dir),
        "dataset_path": dataset_abs,
        "model": OLLAMA_MODEL,
        "task": task,
        "code_path": code_path,
        "raw_response_path": raw_path,
        "stdout_path": os.path.abspath(os.path.join(run_dir, "stdout.log")),
        "stderr_path": os.path.abspath(os.path.join(run_dir, "stderr.log")),
        "image_files": image_files,
        "execution_success": execution_success,
        "required_steps": step_flags,
        "required_steps_completed": required_steps_completed,
        "llm_latency_sec": round(llm_latency_sec, 3),
        "execution_latency_sec": round(execution_latency_sec, 3),
        "total_latency_sec": round(llm_latency_sec + execution_latency_sec, 3),
        "return_code": proc.returncode,
    }
    evaluation_path = os.path.abspath(os.path.join(run_dir, "evaluation_summary.json"))
    result["evaluation_path"] = evaluation_path
    _write_text(evaluation_path, json.dumps(result, indent=2))
    return result


def run_oneshot_baseline_sync(task: str) -> Dict[str, Any]:
    if sys.platform == "win32":
        # Keeps subprocess behavior stable in this project on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(run_oneshot_baseline(task))
