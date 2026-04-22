import ast
import os
import subprocess
import sys
from datetime import datetime
from typing import Iterable
from autogen_core.tools import FunctionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL, WORKING_DIR
from app.core.custom_client import SimpleOllamaClient

IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
}

def get_ollama_client():
    """Returns a configured SimpleOllamaClient."""
    return SimpleOllamaClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_BASE_URL
    )

def get_code_execution_tool(work_dir: str | None = None):
    """Returns a configured PythonCodeExecutionTool.

    Parameters
    ----------
    work_dir
        Code executor working directory (saved figures, cwd for runs).
        Defaults to WORKING_DIR.
    """
    wd = work_dir or WORKING_DIR
    if not os.path.exists(wd):
        os.makedirs(wd)
    executor = LocalCommandLineCodeExecutor(work_dir=wd)
    return PythonCodeExecutionTool(executor=executor)


def _extract_import_roots(code: str) -> set[str]:
    """Extract top-level imported module names from Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _stdlib_modules() -> set[str]:
    """Return modules provided by the Python standard library."""
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "os",
        "sys",
        "json",
        "math",
        "datetime",
        "re",
        "csv",
        "pathlib",
        "typing",
        "itertools",
        "collections",
        "functools",
        "statistics",
        "random",
        "subprocess",
    }


def _to_packages(modules: Iterable[str]) -> list[str]:
    """Convert import roots to pip package names."""
    packages = []
    stdlib = _stdlib_modules()
    for module in sorted(set(modules)):
        if module in stdlib:
            continue
        packages.append(IMPORT_TO_PACKAGE.get(module, module))
    return sorted(set(packages))


def install_run_dependencies(code: str) -> str:
    """
    Generate a per-run requirements file from code imports and install packages.
    Returns a status message with file path and install output.
    """
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    packages = _to_packages(_extract_import_roots(code))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requirements_path = os.path.join(WORKING_DIR, f"requirements_{timestamp}.txt")

    with open(requirements_path, "w", encoding="utf-8") as req_file:
        req_file.write("\n".join(packages))
        if packages:
            req_file.write("\n")

    if not packages:
        return (
            f"No external imports found. Created empty requirements file at "
            f"{requirements_path}."
        )

    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return (
            f"Dependency installation failed using {requirements_path}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return (
        f"Dependencies installed successfully from {requirements_path}.\n"
        f"STDOUT:\n{result.stdout}"
    )


def get_dependency_install_tool() -> FunctionTool:
    """Returns a tool that installs dependencies from generated code imports."""
    return FunctionTool(
        install_run_dependencies,
        description=(
            "Generate a run-specific requirements.txt from Python code imports and "
            "install dependencies before execution."
        ),
        name="install_run_dependencies",
    )
