import os
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL, WORKING_DIR
from app.core.custom_client import SimpleOllamaClient

def get_ollama_client():
    """Returns a configured SimpleOllamaClient."""
    return SimpleOllamaClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_BASE_URL
    )

def get_code_execution_tool():
    """Returns a configured PythonCodeExecutionTool."""
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    executor = LocalCommandLineCodeExecutor(work_dir=WORKING_DIR)
    return PythonCodeExecutionTool(executor=executor)
