import os
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


# LLM Configurations
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Cloud Model Configurations (Optional)
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# Dataset Configurations
DATASET_PATH = "amazon_products_sales_2025.csv"
DATASET_URL = "ikramshah512/amazon-products-sales-dataset-42k-items-2025"
PROJECT_ROOT = _REPO_ROOT

DEFAULT_DATASET_REF = os.getenv(
    "DEFAULT_DATASET_REF",
    "ikramshah512/amazon-products-sales-dataset-42k-items-2025",
)
DEFAULT_DATASET_PATH = os.path.join(
    PROJECT_ROOT, "data", "amazon_products_sales_2025.csv"
)
DATASET_PATH = DEFAULT_DATASET_PATH

# Environment Settings
WORKING_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)


def ollama_http_headers() -> dict[str, str] | None:
    """Return Authorization headers for ollama.com (or any host requiring an API key)."""
    key = (OLLAMA_API_KEY or "").strip()
    if not key:
        return None
    return {"Authorization": f"Bearer {key}"}


def ollama_async_client_kwargs(host: str | None = None) -> dict[str, Any]:
    """Keyword args for ``ollama.AsyncClient`` including optional Bearer auth."""
    kw: dict[str, Any] = {"host": (host or OLLAMA_BASE_URL).rstrip("/")}
    headers = ollama_http_headers()
    if headers is not None:
        kw["headers"] = headers
    return kw
