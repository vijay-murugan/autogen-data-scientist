import os

# LLM Configurations
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Cloud Model Configurations (Optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_DATASET_REF = os.getenv(
    "DEFAULT_DATASET_REF",
    "ikramshah512/amazon-products-sales-dataset-42k-items-2025",
)
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "amazon_products_sales_2025.csv")
DATASET_PATH = DEFAULT_DATASET_PATH

# Environment Settings
WORKING_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
