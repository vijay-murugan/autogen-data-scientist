import os

# LLM Configurations
# Standard Ollama local URL
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "glm-5:cloud"  # Correct model name
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Cloud Model Configurations (Optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# Dataset Configurations
DATASET_PATH = os.path.join("data", "amazon_products_sales_2025.csv")

DATASET_URL = "ikramshah512/amazon-products-sales-dataset-42k-items-2025"

# Environment Settings
WORKING_DIR = "run_artifacts"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
