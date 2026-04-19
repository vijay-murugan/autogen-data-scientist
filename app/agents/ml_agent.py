"""
ML Agent Pipeline
-----------------
Adds two agents after the standard EDA/cleaning pipeline:

  ModelSelector  — reads the task prompt + dataset profile, outputs a
                   structured JSON decision (model key + parameters).

  MLAnalyst      — receives the decision, calls the corresponding
                   pre-built function from MODEL_REGISTRY, and saves
                   artifacts to WORKING_DIR.

Usage:
  from app.agents.ml_agent import run_ml_pipeline
  async for message in run_ml_pipeline(task):
      ...
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from app.core.config import DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool

DATASET_ABS = os.path.abspath(DATASET_PATH)

# ------------------------------------------------------------------
# Known columns on the Amazon dataset — given to the selector so it
# can reason about which columns exist without needing to read the file.
# ------------------------------------------------------------------
DATASET_SCHEMA = """
Columns available in the Amazon Products Sales 2025 dataset:
  - product_name        (string)
  - product_category    (string)   e.g. Electronics, Clothing, Home
  - discounted_price    (float)    price after discount, in USD
  - original_price      (float)    original listed price, in USD
  - discount_percentage (float)    0-100
  - product_rating      (float)    0-5 stars
  - total_reviews       (int)      number of customer reviews
  - is_best_seller      (bool)     True/False
  - is_sponsored        (bool)     True/False
  - has_coupon          (bool)     True/False
  - purchased_last_month(int)      units purchased in the last 30 days
"""

# ------------------------------------------------------------------
# Decision guide embedded in the selector's system prompt
# ------------------------------------------------------------------
SELECTOR_RULES = """
Decision rules (apply in order, pick the FIRST that matches):

1. Task mentions predicting / estimating a numeric column AND asks for
   simplicity, interpretability, or a quick baseline
   → model: "linear_regression"
     target_col: the numeric column to predict
     features: []   (empty list means use all numeric columns)

2. Task mentions predicting a binary flag or category
   (is_best_seller, is_sponsored, has_coupon, product_category)
   → model: "random_forest_classifier"
     target_col: the binary/categorical column
     features: []

3. Task mentions grouping, segmenting, clustering, finding patterns,
   or discovering product types — with NO explicit target column
   → model: "kmeans"
     target_col: null
     features: ["discounted_price", "product_rating", "total_reviews"]
       (or whichever numeric features the task focuses on)
     k: 4   (or a number mentioned in the task)

4. Task mentions outliers, anomalies, unusual, suspicious, weird,
   or products that don't fit the pattern
   → model: "isolation_forest"
     target_col: null
     features: ["discounted_price", "product_rating", "total_reviews",
                "discount_percentage"]

5. Task asks "what drives X", "explain why", "what factors influence",
   "feature importance", or "which variables matter most"
   → model: "xgboost_shap"
     target_col: the column whose drivers are being investigated
     features: []

6. None of the above — default:
   → model: "xgboost_shap"
     target_col: "purchased_last_month"
     features: []
"""


def _build_selector_agent(client) -> AssistantAgent:
    return AssistantAgent(
        name="ModelSelector",
        model_client=client,
        system_message=(
            "You are an ML strategy expert. Your only job is to read the user's "
            "analytics task and decide which ML model is most appropriate.\n\n"
            + DATASET_SCHEMA
            + "\n"
            + SELECTOR_RULES
            + "\nOutput ONLY a single JSON object — no extra text, no markdown "
            "fences, no explanation before or after it:\n"
            '{"model": "<key>", "target_col": "<col or null>", '
            '"features": ["col1", ...], "k": 4, '
            '"rationale": "<one concise sentence>"}\n\n'
            "Valid model keys: linear_regression, random_forest_classifier, "
            "kmeans, isolation_forest, xgboost_shap\n\n"
            "After outputting the JSON say HANDOFF_TO_ML on a new line."
        ),
    )


def _build_ml_analyst_agent(client, code_tool) -> AssistantAgent:
    return AssistantAgent(
        name="MLAnalyst",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are an ML execution specialist.\n\n"
            "Wait for a message containing HANDOFF_TO_ML. "
            "Extract the JSON decision from that message, then use your "
            "code execution tool to run the corresponding model.\n\n"
            "Template — copy this exactly and fill in the blanks:\n\n"
            "```python\n"
            "import pandas as pd\n"
            "import sys\n"
            f"sys.path.insert(0, '{os.path.abspath('.')}')\n"
            "from app.ml.models import MODEL_REGISTRY\n\n"
            f"df = pd.read_csv('{DATASET_ABS}')\n\n"
            "# --- fill these from the JSON decision ---\n"
            "model_key  = '<model>'          # e.g. 'kmeans'\n"
            "target_col = '<col or None>'    # None for clustering/anomaly\n"
            "features   = [<list or empty>]  # [] means use all numeric\n"
            "k          = 4                  # only used by kmeans\n"
            "# -----------------------------------------\n\n"
            "fn = MODEL_REGISTRY[model_key]\n\n"
            "# Route arguments based on model type\n"
            "if model_key == 'kmeans':\n"
            "    result = fn(df, features=features, k=k)\n"
            "elif model_key == 'isolation_forest':\n"
            "    result = fn(df, features=features if features else None)\n"
            "else:\n"
            "    result = fn(df, target_col=target_col)\n\n"
            "print('ML_RESULT:', result)\n"
            "```\n\n"
            "After the tool returns successfully, provide a short technical "
            "result note (1-2 sentences) and then say HANDOFF_TO_SUMMARY."
        ),
    )


def _build_ml_summary_agent(client) -> AssistantAgent:
    return AssistantAgent(
        name="ResultSummarizer",
        model_client=client,
        system_message=(
            "You are a user-facing ML explainer.\n\n"
            "Wait until you see HANDOFF_TO_SUMMARY. Then read the original user "
            "task plus the full MLAnalyst output (including ML_RESULT values) and "
            "produce a concise final answer for chat.\n\n"
            "Output rules (strict):\n"
            "- Use plain text only. No markdown bold, no tables, no code fences.\n"
            "- Keep total length between 6 and 10 lines.\n"
            "- Use this exact structure and labels:\n"
            "  Outcome: <one short sentence>\n"
            "  Highlights:\n"
            "  - <finding 1>\n"
            "  - <finding 2>\n"
            "  - <finding 3>\n"
            "  Recommendation: <one practical next step>\n"
            "- If relevant, include one short caution inside Highlights.\n"
            "- Focus on what the user can do next, not model internals.\n"
            "- Never include raw function-call traces.\n"
            "- Never include the word TERMINATE in the visible summary text.\n\n"
            "Keep it practical and aligned to the user's prompt intent.\n"
            "End with TERMINATE."
        ),
    )


async def run_ml_pipeline(task: str):
    """
    Standalone ML pipeline: ModelSelector → MLAnalyst.
    Called after EDA/cleaning is already complete.
    Yields AutoGen message objects for SSE streaming.
    """
    client = get_ollama_client()
    code_tool = get_code_execution_tool()

    selector = _build_selector_agent(client)
    analyst = _build_ml_analyst_agent(client, code_tool)
    summarizer = _build_ml_summary_agent(client)

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [selector, analyst, summarizer],
        termination_condition=termination,
    )

    async for message in team.run_stream(task=task):
        yield message


async def run_multi_agent_ml_pipeline(task: str):
    """
    Full pipeline: Planner → DataScientist (EDA+cleaning) → Reviewer
                   → ModelSelector → MLAnalyst.

    Yields AutoGen message objects for SSE streaming.
    """
    from autogen_agentchat.teams import SelectorGroupChat

    client = get_ollama_client()
    code_tool = get_code_execution_tool()

    # ---- existing agents (unchanged) ----
    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message=(
            "You are the Strategic Planner. Decompose the goal into steps.\n\n"
            "Steps:\n"
            "1. Data Loading.\n"
            "2. Data Cleaning.\n"
            "3. Exploratory Analysis & Visualisation.\n"
            "4. Hand off to ModelSelector for ML analysis.\n\n"
            "Provide the plan, then hand over to the DataScientist."
        ),
    )

    coder = AssistantAgent(
        name="DataScientist",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are an Expert Coder in Python and Pandas. Implement the Plan.\n\n"
            "Requirements:\n"
            "1. Write clean, efficient code using pandas.\n"
            f"2. For visualisations, save to '{WORKING_DIR}/'.\n"
            f"3. Load the dataset from {DATASET_ABS}.\n"
            "4. Use your tool to verify results.\n\n"
            "When EDA and cleaning are complete say CODE_APPROVED so the "
            "Reviewer can check, or fix issues if the Reviewer requests changes."
        ),
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=client,
        system_message=(
            "You are a Quality Assurance Specialist. Review the DataScientist's work.\n\n"
            "If EDA results are correct and charts are saved, say 'CODE_APPROVED' "
            "and hand off to ModelSelector.\n"
            "Otherwise, request specific fixes from DataScientist."
        ),
    )

    # ---- new ML agents ----
    selector = _build_selector_agent(client)
    analyst = _build_ml_analyst_agent(client, code_tool)
    summarizer = _build_ml_summary_agent(client)

    termination = TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [planner, coder, reviewer, selector, analyst, summarizer],
        model_client=client,
        termination_condition=termination,
    )

    async for message in team.run_stream(task=task):
        yield message


if __name__ == "__main__":
    async def _smoke_test():
        task = (
            "What factors most influence how many units a product sells "
            "in a month? Run the appropriate ML analysis."
        )
        async for msg in run_ml_pipeline(task):
            print(getattr(msg, "source", "sys"), ":",
                  str(getattr(msg, "content", msg))[:200])

    asyncio.run(_smoke_test())
