import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DEFAULT_DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool

# Baseline: one agent plans, codes, and checks results—no separate Planner/Reviewer.
BASELINE_MAX_TURNS = 40


async def run_single_agent_pipeline(
    task: str,
    dataset_path: str,
    artifact_dir: str | None = None,
):
    """
    Executes a data analytics task using a single-agent baseline.
    One AssistantAgent handles the full loop (load → clean → analyze → visualize).
    Yields messages as they occur.

    Parameters
    ----------
    artifact_dir
        Directory for saved figures and code-executor cwd. Defaults to WORKING_DIR.
    """
    client = get_ollama_client()
    work_abs = os.path.abspath(artifact_dir or WORKING_DIR)
    os.makedirs(work_abs, exist_ok=True)
    code_tool = get_code_execution_tool(work_dir=work_abs)
    if not dataset_path:
        dataset_path = DEFAULT_DATASET_PATH

    # 2. Define the Analyst Agent
    dataset_abs = os.path.abspath(dataset_path)

    analyst = AssistantAgent(
        name="Analyst",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are a Senior Data Analyst. Your job is to analyze data and provide clear, natural language answers.\n\n"
            f"Dataset CSV path: {dataset_abs}\n"
            f"Save charts to: {work_abs}/\n\n"
            "CRITICAL FIRST STEP - ALWAYS DO THIS:\n"
            "1. Load the CSV and print df.columns to see what columns actually exist\n"
            "2. Use ONLY the columns that are present - never assume column names\n"
            "3. Map user questions to available columns (e.g., 'purchased_last_month' = sales quantity)\n\n"
            "INTERNAL WORKFLOW (do NOT mention in answer):\n"
            "- Analyze the actual CSV columns and data\n"
            "- Create visualizations using available columns\n"
            "- Save charts and JSON sidecars\n\n"
            "OUTPUT RULES - FOLLOW STRICTLY:\n"
            "- FINAL_ANSWER must be ONLY natural language response\n"
            "- NO procedure text, NO 'step 1', NO 'I loaded'\n"
            "- NEVER say data is unavailable - work with what exists\n"
            "- State findings directly with specific numbers\n\n"
            "WRONG: 'The required sales data is not available'\n"
            "RIGHT: 'The top product is iPhone 15 with 45,230 units purchased last month'\n\n"
            "FINAL_ANSWER: <your answer>\n"
            "TERMINATE"
        ),
    )

    termination = TextMentionTermination("TERMINATE", sources=["Analyst"])
    team = RoundRobinGroupChat(
        [analyst],
        name="BaselineSingleAgent",
        termination_condition=termination,
        max_turns=BASELINE_MAX_TURNS,
    )

    async for message in team.run_stream(task=task):
        yield message


if __name__ == "__main__":

    async def main():
        async for msg in run_single_agent_pipeline("Show first 5 rows.", DEFAULT_DATASET_PATH):
            print(msg)

    asyncio.run(main())
