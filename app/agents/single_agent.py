import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DEFAULT_DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool

# Baseline: one agent plans, codes, and checks results—no separate Planner/Reviewer.
BASELINE_MAX_TURNS = 40


async def run_single_agent_pipeline(task: str, dataset_path: str):
    """
    Executes a data analytics task using a single-agent baseline.
    One AssistantAgent handles the full loop (load → clean → analyze → visualize).
    Yields messages as they occur.
    """
    client = get_ollama_client()
    code_tool = get_code_execution_tool()
    dataset_abs = os.path.abspath(dataset_path)
    work_abs = os.path.abspath(WORKING_DIR)

    analyst = AssistantAgent(
        name="Analyst",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are the Baseline pipeline: a single Senior Data Analyst.\n"
            "Unlike a multi-agent team, you work alone—plan briefly, then implement.\n\n"
            f"Dataset CSV path: {dataset_abs}\n"
            f"Save any figures to the directory: {work_abs}/\n\n"
            "Workflow:\n"
            "1. Load the CSV with pandas (handle dtypes and missing values as needed).\n"
            "2. Perform the analysis the user asked for.\n"
            "3. For charts, use matplotlib or seaborn and save files into the artifacts directory above.\n"
            "4. Run your code with the provided tool and fix issues until results are sensible.\n"
            "5. Summarize findings for the user, then end your reply with the word TERMINATE "
            "when you are fully done.\n"
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
