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
            "You are a Senior Data Analyst. You have a code execution tool. USE IT.\n\n"
            f"Dataset: {dataset_abs}\n"
            f"Working dir: {work_abs}/\n\n"
            "MANDATORY - EXECUTE CODE FIRST:\n"
            "1. Use the code tool to run: df = pd.read_csv() and print df.columns\n"
            "2. Use the code tool to analyze the data\n"
            "3. Use the code tool to create a chart and save with plt.savefig()\n"
            "4. ONLY after code execution, provide FINAL_ANSWER with results\n\n"
            "FORBIDDEN:\n"
            "- NEVER say data is unavailable without executing code first\n"
            "- NEVER write text instead of using the code tool\n"
            "- The code tool IS available - you MUST use it\n\n"
            "FINAL_ANSWER: <state findings with numbers>\nTERMINATE"
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
