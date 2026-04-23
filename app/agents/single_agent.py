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
    work_abs = os.path.abspath(artifact_dir or WORKING_DIR)
    os.makedirs(work_abs, exist_ok=True)

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
            "3. For charts, use matplotlib or seaborn and save files into the artifacts directory above "
            f"(e.g. plt.savefig('{work_abs}/chart_1.png')).\n"
            "4. IMPORTANT: After saving EACH PNG chart, also save a JSON sidecar with the SAME base filename "
            "(e.g. chart_1.png + chart_1.json) in the same directory. "
            'The JSON must contain: {"title": str, "chart_type": "bar"|"line"|"pie"|"scatter"|"histogram", '
            '"x_axis": {"label": str, "values": list}, "y_axis": {"label": str, "values": list}, '
            '"description": "one sentence describing what the chart shows"}. '
            "Use Python: `import json; json.dump(data, open('"
            + work_abs
            + "/chart_1.json', 'w'))`. "
            "This sidecar is REQUIRED so downstream verification and chart Q&A can work.\n"
            "5. Run your code with the provided tool and fix issues until results are sensible.\n"
            "6. Provide only the direct answer to the user question as: 'FINAL_ANSWER: <answer>'.\n"
            "7. Do not include workflow steps or tool traces in FINAL_ANSWER.\n"
            "8. End your reply with the word TERMINATE "
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
