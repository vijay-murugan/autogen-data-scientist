import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool


async def run_multi_agent_pipeline(task: str):
    """
    Executes a data analytics task using the multi-agent framework.
    Yields events for streaming.
    """
    # 1. Setup Client and Tools
    client = get_ollama_client()
    code_tool = get_code_execution_tool()

    # 2. Define the Specialized Agents (Tool-Enabled)

    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message=(
            "You are the Strategic Planner. Decompose the goal into steps.\n\n"
            "Steps should include:\n"
            "1. Data Loading.\n"
            "2. Data Cleaning.\n"
            "3. Analysis.\n"
            "4. Viz.\n\n"
            "Provide the plan and handover to the DataScientist."
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
            "2. For visualizations, save the PNG to '"
            + WORKING_DIR
            + "/' (e.g. plt.savefig('"
            + WORKING_DIR
            + "/chart_1.png')).\n"
            "3. IMPORTANT: After saving each PNG chart, also save a JSON sidecar with the SAME base filename (e.g. chart_1.png + chart_1.json) in the same directory. "
            'The JSON must contain: {"title": str, "chart_type": "bar"|"line"|"pie"|"scatter"|"histogram", '
            '"x_axis": {"label": str, "values": list}, "y_axis": {"label": str, "values": list}, '
            '"description": "one sentence describing what the chart shows"}. '
            "Use Python: `import json; json.dump(data, open('"
            + WORKING_DIR
            + "/chart_1.json', 'w'))`.\n"
            "4. Load the dataset from " + os.path.abspath(DATASET_PATH) + ".\n"
            "5. Use your tool to verify results.\n\n"
            "If the Reviewer gives feedback, fix the code and resubmit."
        ),
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=client,
        system_message=(
            "You are a Quality Assurance Specialist. Review the DataScientist's code.\n\n"
            "If the results are correct and charts saved, say 'CODE_APPROVED' and 'TERMINATE'.\n"
            "Otherwise, request specific changes."
        ),
    )

    # 3. Setup the Team
    termination = TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [planner, coder, reviewer],
        model_client=client,
        termination_condition=termination,
    )

    # 4. Yield task stream
    async for message in team.run_stream(task=task):
        yield message


if __name__ == "__main__":

    async def main():
        async for msg in run_multi_agent_pipeline("Do a category analysis."):
            print(msg)

    asyncio.run(main())
