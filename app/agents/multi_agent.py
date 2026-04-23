import asyncio
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from app.core.config import DEFAULT_DATASET_PATH, WORKING_DIR
from app.agents.base import (
    get_code_execution_tool,
    get_dependency_install_tool,
    get_ollama_client,
)
from app.core.config import DEFAULT_DATASET_PATH, WORKING_DIR


async def run_multi_agent_pipeline(
    task: str,
    dataset_path: str,
    artifact_dir: str | None = None,
):
    """
    Executes a data analytics task using the multi-agent framework.
    Yields events for streaming.
    """
    dataset_abs = os.path.abspath(dataset_path)
    artifact_abs = os.path.abspath(artifact_dir or WORKING_DIR)
    os.makedirs(artifact_abs, exist_ok=True)

    client = get_ollama_client()
    code_tool = get_code_execution_tool(work_dir=artifact_abs)
    dependency_tool = get_dependency_install_tool(work_dir=artifact_abs)

    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message=(
            "You are the Strategic Planner. Decompose the goal into steps.\n\n"
            "Steps should include:\n"
            "1. Data Loading.\n"
            "2. Data Cleaning.\n"
            "3. Research and visualization strategy.\n"
            "4. Analysis implementation.\n"
            "5. Validation.\n\n"
            "Do not write code. Hand over to ResearchAgent."
        ),
        ),
    researcher = AssistantAgent(
        name="ResearchAgent",
        model_client=client,
        system_message=(
            "You are a Data Research Specialist.\n"
            "Given the user's task and plan, provide:\n"
            "1) Key analytical questions to answer.\n"
            "2) Recommended plots that best support the solution.\n"
            "3) Required Python packages for the proposed analysis and plots.\n"
            "4) Potential risks (data quality, leakage, misleading visuals).\n\n"
            "5) Analysis.\n"
            "6) Viz.\n\n"
            "Output a concise checklist with explicit package names and chart types.\n"
            "Then hand over to DataScientist."
        ),
    )

    coder = AssistantAgent(
        name="DataScientist",
        model_client=client,
        tools=[dependency_tool, code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are an Expert Coder in Python and Pandas. Implement the Plan.\n\n"
            "Requirements:\n"
            "1. Write clean, efficient code using pandas.\n"
            f"2. Load the dataset from {dataset_abs}.\n"
            f"3. For visualizations, save the PNG to '{artifact_abs}/' "
            f"(e.g. plt.savefig('{artifact_abs}/chart_1.png')).\n"
            "4. IMPORTANT: After saving EACH PNG chart, also save a JSON sidecar with "
            "the SAME base filename (e.g. chart_1.png + chart_1.json) in the same directory. "
            'The JSON must contain: {"title": str, "chart_type": "bar"|"line"|"pie"|"scatter"|"histogram", '
            '"x_axis": {"label": str, "values": list}, "y_axis": {"label": str, "values": list}, '
            '"description": "one sentence describing what the chart shows"}. '
            f"Use Python: `import json; json.dump(data, open('{artifact_abs}/chart_1.json', 'w'))`.\n"
            "5. BEFORE running code, call `install_run_dependencies` with the exact "
            "Python script you will execute. This generates a per-run requirements file "
            "and installs dependencies.\n"
            "6. Execute code only after dependency installation succeeds.\n"
            "7. Use your execution tool to verify results.\n"
            "8. If dependency install or execution fails, fix and retry.\n\n"
            "Review loop contract:\n"
            "- If CodeReviewerAgent replies with 'APPROVED', stop making changes.\n"
            "- Otherwise, CodeReviewerAgent will provide up to 3 blocking fixes.\n"
            "- Address all listed blocking fixes in one revision and resubmit."
        ),
    )

    reviewer = AssistantAgent(
        name="CodeReviewerAgent",
        model_client=client,
        system_message=(
            "You are a rigorous code reviewer.\n"
            "Review the DataScientist output for:\n"
            "1) Runtime risks and logical bugs.\n"
            "2) Missing/incorrect package requirements.\n"
            "3) Alignment with ResearchAgent plot recommendations.\n"
            "4) Evidence that dependencies were installed before execution.\n"
            "5) Each saved PNG chart has a matching JSON sidecar with the same base filename.\n\n"
            "Output format is strict and must be one of:\n"
            "1) APPROVED\n"
            "2) BLOCKING_FIXES:\n"
            "- <fix 1>\n"
            "- <fix 2>\n"
            "- <fix 3>\n\n"
            "Rules for BLOCKING_FIXES:\n"
            "- Provide at most 3 items.\n"
            "- Each item must be concrete, blocking, and directly actionable.\n"
            "- Do not include non-blocking suggestions or conversational text.\n\n"
            "If APPROVED, also include 'TERMINATE'. If not approved, hand over to DataScientist."
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)
    selector_prompt = (
        "You manage a data-science team with roles: {roles}.\n"
        "Pick exactly one next role from {participants}.\n\n"
        "Workflow order:\n"
        "- Start with Planner.\n"
        "- Then ResearchAgent.\n"
        "- Then DataScientist.\n"
        "- Then CodeReviewerAgent.\n"
        "- If reviewer reports issues, send back to DataScientist.\n"
        "- Continue coder/reviewer loop until approval or termination.\n\n"
        "Conversation:\n{history}\n\n"
        "Return only one role name."
    )
    team = SelectorGroupChat(
        [planner, researcher, coder, reviewer],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
    )

    async for message in team.run_stream(task=task):
        yield message


if __name__ == "__main__":

    async def main():
        async for msg in run_multi_agent_pipeline(
            "Do a category analysis.", DEFAULT_DATASET_PATH
        ):
            print(msg)

    asyncio.run(main())
