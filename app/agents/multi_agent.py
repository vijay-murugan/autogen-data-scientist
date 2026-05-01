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
    )
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
            "You are an Expert Coder. Your ONLY job is to execute Python code using the provided code tool.\n\n"
            f"Dataset path: {dataset_abs}\n"
            f"Working directory: {artifact_abs}/\n\n"
            "REQUIRED - USE CODE TOOL IMMEDIATELY:\n"
            "1. Call the code tool to execute: df = pd.read_csv()\n"
            "2. Call the code tool to print: df.columns and df.head()\n"
            "3. Call the code tool to analyze and find the answer\n"
            "4. Call the code tool to create and save a chart with plt.savefig()\n"
            "5. DO NOT RESPOND WITH TEXT - ONLY EXECUTE CODE\n\n"
            "FORBIDDEN:\n"
            "- NEVER say 'data unavailable' without running code first\n"
            "- NEVER write text explanations - ONLY execute Python code\n"
            "- Code tool is available - USE IT for every step\n\n"
            "Your output should be code execution results, not chat."
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
            "If APPROVED, hand over to ResultSummarizer. If not approved, hand over to DataScientist."
        ),
    )

    summarizer = AssistantAgent(
        name="ResultSummarizer",
        model_client=client,
        system_message=(
            "You are the Final Result Summarizer. State the answer directly.\n\n"
            "FORBIDDEN:\n"
            "- NEVER say data is unavailable, missing, or not loaded\n"
            "- NEVER say 'If you run', 'view the outputs', 'follow these steps'\n"
            "- NEVER tell user HOW to get answer - just GIVE the answer\n"
            "- NEVER mention agents, workflows, or tools\n\n"
            "REQUIRED:\n"
            "- State findings with specific numbers from the analysis\n"
            "- Answer as if you personally did the analysis\n"
            "- If analysis failed, say what analysis was attempted\n\n"
            "FINAL_ANSWER: <your answer>\nTERMINATE"
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(35)
    selector_prompt = (
        "You manage a data-science team with roles: {roles}.\n"
        "Pick exactly one next role from {participants}.\n\n"
        "Workflow order:\n"
        "- Start with Planner.\n"
        "- Then ResearchAgent.\n"
        "- Then DataScientist.\n"
        "- Then CodeReviewerAgent.\n"
        "- If reviewer reports issues, send back to DataScientist.\n"
        "- If reviewer approves, hand over to ResultSummarizer.\n"
        "- ResultSummarizer always runs last after approval.\n\n"
        "Conversation:\n{history}\n\n"
        "Return only one role name."
    )
    team = SelectorGroupChat(
        [planner, researcher, coder, reviewer, summarizer],
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
