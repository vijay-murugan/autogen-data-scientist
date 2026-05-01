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
            "You are an Expert Coder in Python and Pandas. Execute the analysis plan internally.\n\n"
            f"Dataset: {dataset_abs}\n"
            f"Save charts to: {artifact_abs}/\n\n"
            "CRITICAL FIRST STEP:\n"
            "1. Load CSV and check df.columns - use ONLY columns that exist\n"
            "2. Map user questions to available columns (e.g., 'purchased_last_month' = sales)\n"
            "3. NEVER say data is unavailable - work with what exists\n\n"
            "TECHNICAL EXECUTION (silent):\n"
            "- Execute code with install_run_dependencies first\n"
            "- Save charts: plt.savefig() then plt.close()\n"
            "- Save JSON sidecars\n\n"
            "FINAL ANSWER RULES:\n"
            "- ONLY natural language answer, NO procedure text\n"
            "- NEVER say 'If you view', 'the chart shows', 'data unavailable'\n"
            "- STATE ACTUAL FINDINGS with specific numbers\n\n"
            "WRONG: 'The required data is not available'\n"
            "RIGHT: 'iPhone 15 leads with 45,230 units sold'\n\n"
            "Start with: FINAL_ANSWER: <your answer>\n\n"
            "Stop when CodeReviewerAgent says APPROVED."
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
            "You are the Final Result Summarizer. State the answer directly from the analysis results.\n\n"
            "FORBIDDEN - NEVER DO THESE:\n"
            "- NEVER say 'If you run', 'view the outputs', 'follow these steps', 'the chart will show'\n"
            "- NEVER tell the user HOW to get the answer - just GIVE the answer\n"
            "- NEVER mention agents, workflows, or tools used\n"
            "- NEVER use 'We found', 'The team analyzed', 'The data shows'\n\n"
            "REQUIRED - ALWAYS DO THESE:\n"
            "- State findings directly with specific numbers and insights\n"
            "- Use natural language only, business-friendly tone\n"
            "- Answer the question as if you personally did the analysis\n\n"
            "EXAMPLE:\n"
            "WRONG: 'If you view the chart titles, you will see that Electronics is the top seller'\n"
            "RIGHT: 'Electronics is the highest-selling category with 12,450 units sold, representing 45% of total sales.'\n\n"
            "State the final answer clearly, then end with TERMINATE."
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
