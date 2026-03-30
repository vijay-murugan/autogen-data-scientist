import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool

async def run_single_agent_pipeline(task: str):
    """
    Executes a data analytics task using a single-agent baseline.
    Yields messages as they occur.
    """
    # 1. Setup Client and Tools
    client = get_ollama_client()
    code_tool = get_code_execution_tool()
    
    # 2. Define the Analyst Agent
    analyst = AssistantAgent(
        name="Analyst",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are a Senior Data Analyst. Your task is to solve problems "
            "using Python. You have a dataset at " + os.path.abspath(DATASET_PATH) + ".\n\n"
            "Requirements:\n"
            "1. Write clean Python code using pandas.\n"
            "2. For visualizations, save to '" + WORKING_DIR + "/'.\n"
            "3. Use your tool to verify results.\n"
            "After verifying, say 'TERMINATE'."
        )
    )
    
    # 3. Setup Team
    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([analyst], termination_condition=termination)
    
    # 4. Yield task stream
    async for message in team.run_stream(task=task):
        yield message

if __name__ == "__main__":
    async def main():
        async for msg in run_single_agent_pipeline("Show first 5 rows."):
            print(msg)
    asyncio.run(main())
