import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DEFAULT_DATASET_PATH, WORKING_DIR
from app.agents.base import get_ollama_client, get_code_execution_tool

async def run_qa_pipeline(question: str, dataset_path: str):
    """
    Executes a Q&A analysis using the DataConsultant agent.
    Focuses on understanding the dataset schema and statistics.
    """
    # 1. Setup Client and Tools
    client = get_ollama_client()
    code_tool = get_code_execution_tool()
    
    # 2. Define the DataConsultant Agent
    consultant = AssistantAgent(
        name="DataConsultant",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are a Data Strategy Consultant. Your goal is to answer questions "
            "about the dataset located at " + os.path.abspath(dataset_path) + ".\n\n"
            "Responsibilities:\n"
            "1. Explain column meanings based on the data.\n"
            "2. Provide descriptive statistics (means, ranges, counts).\n"
            "3. Identify data quality issues (missing values, types).\n"
            "Always use your tool to verify data facts before answering.\n"
            "Return only the direct answer as: 'FINAL_ANSWER: <answer>'.\n"
            "Do not include process trail, numbered internal steps, or tool-call traces.\n"
            "When finished answering the user's question, say 'TERMINATE'."
        )
    )
    
    # 3. Setup Team
    termination = TextMentionTermination("TERMINATE", sources=["DataConsultant"])
    team = RoundRobinGroupChat([consultant], termination_condition=termination)
    
    # 4. Yield task stream
    async for message in team.run_stream(task=question):
        yield message

if __name__ == "__main__":
    async def main():
        async for msg in run_qa_pipeline("What are the columns in this dataset?", DEFAULT_DATASET_PATH):
            print(msg)
    asyncio.run(main())
