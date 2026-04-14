import os
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from app.core.config import DATASET_PATH
from app.agents.base import get_ollama_client, get_code_execution_tool


async def run_verify_chart_pipeline(chart_metadata: dict):
    """
    Independently re-computes the values for a generated chart from the raw dataset
    and compares them to the chart's JSON sidecar. Returns a verdict dict:
        {"status": "PASS" | "WARN" | "FAIL" | "UNKNOWN", "details": str, "log": str}
    """

    client = get_ollama_client()
    code_tool = get_code_execution_tool()

    verifier = AssistantAgent(
        name="Verifier",
        model_client=client,
        tools=[code_tool],
        reflect_on_tool_use=False,
        system_message=(
            "You are a Data Verification Specialist. Your job is to independently "
            "verify whether a chart's underlying data is correct.\n\n"
            f"Dataset absolute path: {os.path.abspath(DATASET_PATH)}\n\n"
            "Chart metadata to verify (the JSON sidecar the DataScientist produced):\n"
            f"{json.dumps(chart_metadata, indent=2)}\n\n"
            "Procedure:\n"
            "1. Load the dataset with pandas.\n"
            "2. Read the chart's title, description, x_axis label, and y_axis label. "
            "From the SEMANTICS of those fields (NOT the reported values), write pandas code that "
            "re-computes what the chart should show.\n"
            "3. Execute the code. Print both your computed (label, value) pairs AND the chart's reported pairs.\n"
            "4. Compare them: labels must match exactly (case-insensitive, trimmed). Numeric values must agree "
            "within a relative error of 1% (or absolute error of 0.01 for small values).\n"
            "5. On the LAST LINE of your final message, emit the verdict in this EXACT format "
            "(no other text on that line):\n"
            "VERDICT::PASS::<short explanation> | VERDICT::WARN::<explanation> | VERDICT::FAIL::<explanation>\n"
            "Use PASS if everything matches, WARN if there are minor discrepancies (<5% off, or partial label mismatch), "
            "FAIL if the chart clearly misrepresents the data.\n"
            "After the verdict line, say TERMINATE."
        ),
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([verifier], termination_condition=termination)

    verdict = None
    log_pieces = []

    async for message in team.run_stream(
        task="Please verify this chart per the procedure in your system message."
    ):
        content = getattr(message, "content", "")
        if isinstance(content, str):
            log_pieces.append(f"[{getattr(message, 'source', '?')}] {content}")
            if "VERDICT::" in content:
                # Grab the last occurrence in the message
                lines = [l for l in content.splitlines() if "VERDICT::" in l]
                if lines:
                    raw = lines[-1].split("VERDICT::", 1)[1]
                    parts = raw.split("::", 1)
                    status = parts[0].strip().upper()
                    details = parts[1].strip() if len(parts) > 1 else ""
                    if status in ("PASS", "WARN", "FAIL"):
                        verdict = {"status": status, "details": details}

    if verdict is None:
        verdict = {
            "status": "UNKNOWN",
            "details": "Verifier did not produce a parseable verdict. Try Re-verify.",
        }
    verdict["log"] = "\n\n".join(log_pieces)
    return verdict
