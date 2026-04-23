"""Single-agent HumanEval runner that wraps the *actual* single-agent pipeline.

This runner mirrors ``app/agents/single_agent.py``: one ``AssistantAgent``
(the "Analyst") driven by a ``RoundRobinGroupChat``, tool-enabled for Python
execution, terminating on ``TERMINATE``. System prompts are swapped for
HumanEval-appropriate ones (no CSV loading, no chart saving) so the baseline
is compared on function-body completion rather than end-to-end analytics
scaffolding.

Differences vs the old one-shot direct-Ollama baseline:
  * Uses AutoGen's ``RoundRobinGroupChat`` + ``SimpleOllamaClient``, same as
    the production backend's baseline.
  * Analyst has a ``PythonCodeExecutionTool`` so it can iterate on its code
    mid-run (plan -> try -> fix), which is the whole point of the baseline
    loop described in Section 4.1 of the proposal.
  * ``interaction_count`` counts every Analyst TextMessage in the stream.

Termination: the Analyst emits ``TERMINATE`` (caught by a
``TextMentionTermination``), or we hit ``max_messages``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from app.agents.base import get_ollama_client
from humaneval.code_extraction import extract_function_body


ANALYST_SYS = (
    "You are the Baseline pipeline: a single Senior Data Analyst working "
    "alone on small Python coding tasks. You will receive a function "
    "signature and docstring. Plan briefly in your head, then implement "
    "and verify the function.\n\n"
    "Ground rules:\n"
    "- The function operates ONLY on arguments passed to it. Do NOT read "
    "CSV files, make network calls, or touch the filesystem.\n"
    "- Do NOT attempt to ``pip install`` anything; pandas and the standard "
    "library are available.\n\n"
    "Output contract:\n"
    "1. When you have a working implementation, emit the FULL function "
    "(``def`` signature AND body) inside a single ```python``` fenced block.\n"
    "2. On the same message, after the fenced block, end your reply with "
    "the literal word TERMINATE so the harness stops.\n"
    "3. Do NOT repeat the docstring. Do NOT include anything outside the "
    "fenced block other than the final TERMINATE keyword."
)


@dataclass
class SingleRunResult:
    completion: str
    raw_response: str
    llm_latency_sec: float
    interaction_count: int
    messages: list = field(default_factory=list)


def _build_user_task(task):
    return (
        "Implement the Python function below so that it passes its docstring "
        "specification. Return the final implementation as a full ``def`` "
        "in a single ```python``` fenced block, followed by TERMINATE.\n\n"
        + task["prompt"]
    )


def _message_text(msg):
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    try:
        return "\n".join(str(c) for c in content)
    except TypeError:
        return str(content)


def _pick_body(analyst_outputs, entry_point):
    """Walk Analyst text messages newest-first; return first that extracts."""
    for raw in reversed(analyst_outputs):
        body = extract_function_body(raw, entry_point)
        stripped = body.strip()
        if stripped and stripped != "pass":
            return body, raw
    if analyst_outputs:
        fallback = extract_function_body(analyst_outputs[-1], entry_point)
        return fallback, analyst_outputs[-1]
    return "    pass\n", ""


async def run_single_async(task, *, max_messages=12):
    client = get_ollama_client()
    # Note: the code execution tool is intentionally NOT attached here.
    # HumanEval-style tasks are function-body completions on small synthetic
    # inputs -- the tool adds no signal. Keeping it exposed the agent to
    # AutoGen's "Reflect on tool use produced no valid text response"
    # hard-error when glm-5 emits an empty reflection response. The
    # production baseline uses tools for real dataset analysis, which is a
    # different regime than HumanEval.

    analyst = AssistantAgent(
        name="Analyst",
        model_client=client,
        system_message=ANALYST_SYS,
    )

    termination = (
        TextMentionTermination("TERMINATE", sources=["Analyst"])
        | MaxMessageTermination(max_messages)
    )
    team = RoundRobinGroupChat(
        [analyst],
        name="BaselineSingleAgent",
        termination_condition=termination,
        max_turns=max_messages,
    )

    messages = []
    analyst_outputs = []
    interaction_count = 0

    start = time.perf_counter()
    async for msg in team.run_stream(task=_build_user_task(task)):
        if isinstance(msg, TaskResult):
            continue

        source = getattr(msg, "source", "") or ""
        text = _message_text(msg)
        messages.append(
            {"source": source, "type": type(msg).__name__, "content": text}
        )

        if source == "Analyst" and isinstance(msg, TextMessage):
            interaction_count += 1
            analyst_outputs.append(text)

    latency = time.perf_counter() - start
    completion, raw = _pick_body(analyst_outputs, task["entry_point"])

    return SingleRunResult(
        completion=completion,
        raw_response=raw,
        llm_latency_sec=latency,
        interaction_count=interaction_count,
        messages=messages,
    )


def run_single(task, **kwargs):
    """Synchronous entry point for the harness."""
    return asyncio.run(run_single_async(task, **kwargs))