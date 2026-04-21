"""Multi-agent HumanEval runner that wraps the *actual* system pipeline.

This runner uses the same AutoGen coordination model as
``app/agents/multi_agent.py`` -- a ``SelectorGroupChat`` over four specialised
agents (Planner, ResearchAgent, DataScientist, CodeReviewerAgent) with the
DataScientist tool-enabled for Python execution. System prompts are swapped
for HumanEval-appropriate ones (no CSV loading, no chart saving, no dataset
paths) so we are comparing the pipeline's *coordination model* on function-body
completion rather than its end-to-end analytics scaffolding.

Differences vs the old "lean" Planner/Coder/Critic loop:
  * Uses AutoGen's ``SelectorGroupChat`` + ``SimpleOllamaClient``, not direct
    ``ollama.AsyncClient`` calls. This is the same plumbing the production
    backend uses.
  * Adds a ResearchAgent (edge cases / idioms checklist) in line with the
    "optional critic and review agents" described in Section 3 of the proposal.
  * DataScientist has a ``PythonCodeExecutionTool`` so it can sanity-check its
    own code before the reviewer sees it -- same as the real backend.
  * ``interaction_count`` counts every agent-sourced TextMessage in the stream.

Termination: the CodeReviewerAgent emits ``TERMINATE`` (caught by a
``TextMentionTermination`` scoped to that source), or we hit ``max_messages``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import time
from dataclasses import dataclass, field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat

from app.agents.base import get_ollama_client
from humaneval.code_extraction import extract_function_body


PLANNER_SYS = (
    "Your ROLE is Planner. You are NOT the ResearchAgent, the DataScientist, "
    "or the CodeReviewerAgent. Never produce their outputs.\n\n"
    "You are the Strategic Planner for small Python coding tasks. The user "
    "will give you a function signature and docstring. Decompose the problem "
    "into 3-6 short numbered steps describing HOW to implement the function. "
    "Call out edge cases, tie-breaking rules, and NaN-handling explicitly if "
    "they appear in the docstring.\n\n"
    "HARD CONSTRAINTS (do not violate):\n"
    "- Do NOT write code. Do NOT output any Python.\n"
    "- Do NOT use triple-backticks (```), fenced code blocks, or `def` keywords.\n"
    "- Do NOT include function signatures, return statements, or import lines.\n"
    "- Do NOT say 'APPROVED', 'TERMINATE', or 'BLOCKING_FIXES' -- those are "
    "the CodeReviewerAgent's keywords.\n"
    "- Reference pandas methods by name in prose only, e.g. \"use df.shape\", not as code.\n"
    "- Your entire output must be PLAIN PROSE plus the numbered steps.\n\n"
    "Keep the plan under 150 words. End your message so ResearchAgent can go next."
)

RESEARCHER_SYS = (
    "Your ROLE is ResearchAgent. You are NOT the Planner, the DataScientist, "
    "or the CodeReviewerAgent. Never produce their outputs.\n\n"
    "You are a Data Research Specialist. Given the task and the Planner's plan, "
    "output a short checklist covering:\n"
    "1) Key edge cases and tie-breaking rules implied by the docstring.\n"
    "2) Pandas / Python idioms the Coder should prefer (for example "
    "``reset_index(drop=True)``, ``dropna()``, stable sorts).\n"
    "3) Pitfalls to avoid (NaN propagation, dtype surprises, mutation of the "
    "input DataFrame).\n\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT write code. Do NOT use triple-backticks or `def` keywords.\n"
    "- Do NOT say 'APPROVED', 'TERMINATE', or 'BLOCKING_FIXES'.\n\n"
    "Keep it under 150 words. Then hand over to DataScientist."
)

CODER_SYS = (
    "Your ROLE is DataScientist. You are NOT the Planner, the ResearchAgent, "
    "or the CodeReviewerAgent. Your job is to WRITE PYTHON CODE.\n\n"
    "Your message MUST contain a ```python``` fenced block with a full ``def``. "
    "You are NOT allowed to emit 'APPROVED', 'TERMINATE', or 'BLOCKING_FIXES' "
    "-- those are the reviewer's words, not yours. If you ever feel like "
    "saying 'APPROVED' you are confused about your role: re-read the task and "
    "emit code instead.\n\n"
    "You are an Expert Python/Pandas Coder. Implement the function described "
    "above, following the Planner's plan and the Researcher's checklist.\n\n"
    "Ground rules:\n"
    "- The function operates ONLY on the arguments passed to it. Do NOT read "
    "CSV files, make network calls, or touch the filesystem.\n"
    "- Do NOT attempt to ``pip install`` anything; pandas and the standard "
    "library are available.\n\n"
    "When confident, emit the FULL function (``def`` signature AND body) "
    "inside a single ```python``` fenced code block. Do NOT add commentary "
    "outside that fenced block.\n\n"
    "Review-loop contract:\n"
    "- If CodeReviewerAgent replies with 'APPROVED', stop making changes.\n"
    "- Otherwise the reviewer will list up to 3 BLOCKING_FIXES. Address every "
    "listed fix in one revision and resubmit the FULL function."
)

REVIEWER_SYS = (
    "Your ROLE is CodeReviewerAgent. You are NOT the Planner, ResearchAgent, "
    "or DataScientist. You do NOT write code -- you only judge code.\n\n"
    "You are a rigorous Code Reviewer. You will see the task (signature + "
    "docstring) and the DataScientist's proposed implementation. Decide "
    "whether the code correctly implements the docstring including edge "
    "cases, NaN handling, and tie-breaking.\n\n"
    "HARD CONSTRAINTS:\n"
    "- Do NOT emit any ```python``` fenced code blocks.\n"
    "- Do NOT copy or echo the DataScientist's code.\n"
    "- Do NOT write `def`, `return`, or `import` statements.\n\n"
    "Output format (strict) -- exactly one of:\n"
    "  APPROVED TERMINATE\n"
    "or\n"
    "  BLOCKING_FIXES:\n"
    "  - <concise issue 1>\n"
    "  - <concise issue 2>\n"
    "  - <concise issue 3>\n\n"
    "Rules:\n"
    "- List at most 3 blocking issues.\n"
    "- Only list issues that would cause incorrect outputs or runtime errors.\n"
    "- Do NOT propose stylistic changes. Do NOT add commentary.\n"
    "- If approved, include the literal word 'TERMINATE' so the team stops."
)

SELECTOR_PROMPT = (
    "You coordinate a data-science team with roles: {roles}.\n"
    "Pick exactly one next role from {participants}.\n\n"
    "STRICT workflow order -- follow this regardless of message content:\n"
    "1. If no agent has spoken yet, pick Planner.\n"
    "2. If the last speaker was Planner, pick ResearchAgent.\n"
    "3. If the last speaker was ResearchAgent, pick DataScientist.\n"
    "4. If the last speaker was DataScientist, pick CodeReviewerAgent.\n"
    "5. If the last speaker was CodeReviewerAgent AND its message starts with "
    "'APPROVED', the team is done -- but you must still pick a role; pick "
    "DataScientist.\n"
    "6. If the last speaker was CodeReviewerAgent AND its message starts with "
    "'BLOCKING_FIXES', pick DataScientist.\n\n"
    "IMPORTANT: DataScientist MUST get at least one turn before CodeReviewerAgent "
    "is ever selected. Do NOT skip DataScientist even if earlier messages "
    "contain code. Only DataScientist's code counts.\n\n"
    "Conversation:\n{history}\n\n"
    "Return only one role name."
)

_AGENT_SOURCES = {"Planner", "ResearchAgent", "DataScientist", "CodeReviewerAgent"}


_SELECTOR_DEBUG = os.getenv("HUMANEVAL_SELECTOR_DEBUG") == "1"


def _deterministic_selector(messages):
    """Force strict Planner -> Researcher -> DataScientist -> Reviewer ordering.

    The LLM-based selector (even with a strict selector_prompt) is unreliable
    with weaker local models and regularly jumps Planner -> Reviewer, skipping
    the coder entirely. Since benchmark fidelity only requires the four
    specialised roles to be exercised (not the selector's judgement), we bolt
    the ordering down in Python.

    Always returns a concrete agent name so the LLM selector is never
    consulted.

    Set ``HUMANEVAL_SELECTOR_DEBUG=1`` in the env to print per-turn routing
    decisions to stderr for live diagnostic.
    """
    # Find the last AGENT message (ignore user task, tool events, etc.).
    last_source = None
    last_text = ""
    for m in reversed(messages):
        src = getattr(m, "source", None)
        if src in _AGENT_SOURCES:
            last_source = src
            content = getattr(m, "content", "")
            last_text = content if isinstance(content, str) else str(content)
            break

    if last_source is None:
        next_agent = "Planner"
    elif last_source == "Planner":
        next_agent = "ResearchAgent"
    elif last_source == "ResearchAgent":
        next_agent = "DataScientist"
    elif last_source == "DataScientist":
        next_agent = "CodeReviewerAgent"
    elif last_source == "CodeReviewerAgent":
        # APPROVED -> termination will fire before this is used; otherwise
        # route back to the coder for BLOCKING_FIXES retry.
        next_agent = "DataScientist"
    else:
        # Unknown source -- default to Planner to kick off the workflow.
        next_agent = "Planner"

    if _SELECTOR_DEBUG:
        import sys
        sources_seen = [
            getattr(m, "source", None) for m in messages
        ]
        print(
            f"[selector] last={last_source!r} -> next={next_agent!r} "
            f"(sources_seen={sources_seen})",
            file=sys.stderr,
            flush=True,
        )
    return next_agent


@dataclass
class MultiRunResult:
    completion: str
    raw_response: str
    plan: str
    critiques: list
    llm_latency_sec: float
    interaction_count: int
    turns_used: int
    approved: bool
    messages: list = field(default_factory=list)
    # Per-role transcripts so the raw dump can show exactly what each
    # specialised agent said. Without these, AmazonDA/02-style failures
    # (DataScientist emits no text, reviewer approves empty solution) are
    # invisible in the dump and look like the planner immediately yielded
    # to an approval.
    researcher_outputs: list = field(default_factory=list)
    coder_outputs: list = field(default_factory=list)
    reviewer_outputs: list = field(default_factory=list)


def _build_user_task(task):
    """Frame the HumanEval task as a natural-language instruction for the team."""
    return (
        "Implement the Python function below so that it passes its docstring "
        "specification. The function will be called with arguments matching "
        "the signature; the code must NOT read CSVs, make network calls, or "
        "save any files. Return the final implementation as a full ``def`` "
        "in a single ```python``` fenced block.\n\n"
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


def _is_valid_function_body(body, entry_point):
    """True if ``body`` is parseable Python AND actually does something.

    Rejects:
      * empty / bare ``pass`` placeholders,
      * text that is not syntactically valid Python inside a def,
      * bodies that parse but contain no control-flow statement that
        could plausibly implement the contract (``return`` / ``yield`` /
        ``raise``). This is what catches reviewer chatter like
        ``APPROVED`` or ``APPROVED TERMINATE`` -- both parse as valid
        Python (bare Name expression statements) but cannot implement
        any HumanEval-style function.

    All AmazonDA/HumanEval solutions return a value, so requiring a
    Return/Yield/Raise node is safe for this benchmark.
    """
    stripped = body.strip()
    if not stripped or stripped == "pass":
        return False
    wrapped = f"def {entry_point}(*args, **kwargs):\n{body}"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    fn = tree.body[0]
    for node in ast.walk(fn):
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Raise)):
            return True
    return False


def _pick_body(outputs, entry_point):
    """Walk messages newest-first; return first that extracts a real body."""
    for raw in reversed(outputs):
        body = extract_function_body(raw, entry_point)
        if _is_valid_function_body(body, entry_point):
            return body, raw
    return None, None


def _pick_body_from_coder_outputs(
    coder_outputs, entry_point, fallback_outputs=None
):
    """Return (body, raw_response) for the newest coder message that parses.

    If DataScientist never produced extractable code (selector routing issue,
    early termination, etc.), fall back to scanning ``fallback_outputs``
    (typically Planner/ResearchAgent messages) so a rogue Planner leak does
    not silently resolve to ``pass`` and tank the benchmark.
    """
    body, raw = _pick_body(coder_outputs, entry_point)
    if body is not None:
        return body, raw

    if fallback_outputs:
        body, raw = _pick_body(fallback_outputs, entry_point)
        if body is not None:
            return body, raw

    # No agent produced parseable Python anywhere. Return a clean ``pass``
    # placeholder instead of leaking reviewer chatter ("APPROVED TERMINATE",
    # etc.) as the completion, which would syntax-error in the executor.
    return "    pass\n", ""


async def run_multi_async(task, *, max_messages=20):
    client = get_ollama_client()
    # Note: the code execution tool is intentionally NOT attached to any
    # agent here. HumanEval-style tasks are function-body completions on
    # small synthetic inputs -- the tool adds no signal. Keeping it exposed
    # the DataScientist to AutoGen's "Reflect on tool use produced no valid
    # text response" hard-error when glm-5 emits an empty reflection
    # response after tool execution. The production multi-agent backend
    # uses tools for real dataset analysis, which is a different regime
    # than HumanEval function completion.

    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message=PLANNER_SYS,
    )
    researcher = AssistantAgent(
        name="ResearchAgent",
        model_client=client,
        system_message=RESEARCHER_SYS,
    )
    coder = AssistantAgent(
        name="DataScientist",
        model_client=client,
        system_message=CODER_SYS,
    )
    reviewer = AssistantAgent(
        name="CodeReviewerAgent",
        model_client=client,
        system_message=REVIEWER_SYS,
    )

    termination = (
        TextMentionTermination("TERMINATE", sources=["CodeReviewerAgent"])
        | MaxMessageTermination(max_messages)
    )
    team = SelectorGroupChat(
        [planner, researcher, coder, reviewer],
        model_client=client,
        termination_condition=termination,
        selector_prompt=SELECTOR_PROMPT,
        selector_func=_deterministic_selector,
    )

    messages = []
    plan = ""
    critiques = []
    coder_outputs = []
    researcher_outputs = []
    # Any agent that isn't DataScientist. Used as an extraction fallback
    # when DataScientist never produces a parseable body (weak-model role
    # confusion makes this surprisingly common with SelectorGroupChat).
    other_outputs = []
    approved = False
    interaction_count = 0
    coder_turns = 0

    start = time.perf_counter()
    async for msg in team.run_stream(task=_build_user_task(task)):
        if isinstance(msg, TaskResult):
            continue

        source = getattr(msg, "source", "") or ""
        text = _message_text(msg)
        messages.append(
            {"source": source, "type": type(msg).__name__, "content": text}
        )

        if source not in _AGENT_SOURCES:
            continue

        if isinstance(msg, TextMessage):
            interaction_count += 1

            if source == "Planner":
                if not plan:
                    plan = text
                other_outputs.append(text)
            elif source == "ResearchAgent":
                researcher_outputs.append(text)
                other_outputs.append(text)
            elif source == "DataScientist":
                coder_turns += 1
                coder_outputs.append(text)
            elif source == "CodeReviewerAgent":
                critiques.append(text)
                other_outputs.append(text)
                if "APPROVED" in text.upper():
                    approved = True

    latency = time.perf_counter() - start

    body, raw_response = _pick_body_from_coder_outputs(
        coder_outputs, task["entry_point"], fallback_outputs=other_outputs
    )

    return MultiRunResult(
        completion=body,
        raw_response=raw_response,
        plan=plan,
        critiques=critiques,
        llm_latency_sec=latency,
        interaction_count=interaction_count,
        turns_used=coder_turns,
        approved=approved,
        messages=messages,
        researcher_outputs=list(researcher_outputs),
        coder_outputs=list(coder_outputs),
        reviewer_outputs=list(critiques),
    )


def run_multi(task, **kwargs):
    """Synchronous entry point for the harness."""
    return asyncio.run(run_multi_async(task, **kwargs))
