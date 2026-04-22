"""LLM-as-judge over benchmark runs using Ollama (optional vision for PNG charts)."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from ollama import AsyncClient

from app.core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_JUDGE_MODEL,
    OLLAMA_MODEL,
    ollama_async_client_kwargs,
)


JUDGE_SYSTEM = """You are an impartial evaluation judge for data-science agent runs.
Score each run independently using the rubric. Output a single JSON object only — no markdown fences, no text before or after.

Required JSON shape:
{
  "single": {
    "overall_1_5": <int 1-5>,
    "correctness_1_5": <int>,
    "visual_quality_1_5": <int>,
    "methodology_ml_1_5": <int>,
    "leakage_and_validation_1_5": <int>,
    "clarity_1_5": <int>,
    "brief_rationale": "<one paragraph>"
  },
  "multi": {
    "overall_1_5": <int>,
    "correctness_1_5": <int>,
    "visual_quality_1_5": <int>,
    "methodology_ml_1_5": <int>,
    "leakage_and_validation_1_5": <int>,
    "clarity_1_5": <int>,
    "brief_rationale": "<one paragraph>"
  },
  "comparison_winner": "single" | "multi" | "tie",
  "comparison_notes": "<one paragraph explaining winner>"
}

Rules:
- methodology_ml_1_5: for non-ML tasks, score how sound the analytical approach is; for ML tasks, weight train/test awareness, metrics, baselines, and stated limitations.
- leakage_and_validation_1_5: data leakage awareness, appropriate validation, uncertainty.
- If transcripts are empty, score low and say so in rationale.
- comparison_winner: pick the run that better satisfies the user task overall; do not favor multi by default.
- Complexity-bias policy (applies when REFERENCE_HINT is present and implies strict contracts such as reviewer-gated QA, reconciliation, leakage-control protocol, reproducibility contract, or threshold governance):
  - Prefer the run with explicit multi-step workflow evidence and contract compliance.
  - Workflow evidence means visible decomposition/review loop behavior in transcript (e.g., planning/research/review stages, blocking fixes addressed, validation checklist evidence).
  - If a run misses required contract evidence from REFERENCE_HINT, cap overall_1_5 at 2.
  - For contract-heavy tasks, a single-pass response without clear staged validation should be penalized in methodology_ml_1_5 and leakage_and_validation_1_5.
  - In close calls on contract-heavy tasks, break ties in favor of the run with stronger workflow evidence and stricter compliance.
"""


def _judge_model() -> str:
    return OLLAMA_JUDGE_MODEL.strip() or OLLAMA_MODEL


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def parse_judge_json(text: str) -> dict[str, Any]:
    """Parse judge output; raise ValueError if not valid JSON object."""
    cleaned = _strip_json_fence(text)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("Judge output must be a JSON object")
    return obj


async def judge_pair_with_ollama(
    *,
    task_text: str,
    expected_output: str,
    reference_hint: str | None,
    single_transcript: str,
    multi_transcript: str,
    image_paths_single: list[str],
    image_paths_multi: list[str],
    max_images_per_side: int = 4,
) -> tuple[dict[str, Any], str]:
    """
    Call Ollama once (with optional images) and return (parsed_dict, raw_text).

    Uses the chat API with base64 images when paths are provided and files exist.
    """
    client = AsyncClient(**ollama_async_client_kwargs(host=OLLAMA_BASE_URL))
    model = _judge_model()

    user_text = (
        f"USER_TASK:\n{task_text}\n\n"
        f"EXPECTED_OUTPUT_LABEL:\n{expected_output}\n\n"
    )
    if reference_hint:
        user_text += f"REFERENCE_HINT:\n{reference_hint}\n\n"
    user_text += (
        "=== SINGLE_AGENT_TRANSCRIPT (may be truncated) ===\n"
        f"{single_transcript[:120000]}\n\n"
        "=== MULTI_AGENT_TRANSCRIPT (may be truncated) ===\n"
        f"{multi_transcript[:120000]}\n"
    )

    images: list[str] = []
    for p in (image_paths_single + image_paths_multi)[: max_images_per_side * 2]:
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        try:
            with open(p, "rb") as fp:
                images.append(base64.b64encode(fp.read()).decode("ascii"))
        except OSError:
            continue

    msg: dict[str, Any] = {"role": "user", "content": user_text}
    if images:
        msg["images"] = images

    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            msg,
        ],
    )
    raw = (response.message.content or "").strip()
    try:
        return parse_judge_json(raw), raw
    except json.JSONDecodeError:
        repair = await client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The following text was supposed to be a single JSON object "
                        "but is invalid. Emit ONLY a corrected valid JSON object, same schema "
                        "as described earlier (single, multi, comparison_winner, comparison_notes).\n\n"
                        f"INVALID:\n{raw[:8000]}"
                    ),
                }
            ],
        )
        raw2 = (repair.message.content or "").strip()
        return parse_judge_json(raw2), raw + "\n---REPAIR---\n" + raw2


@dataclass
class RunPairPaths:
    task_id: str
    dataset_id: str
    single_trail: str
    multi_trail: str
    single_manifest: str
    multi_manifest: str


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _trail_to_transcript(trail_path: str) -> str:
    lines: list[str] = []
    try:
        with open(trail_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = obj.get("source", "")
                typ = obj.get("message_type", "")
                content = obj.get("content", "")
                lines.append(f"[{src}][{typ}] {content}")
    except OSError:
        return ""
    return "\n".join(lines)


def _manifest_images(manifest_path: str) -> list[str]:
    try:
        m = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return []
    out: list[str] = []
    for item in m.get("files", []):
        p = item.get("path")
        if isinstance(p, str) and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            out.append(p)
    return out


async def judge_run_pair_from_disk(
    meta_task: dict[str, Any],
    paths: RunPairPaths,
) -> dict[str, Any]:
    """Build transcripts and call judge for one (dataset, task) pair."""
    single_t = _trail_to_transcript(paths.single_trail)
    multi_t = _trail_to_transcript(paths.multi_trail)
    img_s = _manifest_images(paths.single_manifest)
    img_m = _manifest_images(paths.multi_manifest)
    parsed, raw_combined = await judge_pair_with_ollama(
        task_text=str(meta_task.get("task", "")),
        expected_output=str(meta_task.get("expected_output", "")),
        reference_hint=meta_task.get("reference_metrics_hint"),
        single_transcript=single_t,
        multi_transcript=multi_t,
        image_paths_single=img_s,
        image_paths_multi=img_m,
    )
    return {
        "task_id": paths.task_id,
        "dataset_id": paths.dataset_id,
        "scores": parsed,
        "raw_judge_output": raw_combined,
    }
