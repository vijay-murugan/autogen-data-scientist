from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
from contextlib import asynccontextmanager
import shutil

# Import refactored pipelines
from app.agents.single_agent import run_single_agent_pipeline
from app.agents.multi_agent import run_multi_agent_pipeline
from app.agents.qa_agent import run_qa_pipeline
from app.agents.ml_agent import run_ml_pipeline, run_multi_agent_ml_pipeline
from app.core.config import WORKING_DIR

from app.core.custom_client import SimpleOllamaClient
from app.core.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from autogen_core.models import SystemMessage, UserMessage

from app.backend.dataset_resolver import (
    CLEANED_SESSIONS_SUBDIR,
    get_dataset_manifest,
    get_or_create_cleaned_session_file,
    resolve_selected_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure run_artifacts exists
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    yield


app = FastAPI(lifespan=lifespan)

# Allow React frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Since we are local, * is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images as static files
# WORKING_DIR is relative to project root.
# Since this script runs from the project root (via python -m app.backend.main),
# the relative path should work if we are careful.
app.mount("/artifacts", StaticFiles(directory=WORKING_DIR), name="artifacts")


def _clear_run_artifacts() -> None:
    """Remove generated run artifacts while preserving cleaned session cache."""
    if not os.path.exists(WORKING_DIR):
        return

    for f in os.listdir(WORKING_DIR):
        if f == CLEANED_SESSIONS_SUBDIR:
            continue
        filepath = os.path.join(WORKING_DIR, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)

@app.get("/")
async def root():
    return {"status": "Multi-Agent Dashboard API is running"}


def _inject_dataset_metadata_into_sidecars(
    dataset_path: str,
    dataset_ref: str = "",
    selected_file: str = "",
) -> None:
    """
    Walk WORKING_DIR for every chart JSON sidecar produced by the most recent
    run and write dataset_path (plus dataset_ref / selected_file when available)
    into each one. This gives the verifier a deterministic, per-chart reference
    back to the exact dataset that produced the chart, independent of whatever
    the user may later select in the UI. Backend-driven (not LLM-driven) so the
    metadata is guaranteed to be present and correct.
    """
    if not dataset_path or not os.path.exists(WORKING_DIR):
        return

    chart_extensions = (".png", ".jpg", ".jpeg", ".svg")
    abs_dataset_path = os.path.abspath(dataset_path)

    for root, _dirs, filenames in os.walk(WORKING_DIR):
        for fname in filenames:
            if not fname.lower().endswith(chart_extensions):
                continue
            base_name = os.path.splitext(fname)[0]
            json_path = os.path.join(root, base_name + ".json")
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except Exception as e:
                print(f"Failed to read sidecar {json_path} for metadata injection: {e}")
                continue

            # Preserve non-dict payloads by wrapping them so we can still attach metadata.
            if not isinstance(sidecar, dict):
                sidecar = {"_chart_payload": sidecar}

            sidecar["dataset_path"] = abs_dataset_path
            if dataset_ref:
                sidecar["dataset_ref"] = dataset_ref
            if selected_file:
                sidecar["selected_file"] = selected_file

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)
            except Exception as e:
                print(f"Failed to write sidecar {json_path} after metadata injection: {e}")


async def agent_event_generator(
    task: str,
    mode: str,
    dataset_path: str | None = None,
    preflight_warning: str = "",
    dataset_ref: str = "",
    selected_file: str = "",
):
    """
    Generator that runs the agent pipeline and yields SSE events.
    """
    seen_agents: list[str] = []
    seen_agent_keys: set[str] = set()
    last_result_by_source: dict[str, str] = {}
    all_agent_messages: list[dict] = []  # Collect all substantive messages

    def _is_textual_event(event_type: str) -> bool:
        return event_type.lower() in {"textmessage", "finalresult"}

    def _extract_direct_answer(text: str) -> str:
        cleaned = (text or "").replace("TERMINATE", "").strip()
        if not cleaned:
            return cleaned

        # Case-insensitive answer marker detection
        lowered = cleaned.lower()
        markers = ["final_answer:", "final answer:", "answer:"]
        
        for marker in markers:
            idx = lowered.find(marker)
            if idx != -1:
                cleaned = cleaned[idx + len(marker):].strip()
                break

        # Aggressively remove procedure text patterns
        import re
        
        # Remove sentences that describe process/steps
        procedure_sentences = [
            r"(?i)by\s+summarizing.*?becomes\s+obvious[.]?",
            r"(?i)after\s+the\s+data\s+is\s+cleaned[,.]?.*?next\s+step\s+is[,.]?",
            r"(?i)duplicate\s+transactions\s+are\s+removed[,.]?.*?missing\s+values\s+handled[,.]?",
            r"(?i)product\s+identifiers\s+standardized[,.]?",
            r"(?i)the\s+next\s+step\s+is\s+to[,.]?",
            r"(?i)group\s+the\s+data[,.]?",
            r"(?i)if\s+you\s+run.*?steps[,.]?",
            r"(?i)view\s+the\s+outputs?[,.]?",
            r"(?i)follow\s+these\s+steps[,.]?",
            r"(?i)the\s+chart\s+(?:will\s+show|titles|reveals?)[,.]?",
            r"(?i)once\s+you\s+(?:view|see|analyze)[,.]?",
            r"(?i)by\s+(?:looking\s+at|examining|reviewing)\s+the[,.]?",
        ]
        
        for pattern in procedure_sentences:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove step/number prefixes
        cleaned = re.sub(r"(?i)^\s*step\s*\d+[:.]?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)^\s*\d+[.)]\s+", "", cleaned)
        cleaned = re.sub(r"(?i)^\s*[-•]\s+", "", cleaned)
        
        # Remove process verbs at start
        cleaned = re.sub(r"(?i)^(first|then|next|after|finally)\s*,?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)^(to\s+(?:answer|find|determine|solve|analyze)\s+this[,.]?\s*)", "", cleaned)
        cleaned = re.sub(r"(?i)^(let\s+me\s+(?:start|begin|explain|walk|guide)\s*)", "", cleaned)
        cleaned = re.sub(r"(?i)^(i\s+(?:will|would|need\s+to|have\s+to|should)\s+(?:start|begin|explain|walk|guide)\s*)", "", cleaned)
        
        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ,.;:")
        
        return cleaned.strip()

    def _select_final_result(current_mode: str) -> str:
        preferred_sources: dict[str, list[str]] = {
            "single": ["analyst"],
            "qa": ["dataconsultant"],
            "ml": ["resultsummarizer", "mlanalyst"],
            "multi_ml": ["resultsummarizer", "mlanalyst", "datascientist"],
            "multi": ["resultsummarizer", "coderevieweragent", "datascientist"],
        }

        for source_key in preferred_sources.get(current_mode, ["datascientist"]):
            result = last_result_by_source.get(source_key, "").strip()
            if result:
                return _extract_direct_answer(result)

        return "No final answer was produced."

    def _synthesize_final_answer(
        task: str,
        mode: str,
        all_messages: list[dict],
        last_by_source: dict[str, str],
    ) -> str:
        """
        Get the final natural language answer from the appropriate agent.
        Only uses the designated final output agent for each mode.
        """
        # Map modes to their final answer source
        final_sources = {
            "single": "analyst",
            "qa": "dataconsultant",
            "multi": "resultsummarizer",
            "ml": "resultsummarizer",
            "multi_ml": "resultsummarizer",
        }

        source_key = final_sources.get(mode, "analyst")
        final_output = last_by_source.get(source_key, "").strip()

        if not final_output:
            return "No final answer was produced."

        # Extract just the natural language answer, removing markers and procedure text
        cleaned = _extract_direct_answer(final_output)

        # Remove any remaining procedure patterns
        procedure_patterns = [
            r"(?i)^\s*step\s*\d+[:.]?\s*",
            r"(?i)^\s*\d+\.\s+",
            r"(?i)\bfirst\s+(?:i|we)\s+",
            r"(?i)\bthen\s+(?:i|we)\s+",
            r"(?i)\bnext\s+(?:i|we)\s+",
            r"(?i)\bafter\s+(?:that|this)\s+",
            r"(?i)\bfinally\s+",
            r"(?i)\bto\s+answer\s+this\s+",
            r"(?i)\blet\s+me\s+",
            r"(?i)\bi\s+(?:will|would|need to|have)\s+",
        ]

        import re
        for pattern in procedure_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip() or final_output

    try:
        if preflight_warning:
            yield (
                "data: "
                + json.dumps(
                    {
                        "source": "system",
                        "type": "PreflightNotice",
                        "content": preflight_warning,
                        "timestamp": "",
                    }
                )
                + "\n\n"
            )

        if mode == "single":
            if not dataset_path:
                raise ValueError("dataset_path is required for single mode")
            pipeline = run_single_agent_pipeline(task, dataset_path)
        elif mode == "qa" or mode == "DataConsultant":
            if not dataset_path:
                raise ValueError("dataset_path is required for qa mode")
            pipeline = run_qa_pipeline(task, dataset_path)
        elif mode == "ml":
            pipeline = run_ml_pipeline(task)
        elif mode == "multi_ml":
            pipeline = run_multi_agent_ml_pipeline(task)
        else:
            if not dataset_path:
                raise ValueError("dataset_path is required for multi mode")
            pipeline = run_multi_agent_pipeline(task, dataset_path)

        async for message in pipeline:
            # We wrap the message in a standardized JSON for the frontend
            data = {
                "source": getattr(message, "source", "system"),
                "content": str(getattr(message, "content", message)),
                "type": getattr(message, "type", "TextMessage"),
                "timestamp": str(getattr(message, "created_at", "")),
            }
            source = str(data["source"])
            source_key = source.lower()
            if (
                source_key
                and source_key not in {"system", "user", "error"}
                and source_key not in seen_agent_keys
            ):
                seen_agent_keys.add(source_key)
                seen_agents.append(source)

            content = str(data["content"]).strip()
            event_type = str(data["type"])
            if (
                source_key
                and source_key not in {"system", "user", "error"}
                and _is_textual_event(event_type)
                and content
            ):
                last_result_by_source[source_key] = content
                # Collect all substantive agent communications for synthesis
                all_agent_messages.append({
                    "source": source,
                    "content": content,
                    "key": source_key,
                })
            yield f"data: {json.dumps(data)}\n\n"

        # Synthesize final answer from all agent communications
        final_content = _synthesize_final_answer(task, mode, all_agent_messages, last_result_by_source)
        final_data = {
            "source": "final_result",
            "content": final_content,
            "type": "FinalResult",
            "timestamp": "",
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'source': 'error', 'content': str(e)})}\n\n"

    # After the pipeline completes (success or error), deterministically stamp the
    # source dataset into every chart sidecar so the verifier can trust it later.
    if dataset_path:
        try:
            _inject_dataset_metadata_into_sidecars(
                dataset_path=dataset_path,
                dataset_ref=dataset_ref,
                selected_file=selected_file,
            )
        except Exception as e:
            print(f"Sidecar metadata injection failed: {e}")

    # Notify frontend that artifacts are ready for refresh
    yield f"data: {json.dumps({'source': 'system', 'type': 'ArtifactsReady', 'content': 'Charts generated'})}\n\n"

    yield "data: [DONE]\n\n"
@app.get("/api/artifacts")
async def list_artifacts():
    """List all generated chart images plus any JSON sidecars with underlying data."""
    files = []
    if os.path.exists(WORKING_DIR):
        for root, dirs, filenames in os.walk(WORKING_DIR):
            for fname in sorted(filenames):
                if fname.endswith((".png", ".jpg", ".jpeg", ".svg")):
                    rel_path = os.path.relpath(os.path.join(root, fname), WORKING_DIR)

                    # Look for a JSON sidecar with the same base name
                    base_name = os.path.splitext(fname)[0]
                    json_path = os.path.join(root, base_name + ".json")
                    metadata = None
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                metadata = json.load(f)
                        except Exception as e:
                            print(f"Failed to load sidecar {json_path}: {e}")

                    files.append(
                        {
                            "name": fname,
                            "url": f"/artifacts/{rel_path.replace(os.sep, '/')}",
                            "metadata": metadata,
                            "modified_at_ms": int(os.path.getmtime(os.path.join(root, fname)) * 1000),
                        }
                    )
    return {"artifacts": files}


@app.post("/api/chart-qa")
async def chart_qa(request: Request):
    """Answer a natural-language question about a previously generated chart
    using its JSON sidecar data."""
    body = await request.json()
    chart_name = body.get("chart_name", "")
    question = body.get("question", "")

    if not chart_name or not question:
        return {"answer": "Missing chart_name or question in request."}

    # Find the JSON sidecar for this chart (recursive search)
    base_name = os.path.splitext(os.path.basename(chart_name))[0]
    json_name = base_name + ".json"
    json_path = None
    for root, dirs, filenames in os.walk(WORKING_DIR):
        if json_name in filenames:
            json_path = os.path.join(root, json_name)
            break

    if not json_path or not os.path.exists(json_path):
        return {
            "answer": f"Sorry, no underlying data was saved for '{chart_name}'. The agent may not have produced a JSON sidecar for this chart."
        }

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            chart_data = json.load(f)
    except Exception as e:
        return {"answer": f"Failed to read chart data: {e}"}

    # Ask the LLM
    client = SimpleOllamaClient(model=OLLAMA_MODEL, host=OLLAMA_BASE_URL)

    system_prompt = (
        "You are a precise data analyst. You will be given the underlying "
        "data of a chart in JSON format, plus a user question about the chart. "
        "Answer concisely using ONLY the data provided. Cite specific numbers when relevant. "
        "If the question cannot be answered from the given data, say so."
    )

    user_prompt = (
        f"Chart data:\n```json\n{json.dumps(chart_data, indent=2)}\n```\n\n"
        f"Question: {question}"
    )

    try:
        response = await client.create(
            [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source="user"),
            ]
        )
        answer = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception as e:
        answer = f"Error calling LLM: {e}"

    return {"answer": answer}


@app.post("/api/verify-chart")
async def verify_chart(request: Request):
    """
    Re-compute the chart's underlying values from the raw dataset and compare
    them to the JSON sidecar saved by the DataScientist.
    Returns: {"status": "PASS"|"WARN"|"FAIL"|"UNKNOWN", "details": str, "log": str}
    """
    body = await request.json()
    chart_name = body.get("chart_name", "")

    if not chart_name:
        return {
            "status": "UNKNOWN",
            "details": "Missing chart_name in request.",
            "log": "",
        }

    # Locate the JSON sidecar for this chart
    base_name = os.path.splitext(os.path.basename(chart_name))[0]
    json_name = base_name + ".json"
    json_path = None
    for root, dirs, filenames in os.walk(WORKING_DIR):
        if json_name in filenames:
            json_path = os.path.join(root, json_name)
            break

    if not json_path or not os.path.exists(json_path):
        return {
            "status": "UNKNOWN",
            "details": f"No JSON sidecar found for '{chart_name}'. Cannot verify without underlying data.",
            "log": "",
        }

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            chart_data = json.load(f)
    except Exception as e:
        return {
            "status": "UNKNOWN",
            "details": f"Failed to read sidecar: {e}",
            "log": "",
        }

    try:
        from app.agents.verifier_agent import run_verify_chart_pipeline

        verdict = await run_verify_chart_pipeline(chart_data)
    except Exception as e:
        verdict = {"status": "UNKNOWN", "details": f"Verifier crashed: {e}", "log": ""}

    return verdict


@app.post("/api/run")
async def run_task(request: Request):
    """
    Endpoint for complex analytics tasks.
    """
    body = await request.json()
    task = body.get("task", "")
    mode = body.get("mode", "multi")  # baseline or team
    dataset_ref = body.get("dataset_ref", "")
    selected_file = body.get("selected_file", "")
    session_id = body.get("session_id", "")

    if not task:
        raise HTTPException(status_code=400, detail="task is required.")
    resolved = resolve_selected_file(dataset_ref, selected_file)
    cleaned = get_or_create_cleaned_session_file(
        resolved["dataset_path"], session_id=session_id
    )
    target_dataset_path = cleaned["cleaned_dataset_path"]
    warning = ""
    if cleaned["cleaning_status"] != "cleaned":
        warning = cleaned["cleaning_message"]

    # DO NOT CLEAR ARTIFACTS - they are needed for frontend display
    # Only clear on fresh browser load / explicit user action

    # We use SSE for the long-running agent stream
    return StreamingResponse(
        agent_event_generator(
            task,
            mode,
            target_dataset_path,
            preflight_warning=warning,
            dataset_ref=resolved.get("dataset_ref", ""),
            selected_file=selected_file,
        ),
        media_type="text/event-stream",
    )

@app.post("/api/ml")
async def run_ml(request: Request):
    """
    Endpoint for ML tasks.
    """
    body = await request.json()
    task = body.get("task", "")
    mode = body.get("mode", "ml")
    _clear_run_artifacts()
    return StreamingResponse(
        agent_event_generator(task, mode, dataset_path=None, preflight_warning=""),
        media_type="text/event-stream",
    )

@app.post("/api/multi_ml")
async def run_multi_ml(request: Request):
    """
    Endpoint for multi-agent ML tasks.
    """
    body = await request.json()
    task = body.get("task", "")
    mode = body.get("mode", "multi_ml")
    _clear_run_artifacts()
    return StreamingResponse(
        agent_event_generator(task, mode, dataset_path=None, preflight_warning=""),
        media_type="text/event-stream",
    )

@app.post("/api/qa")
async def run_qa(request: Request):
    """
    Endpoint for specific dataset questions.
    """
    body = await request.json()
    question = body.get("question", "")
    dataset_ref = body.get("dataset_ref", "")
    selected_file = body.get("selected_file", "")
    session_id = body.get("session_id", "")
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")
    resolved = resolve_selected_file(dataset_ref, selected_file)
    cleaned = get_or_create_cleaned_session_file(
        resolved["dataset_path"], session_id=session_id
    )
    target_dataset_path = cleaned["cleaned_dataset_path"]
    warning = ""
    if cleaned["cleaning_status"] != "cleaned":
        warning = cleaned["cleaning_message"]

    # Clear old artifacts before each new run, preserving cleaned session cache.
    _clear_run_artifacts()

    # Use the Q&A specialist
    return StreamingResponse(
        agent_event_generator(
            question,
            "qa",
            target_dataset_path,
            preflight_warning=warning,
            dataset_ref=resolved.get("dataset_ref", ""),
            selected_file=selected_file,
        ),  # qa mode
        media_type="text/event-stream",
    )


@app.post("/api/datasets/lookup")
async def lookup_dataset(request: Request):
    """
    Resolve dataset ref from URL/slug and list supported files.
    """
    body = await request.json()
    dataset_ref = body.get("dataset_ref", "")
    manifest = get_dataset_manifest(dataset_ref)
    return JSONResponse(content=manifest)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
