from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
from app.core.config import WORKING_DIR

from app.core.custom_client import SimpleOllamaClient
from app.core.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from autogen_core.models import SystemMessage, UserMessage


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


@app.get("/")
async def root():
    return {"status": "Multi-Agent Dashboard API is running"}


async def agent_event_generator(task: str, mode: str):
    """
    Generator that runs the agent pipeline and yields SSE events.
    """
    try:
        if mode == "single":
            pipeline = run_single_agent_pipeline(task)
        elif mode == "qa" or mode == "DataConsultant":
            pipeline = run_qa_pipeline(task)
        else:
            pipeline = run_multi_agent_pipeline(task)

        async for message in pipeline:
            # We wrap the message in a standardized JSON for the frontend
            data = {
                "source": getattr(message, "source", "system"),
                "content": str(getattr(message, "content", message)),
                "type": getattr(message, "type", "TextMessage"),
                "timestamp": str(getattr(message, "created_at", "")),
            }
            yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'source': 'error', 'content': str(e)})}\n\n"

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

    # Clear old artifacts before each new run
    if os.path.exists(WORKING_DIR):
        for f in os.listdir(WORKING_DIR):
            filepath = os.path.join(WORKING_DIR, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)

    # We use SSE for the long-running agent stream
    return StreamingResponse(
        agent_event_generator(task, mode), media_type="text/event-stream"
    )


@app.post("/api/qa")
async def run_qa(request: Request):
    """
    Endpoint for specific dataset questions.
    """
    body = await request.json()
    question = body.get("question", "")

    # Clear old artifacts before each new run
    if os.path.exists(WORKING_DIR):
        for f in os.listdir(WORKING_DIR):
            filepath = os.path.join(WORKING_DIR, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)

    # Use the Q&A specialist
    return StreamingResponse(
        agent_event_generator(question, "qa"),  # qa mode
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
