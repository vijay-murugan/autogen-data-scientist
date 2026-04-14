from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
from contextlib import asynccontextmanager

# Import refactored pipelines
from app.agents.single_agent import run_single_agent_pipeline
from app.agents.multi_agent import run_multi_agent_pipeline
from app.agents.qa_agent import run_qa_pipeline
from app.core.config import WORKING_DIR
from app.backend.dataset_resolver import (
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
    allow_origins=["*"], # Since we are local, * is fine
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

async def agent_event_generator(
    task: str,
    mode: str,
    dataset_path: str,
    preflight_warning: str = "",
):
    """
    Generator that runs the agent pipeline and yields SSE events.
    """
    seen_agents: list[str] = []
    seen_agent_keys: set[str] = set()
    last_datascientist_result = ""
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
            pipeline = run_single_agent_pipeline(task, dataset_path)
        elif mode == "qa" or mode == "DataConsultant":
            pipeline = run_qa_pipeline(task, dataset_path)
        else:
            pipeline = run_multi_agent_pipeline(task, dataset_path)
            
        async for message in pipeline:
            # We wrap the message in a standardized JSON for the frontend
            data = {
                "source": getattr(message, 'source', 'system'),
                "content": str(getattr(message, 'content', message)),
                "type": getattr(message, 'type', 'TextMessage'),
                "timestamp": str(getattr(message, 'created_at', ''))
            }
            source = str(data["source"])
            source_key = source.lower()
            if source_key and source_key not in {"system", "user", "error"} and source_key not in seen_agent_keys:
                seen_agent_keys.add(source_key)
                seen_agents.append(source)

            if source_key == "datascientist":
                content = str(data["content"]).strip()
                if content and "TERMINATE" not in content:
                    last_datascientist_result = content
            yield f"data: {json.dumps(data)}\n\n"

        if mode == "multi":
            final_result = last_datascientist_result or "No execution result was produced by DataScientist."
            agents_used = ", ".join(seen_agents) if seen_agents else "None"
            final_content = (
                "Final Execution Result:\n"
                f"{final_result}\n\n"
                "Agents Used:\n"
                f"{agents_used}"
            )
            final_data = {
                "source": "final_result",
                "content": final_content,
                "type": "FinalResult",
                "timestamp": "",
            }
            yield f"data: {json.dumps(final_data)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'source': 'error', 'content': str(e)})}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/api/run")
async def run_task(request: Request):
    """
    Endpoint for complex analytics tasks.
    """
    body = await request.json()
    task = body.get("task", "")
    mode = body.get("mode", "multi") # baseline or team
    dataset_ref = body.get("dataset_ref", "")
    selected_file = body.get("selected_file", "")
    session_id = body.get("session_id", "")

    if not task:
        raise HTTPException(status_code=400, detail="task is required.")
    resolved = resolve_selected_file(dataset_ref, selected_file)
    cleaned = get_or_create_cleaned_session_file(resolved["dataset_path"], session_id=session_id)
    target_dataset_path = cleaned["cleaned_dataset_path"]
    warning = ""
    if cleaned["cleaning_status"] != "cleaned":
        warning = cleaned["cleaning_message"]
    
    # We use SSE for the long-running agent stream
    return StreamingResponse(
        agent_event_generator(task, mode, target_dataset_path, preflight_warning=warning),
        media_type="text/event-stream"
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
    cleaned = get_or_create_cleaned_session_file(resolved["dataset_path"], session_id=session_id)
    target_dataset_path = cleaned["cleaned_dataset_path"]
    warning = ""
    if cleaned["cleaning_status"] != "cleaned":
        warning = cleaned["cleaning_message"]
    
    # Use the Q&A specialist
    return StreamingResponse(
        agent_event_generator(question, "qa", target_dataset_path, preflight_warning=warning), # qa mode
        media_type="text/event-stream"
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
