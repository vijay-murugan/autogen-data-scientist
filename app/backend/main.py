from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
                "source": getattr(message, 'source', 'system'),
                "content": str(getattr(message, 'content', message)),
                "type": getattr(message, 'type', 'TextMessage'),
                "timestamp": str(getattr(message, 'created_at', ''))
            }
            yield f"data: {json.dumps(data)}\n\n"
            
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
    
    # We use SSE for the long-running agent stream
    return StreamingResponse(
        agent_event_generator(task, mode),
        media_type="text/event-stream"
    )

@app.post("/api/qa")
async def run_qa(request: Request):
    """
    Endpoint for specific dataset questions.
    """
    body = await request.json()
    question = body.get("question", "")
    
    # Use the Q&A specialist
    return StreamingResponse(
        agent_event_generator(question, "qa"), # qa mode
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
