import os
import sys
import json
import asyncio

# Ensure project root is in sys.path for portable environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.tasks import process_and_store_batch, process_file_ingestion
from app.services.loaders import extract_text_from_source, extract_chunks_from_source, structural_splitter
# from app.core.database import pool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger_config import get_logger
from app.core.ai_manager import ai_manager
# from app.core.chroma_client import get_chroma_collection
from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME
from app.core.graph_manager import graph_manager
from app.services.retrieval import perform_agentic_search, synthesize_dashboard_report, memory_streamer

# Initialize Logger
logger = get_logger("main_api")

load_dotenv()

app = FastAPI(title="AI Memory & Knowledge Base Server (Hybrid Powered)")

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MEDIA_DIR = os.path.join(BASE_DIR, "media")
if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR)

# Mount Static Files for Admin UI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Enable CORS for local file viewing and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Initializes heavy resources on startup to ensure fast request handling.
    """
    logger.info("🚀 Server starting up. Initializing Neural Core...")
    try:
        init_milvus_collection()
        logger.info("✅ Neural Core (Milvus) Ready.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Neural Core: {e}")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/styles.css")
async def read_styles():
    return FileResponse(os.path.join(STATIC_DIR, "styles.css"))

@app.get("/admin.js")
async def read_js():
    return FileResponse(os.path.join(STATIC_DIR, "admin.js"))


class IngestRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class FileIngestRequest(BaseModel):
    source: str # Path to PDF, Docx, or a URL
    metadata: Dict[str, Any] = {}
    heavy_parsing: bool = False
    sync: bool = False

class ChatRequest(BaseModel):
    message: str
    limit: int = 20
    metadata_filter: Dict[str, Any] = {}

class FeedbackRequest(BaseModel):
    doc_id: str
    query: str
    score: float # 1.0 for helpful, -1.0 for not helpful

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response Status: {response.status_code}")
    return response

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata_json: str = Form("{}"),
    heavy_parsing: bool = Form(False),
    sync: bool = Form(False)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        metadata = json.loads(metadata_json)
        metadata["filename"] = file.filename
        metadata["source"] = file_path
        
        if sync:
            # Run synchronously in the API request thread for real-time indexing
            task_result = process_file_ingestion.apply(args=(file_path, heavy_parsing, metadata))
            return {"status": "success", "filename": file.filename, "info": task_result.result}
        else:
            # Offload parsing and embedding to Celery in the background
            task = process_file_ingestion.delay(file_path, heavy_parsing, metadata)
            return {"status": "accepted", "filename": file.filename, "task_id": task.id}
    except Exception as e:
        logger.exception(f"Error in upload_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    from celery.result import AsyncResult
    from app.services.tasks import celery_app
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.state, "info": result.info}

@app.get("/graph")
@app.get("/system/graph")
async def get_graph():
    try:
        return graph_manager.get_graph_data()
    except Exception as e:
        logger.error(f"Failed to fetch graph data: {e}")
        return {"nodes": [], "edges": []}

@app.get("/system/health")
async def get_system_health():
    health = {"database": "online", "redis": "offline", "groq_cloud": ai_manager.llm is not None, "graph_rag": "offline"}
    try:
        if graph_manager.graph and len(graph_manager.graph.nodes) > 0: health["graph_rag"] = "online"
    except Exception: pass
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL"), socket_timeout=2)
        r.ping()
        health["redis"] = "online"
    except Exception: pass
    try:
        stats = milvus_client.get_collection_stats(collection_name=COLLECTION_NAME)
        health["memory_count"] = int(stats.get("row_count", 0))
    except Exception: health["memory_count"] = 0
    return health

@app.get("/memories")
async def get_memories(limit: int = 20):
    try:
        res = milvus_client.query(collection_name=COLLECTION_NAME, filter="id >= 0", limit=limit, output_fields=["id", "content", "created_at"])
        return res
    except Exception as e:
        logger.error(f"Failed to fetch memories: {e}")
        return []

@app.post("/ingest/file")
async def ingest_file(request: FileIngestRequest):
    try:
        chunks = extract_chunks_from_source(request.source, request.heavy_parsing)
        if request.sync:
            task_result = process_and_store_batch.apply(args=(chunks, request.metadata))
            return {"status": "success", "info": task_result.result}
        else:
            task = process_and_store_batch.delay(chunks, request.metadata)
            return {"status": "accepted", "task_id": task.id}
    except Exception as e:
        logger.exception(f"Error in ingest_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_memory(request: ChatRequest):
    try:
        return await perform_agentic_search(request.message, request.limit)
    except Exception as e:
        logger.exception(f"Error in chat_with_memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def provide_feedback(request: FeedbackRequest):
    """
    Active Learning: Increments or decrements the 'authority' score of a document based on user feedback.
    """
    try:
        # In a real enterprise system, we would store this in a separate interaction table.
        # For this version, we update the Milvus metadata dynamically.
        current_res = milvus_client.get(collection_name=COLLECTION_NAME, ids=[request.doc_id])
        if not current_res: return {"status": "error", "reason": "document_not_found"}
        
        current_doc = current_res[0]
        current_authority = current_doc.get("authority", 1.0)
        new_authority = current_authority + (request.score * 0.1) # 10% boost/penalty
        
        milvus_client.upsert(
            collection_name=COLLECTION_NAME,
            data=[{"id": request.doc_id, "authority": new_authority}]
        )
        
        logger.info(f"Active Learning: Doc {request.doc_id} authority adjusted to {new_authority}")
        return {"status": "success", "new_authority": new_authority}
    except Exception as e:
        logger.error(f"Feedback failed: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/search")
async def dashboard_search(query: str, limit: int = 20):
    try:
        return await synthesize_dashboard_report(query, limit)
    except Exception as e:
        logger.exception(f"Dashboard search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
