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
from app.services.tasks import process_and_store_batch
from app.services.loaders import extract_text_from_source, extract_chunks_from_source, structural_splitter
# from app.core.database import pool
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger_config import get_logger
from app.core.ai_manager import ai_manager
from app.core.chroma_client import get_chroma_collection
from app.core.graph_manager import graph_manager

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

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/styles.css")
async def read_styles():
    return FileResponse(os.path.join(STATIC_DIR, "styles.css"))

@app.get("/admin.js")
async def read_js():
    return FileResponse(os.path.join(STATIC_DIR, "admin.js"))

# Initialize Local Cross-Encoder for Re-ranking (Advanced RAG)
logger.info("Loading Cross-Encoder model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# LLM functionality is now handled exclusively by ai_manager (Local-First)

# Advanced Intelligence Transformation Prompt
# This handles vague queries and identifies technical patterns
rewrite_prompt = ChatPromptTemplate.from_template(
    "You are an AI 'Prompt Engineer' for a high-precision Vector Search engine. "
    "Your goal is to expand the user's vague query into a rich, descriptive technical search. "
    "Rules: \n"
    "1. If the user uses vague words like 'it', 'them', or 'fix', infer the context based on common IT troubleshooting. \n"
    "2. Identify potential user IDs or Alphanumeric codes and maintain them. \n"
    "3. Provide exactly 1-3 diverse variations of the query, separated by the pipe character '|'. \n"
    "Example Query: 'jwx fix' -> 'Technical resolution and fix details for user jwx1369347 | Maintenance logs and alarm clearance for jwx | Discussion about server fixes involving jwx identifiers' \n"
    "Target Query: {query} \n"
    "Intelligence Expansion:"
)

# --- RRF Constants ---
K = 60 # Standard constant for Reciprocal Rank Fusion
THRESHOLD = -99.0 # DEACTIVATED: User wants to see all results regardless of score (Maximum Recall Mode)

class IngestRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class FileIngestRequest(BaseModel):
    source: str # Path to PDF, Docx, or a URL
    metadata: Dict[str, Any] = {}
    heavy_parsing: bool = False

class ChatRequest(BaseModel):
    message: str
    limit: int = 5
    metadata_filter: Dict[str, Any] = {}

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
    heavy_parsing: bool = Form(False)
):
    """
    Handles direct file uploads from the browser.
    Saves the file to the uploads directory and triggers ingestion.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"Uploading file to: {file_path} (Heavy: {heavy_parsing})")
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        metadata = json.loads(metadata_json)
        metadata["filename"] = file.filename
        metadata["heavy_parsing"] = heavy_parsing
        
        # Trigger ingestion using the newly saved path
        content = extract_text_from_source(file_path, heavy_parsing)
        chunks = structural_splitter.split_text(content)
        task = process_and_store_batch.delay(chunks, metadata)
        
        return {
            "status": "accepted",
            "filename": file.filename,
            "chunks_identified": len(chunks),
            "task_id": task.id
        }
    except Exception as e:
        logger.exception(f"Error in upload_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Checks the status of a background ingestion task.
    """
    from celery.result import AsyncResult
    from app.services.tasks import app as celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.state,
        "meta": result.info # This contains our custom progress meta
    }
    
    if result.state == "SUCCESS":
        response["result"] = result.get()
    elif result.state == "FAILURE":
        response["error"] = str(result.info)
        
    return response

@app.get("/system/health")
async def get_system_health():
    """
    Checks connectivity to all core services.
    """
    health = {
        "database": "online", # ChromaDB is always assume online if the process is up
        "redis": "offline",
        "groq_cloud": ai_manager.llm is not None,
        "graph_rag": "offline"
    }
    
    # Check Knowledge Graph
    try:
        if graph_manager.graph and len(graph_manager.graph.nodes) > 0:
            health["graph_rag"] = "online"
        else:
            # Even if empty, if it's initialized it's 'online'
            health["graph_rag"] = "online (empty)"
    except Exception:
        pass
        
    # Check Redis
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL"), socket_timeout=2)
        r.ping()
        health["redis"] = "online"
    except Exception:
        pass

    # Add memory count
    try:
        collection = get_chroma_collection()
        health["memory_count"] = collection.count()
    except Exception:
        health["memory_count"] = 0
    # Check Memory Count
    try:
        # with pool.connection() as conn:
        #     with conn.cursor() as cur:
        #         cur.execute("SELECT count(*) FROM ai_memory")
        #         health["memory_count"] = cur.fetchone()[0]
        collection = get_chroma_collection()
        health["memory_count"] = collection.count()
    except Exception:
        health["memory_count"] = 0
        
    return health

@app.get("/memories")
async def get_memories(limit: int = 10):
    """
    Retrieves the most recent memories stored in the Database.
    """
    try:
        # with pool.connection() as conn:
        #     with conn.cursor() as cur:
        #         cur.execute(\"\"\"
        #             SELECT id, content, metadata, created_at 
        #             FROM ai_memory 
        #             ORDER BY created_at DESC 
        #             LIMIT %s
        #         \"\"\", (limit,))
        #         rows = cur.fetchall()
                
        #         memories = []
        #         for row in rows:
        #             memories.append({
        #                 "id": row[0],
        #                 "content": row[1],
        #                 "metadata": row[2],
        #                 "created_at": row[3].isoformat()
        #             })
        #         return memories
        
        # Approximate recency check for Chroma by getting recent items 
        # (Chroma doesn't sort well out of the box without filtering, we just get N items)
        collection = get_chroma_collection()
        res = collection.get(
            limit=limit,
            include=["metadatas", "documents"]
        )
        memories = []
        from datetime import datetime, timezone
        
        if res and res["ids"]:
            for i, doc_id in enumerate(res["ids"]):
                memories.append({
                    "id": doc_id,
                    "content": res["documents"][i],
                    "metadata": res["metadatas"][i] if res["metadatas"] else {},
                    "created_at": res["metadatas"][i].get("created_at", datetime.now(timezone.utc).isoformat()) if res["metadatas"] else datetime.now(timezone.utc).isoformat()
                })
        return memories
    except Exception as e:
        logger.error(f"Failed to fetch memories: {e}")
        raise HTTPException(status_code=500, detail="Database retrieval failed")

@app.post("/ingest/file")
async def ingest_file(request: FileIngestRequest):
    """
    Ingests PDFs, Word docs, or Web pages by extracting text first.
    """
    logger.info(f"File Ingestion request for source: {request.source} (Heavy: {request.heavy_parsing})")
    try:
        # Use structural extraction to preserve row/page integrity
        chunks = extract_chunks_from_source(request.source, request.heavy_parsing)
        logger.info(f"Structural extraction identified {len(chunks)} chunks.")
        
        if not chunks:
            logger.warning("Structural extraction found no printable content. Skipping ingestion.")
            return {
                "status": "skipped",
                "source": request.source,
                "reason": "no_content_found"
            }
            
        task = process_and_store_batch.delay(chunks, request.metadata)
        logger.info(f"Accepted file ingestion. Chunks: {len(chunks)}, TaskID: {task.id}")
        return {
            "status": "accepted",
            "source": request.source,
            "chunks_identified": len(chunks),
            "task_id": task.id
        }
    except Exception as e:
        logger.exception(f"Error in ingest_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_memory(request: IngestRequest):
    """
    Asynchronously ingest memories with smart chunking and batch processing.
    """
    logger.info("Manual text ingestion request received.")
    try:
        # Using StructuralSplitter to protect code/tables in manual text
        chunks = structural_splitter.split_text(request.content)
        if not chunks:
            logger.warning("Manual text splitting resulted in 0 chunks. Skipping ingestion.")
            return {
                "status": "skipped",
                "reason": "empty_input_or_whitespace"
            }
            
        task = process_and_store_batch.delay(chunks, request.metadata)
        logger.info(f"Accepted manual ingestion. Chunks: {len(chunks)}, TaskID: {task.id}")
            
        return {
            "status": "accepted",
            "chunks_identified": len(chunks),
            "task_id": task.id
        }
    except Exception as e:
        logger.exception(f"Error in ingest_memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_hybrid_context(query_text: str, limit: int) -> List[Dict[str, Any]]:
    """
    Core retrieval function using RRF (Reciprocal Rank Fusion) + X-Algo Enhancements.
    Includes: Recency Bias, Source Weighting, and Document Diversity.
    """
    logger.info(f"Starting hybrid retrieval for query: '{query_text}'")
    
    # 0. Ensure Capability Check (Thread-safe)
    # from app.core.database import HAS_PGVECTOR, check_vector_capability
    # if not HAS_PGVECTOR:
    #     check_vector_capability()
    # from app.core.database import HAS_PGVECTOR
    
    # 1. Candidate Retrieval (Parallel Resonance)
    queries_to_search = [query_text]
    
    # --- PHASE 0: Query Expansion (Step 6) ---
    # We turn a 'lazy' user query into 3 professional, descriptive technical variations.
    try:
        if len(query_text.split()) < 8: # Only expand shorter, vaguer queries
            logger.info("Expanding query for deeper technical resonance...")
            expanded_raw = await ai_manager.call_llm(rewrite_prompt, {"query": query_text})
            # Handle possible variations separated by |
            variations = [v.strip() for v in expanded_raw.split('|') if v.strip()]
            if variations:
                queries_to_search.extend(variations[:2]) # Keep top 2 expansions + original
                logger.info(f"Query Expanded into: {queries_to_search}")
    except Exception as e:
        logger.warning(f"Query Expansion failed: {e}. Falling back to raw query.")

    # --- PHASE 15: HyDE (Hypothetical Document Embeddings) ---
    logger.info("Executing Phase 15: HyDE Generation...")
    hyde_doc = await ai_manager.generate_hyde_document(query_text)
    if hyde_doc:
        # Add the 'Fake Answer' to the pool of search vectors.
        # This forces the Vector DB to match 'Answer-to-Answer' semantics.
        queries_to_search.append(hyde_doc)
        logger.info(f"HyDE Active. Added synthetic answer to search pool. Total query branches: {len(queries_to_search)}")

    # 1b. Smart ID Extraction & Priority Guard
    import re
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    
    ids = re.findall(r'[a-zA-Z]{2,3}\d{6,8}', query_text)
    
    # 1c. Multi-Query Neural Fusion (Batch Processing)
    model_name = ai_manager.get_model_name()
    all_vector_results = []
    
    logger.info(f"Executing Batch Neural Fusion for {len(queries_to_search)} query variations...")
    query_vectors = await ai_manager.get_embeddings_batch(queries_to_search)
    
    # CHROMADB VECTOR SEARCH
    collection = get_chroma_collection()
    for sq, query_vector in zip(queries_to_search, query_vectors):
        logger.info(f"Executing Vector Branch: '{sq}'")
        try:
            res = collection.query(
                query_embeddings=[query_vector],
                n_results=50,
                include=["documents", "metadatas"]
            )
            
            if res and res["ids"] and len(res["ids"][0]) > 0:
                # Format to match postgres output: (id, content, metadata, created_at)
                for i, doc_id in enumerate(res["ids"][0]):
                    content = res["documents"][0][i]
                    meta = res["metadatas"][0][i] if res["metadatas"] else {}
                    
                    # Parse created_at from string back into datetime safely
                    created_at_time = now
                    if "created_at" in meta:
                        try:
                            # Python 3.11 supports fromisoformat seamlessly with 'Z' or +00:00
                            created_at_time = datetime.fromisoformat(meta["created_at"].replace('Z', '+00:00'))
                        except Exception:
                            pass
                            
                    all_vector_results.append((doc_id, content, meta, created_at_time))
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")

    # 1c. Global Keyword Search (Independent candidate pool)
    # text_results = [] (Disabled strict text search as vector usually covers this local need well, 
    # but we will rely solely on vectors for this ChromaDB iteration until full BM25 is added).
    text_results = []
            
    vector_results = all_vector_results
    logger.info(f"Retrieved {len(vector_results)} vector candidates (total) and {len(text_results)} text candidates.")

    # 3. Reciprocal Rank Fusion (RRF) + X-Algo Weighting (Authority & Recency)
    logger.info("Calculating RRF with X-Algo Boosters...")
    scores = {} # id -> (content, metadata, created_at, score)
    
    from datetime import datetime, timezone

    def calculate_boosted_score(rank, content, metadata, created_at, query_text):
        # Base RRF Score
        base_rrf = 1.0 / (K + rank + 1)
        
        # A. Source Authority Boost
        authority = metadata.get("authority", 1.0)
        
        # B. ID-First Match Boost (X-Algo Shield)
        id_boost = 1.0
        for uid in ids:
            if uid.lower() in content.lower():
                id_boost = 2.0 # Double priority for specific technical IDs
                break

        # C. Semantic Enrichment Boost (Phase 4)
        # If the user query matches a synthetic question, we give a major boost
        synth_boost = 1.0
        synth_qs = metadata.get("synthetic_questions", [])
        for q in synth_qs:
            # Simple overlap check or similarity (fuzzier)
            if query_text.lower() in q.lower() or q.lower() in query_text.lower():
                synth_boost = 2.5 # Predictive match is high value
                break

        # D. Recency Bias (Time Decay)
        now = datetime.now(timezone.utc)
        age_days = (now - created_at).days
        decay_factor = 1.0 / (1.0 + 0.05 * age_days)
        
        return base_rrf * authority * id_boost * synth_boost * decay_factor

    # Process Vector Ranks
    for rank, (doc_id, content, meta, created_at) in enumerate(vector_results):
        scores[doc_id] = [content, meta, created_at, calculate_boosted_score(rank, content, meta, created_at, query_text)]
    
    # Process Text Ranks
    for rank, (doc_id, content, meta, created_at) in enumerate(text_results):
        boosted = calculate_boosted_score(rank, content, meta, created_at, query_text)
        if doc_id in scores:
            scores[doc_id][3] += boosted
        else:
            scores[doc_id] = [content, meta, created_at, boosted]

    # 4. Final Re-ranking with Cross-Encoder
    top_candidates = sorted(scores.values(), key=lambda x: x[3], reverse=True)[:50] # Increased for ID sensitivity
    
    if not top_candidates:
        return []

    logger.info(f"Applying Cross-Encoder re-ranking to {len(top_candidates)} candidates...")
    pairs = [[query_text, doc[0]] for doc in top_candidates]
    rerank_scores = reranker.predict(pairs)
    
    # 5. Document Diversity Selector
    # Penalty for too many results from the same source_url or topic
    final_docs = []
    source_counts = {}
    best_score = -99.0
    
    for i, doc in enumerate(top_candidates):
        raw_score = float(rerank_scores[i])
        
        # X-Algo Protection Shields: Force results that match high-value technical patterns
        # 1. ID-Protection Shield: Technical codes like SW-12345
        has_id_match = any(uid.lower() in doc[0].lower() for uid in ids)
        
        # 2. Synthetic Match Shield: If we PREDICTED this question in Phase 4 Enrichment
        # Use fuzzy word overlap to ensure matching even with minor variations
        synth_qs = doc[1].get("synthetic_questions", [])
        has_synth_match = False
        query_words = set(query_text.lower().replace('?', '').split())
        for q in synth_qs:
            q_words = set(q.lower().replace('?', '').split())
            overlap = query_words.intersection(q_words)
            if len(overlap) >= 2 or (len(query_words) == 1 and overlap): # Match if 2 words overlap
                has_synth_match = True
                break
        
        score = 10.0 if (has_id_match or has_synth_match) else raw_score
        
        best_score = max(best_score, score)
        
        # Log Top 5 candidates for debugging
        if i < 5:
            logger.info(f"Candidate {i+1} | Raw: {raw_score:.3f} | Final: {score:.3f} | Match [ID:{has_id_match}, Synth:{has_synth_match}] | Content: {doc[0][:60]}...")

        if score < THRESHOLD:
            continue
            
        source_id = doc[1].get("source_url") or doc[1].get("topic") or "unknown"
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        
        # Diversity Filter: Cap at 2 chunks from same source in top results
        diversity_penalty = 1.0 if source_counts[source_id] <= 2 else 0.5
        adjusted_score = score * diversity_penalty
        
        final_docs.append({
            "content": doc[0], 
            "metadata": doc[1], 
            "score": adjusted_score,
            "original_score": score,
            "context_content": doc[1].get("parent_content", doc[0]) # Hierarchical fallback
        })

    # --- GRAPH EXPANSION (Phase 18) ---
    graph_context = []
    # Identify unique words/entities in original query
    # Improved splitting to handle technical terms and punctuation
    import re
    words = re.findall(r'\b\w[\w\-\']+\b', query_text.lower())
    logger.info(f"Graph Expansion: Identifying entities from {len(words)} words...")
    potential_entities = [w.strip("'s") for w in words if len(w) > 3]
    logger.info(f"Potential Entities: {potential_entities}")
    
    # Reload graph from disk if changed (Hot Loading)
    graph_manager.load_graph()
    
    for entity in potential_entities:
        related = graph_manager.get_related_facts(entity)
        if related:
            logger.info(f"Graph Match! Entity '{entity}' matched {len(related)} facts.")
        for relation in related:
            graph_context.append({
                "content": f"[Graph Fact] {relation}",
                "score": 0.95, # High priority for relational facts
                "metadata": {"type": "graph_relation", "source": "Knowledge Graph"}
            })
    
    logger.info(f"Graph context contains {len(graph_context)} items.")
    
    # Merge graph facts into final docs (limit expansion to 5 facts)
    if graph_context:
        # Deduplicate graph facts
        unique_graph = []
        seen = set()
        for g in graph_context:
            if g['content'] not in seen:
                unique_graph.append(g)
                seen.add(g['content'])
        final_docs = unique_graph[:5] + sorted(final_docs, key=lambda x: x["score"], reverse=True)
    else:
        final_docs = sorted(final_docs, key=lambda x: x["score"], reverse=True)

    # Log gaps if needed...
    if not final_docs:
        logger.warning(f"KNOWLEDGE GAP: Query '{query_text}' failed threshold (Best: {best_score:.4f})")
        # PostgreSQL is decommissioned, gap logging to DB is currently disabled.

    return sorted(final_docs, key=lambda x: x["score"], reverse=True)[:limit]

async def memory_streamer(query_text: str, limit: int):
    """
    Generator that streams hybrid-ranked search results.
    """
    try:
        results = await get_hybrid_context(query_text, limit)
        for result in results:
            yield json.dumps(result) + "\n"
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.exception(f"Error in memory_streamer: {e}")
        yield json.dumps({"error": str(e)}) + "\n"

@app.post("/chat")
async def chat_with_memory(request: ChatRequest):
    """
    Phase 17: Agentic Reasoning (The Researcher)
    Implements a self-correcting retrieval loop that performs follow-up research
    if the initial context is insufficient.
    """
    try:
        logger.info(f"Agentic Chat request: '{request.message}'")
        thought_process = []
        
        # --- ITERATION 1: Initial Search ---
        iteration = 1
        current_query = request.message
        thought_process.append(f"Iteration {iteration}: Performing initial hybrid search for '{current_query}'...")
        
        context_docs = await get_hybrid_context(current_query, request.limit)
        
        if not context_docs:
            logger.info("No context found initially. Attempting one expansion...")
            thought_process.append("No results found. Attempting query expansion...")
            # If nothing found, try to search with just keywords
            context_docs = await get_hybrid_context(current_query, request.limit) # get_hybrid_context already does expansion

        # --- AGENTIC LOOP (Max 2 Iterations) ---
        all_context_docs = list(context_docs)
        
        if all_context_docs:
            # Prepare context for evaluation
            eval_context = "\n\n".join([f"Fragment {i}: {d['content']}" for i, d in enumerate(all_context_docs)])
            
            # Call the AI Evaluator (Phase 17)
            evaluation = await ai_manager.evaluate_context_sufficiency(request.message, eval_context)
            
            is_sufficient = evaluation.get("sufficient", True)
            thought = evaluation.get("thought", "Analyzing context...")
            thought_process.append(f"Researcher Thought: {thought}")
            
            if not is_sufficient:
                iteration += 1
                missing = evaluation.get("missing_info", "contextual detail")
                suggested = evaluation.get("suggested_query", request.message)
                
                logger.info(f"Context insufficient. Missing: {missing}. Researching: {suggested}")
                thought_process.append(f"Iteration {iteration}: Initial results insufficient. Missing: {missing}. Researching: '{suggested}'...")
                
                # Perform follow-up search
                extra_docs = await get_hybrid_context(suggested, limit=5)
                
                # Deduplicate and merge
                existing_contents = {d['content'] for d in all_context_docs}
                merged_count = 0
                for d in extra_docs:
                    if d['content'] not in existing_contents:
                        all_context_docs.append(d)
                        existing_contents.add(d['content'])
                        merged_count += 1
                
                thought_process.append(f"Research Complete. Merged {merged_count} new technical fragments.")
            else:
                thought_process.append("Initial search deemed sufficient. Proceeding to synthesis.")

        if not all_context_docs:
            return {
                "answer": "I'm sorry, I don't have enough verified information in my memory to answer that accurately.",
                "sources": [],
                "thought_process": thought_process
            }

        # --- FINAL SYNTHESIS ---
        context_parts = []
        for i, d in enumerate(all_context_docs):
            meta = d.get("metadata", {})
            source_type = meta.get("type", "source")
            # Use Parent Context if available (Phase 16)
            llm_context = d.get("context_content", d["content"])
            
            block = f"--- SOURCE [{i+1}] (Type: {source_type}) ---\n{llm_context}\n"
            context_parts.append(block)

        context_text = "\n\n".join(context_parts)
        
        # Build the final Analyst Prompt
        try:
            answer = await ai_manager.call_llm(None, {
                "context": context_text,
                "question": request.message
            })
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            answer = "[Synthesis Offline] - Data retrieved but brain failed to process."

        return {
            "answer": answer,
            "sources": all_context_docs[:request.limit], # Return original limit to UI
            "model": ai_manager.get_model_name(),
            "thought_process": thought_process
        }

    except Exception as e:
        logger.exception(f"Error in agentic chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in chat_with_memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/stream")
async def stream_memories(query: str, limit: int = 5):
    """
    Retrieves memories using Hybrid Search (Vector + Text) and streams results.
    """
    logger.info(f"Stream search request for query: '{query}'")
    return StreamingResponse(
        memory_streamer(query, limit),
        media_type="application/x-ndjson"
    )

@app.get("/search")
async def dashboard_search(query: str, limit: int = 5):
    """
    Unified search endpoint for the Admin Dashboard.
    Returns both the retrieved context and a generated answer.
    """
    try:
        # 1. Get Hybrid Context
        context_docs = await get_hybrid_context(query, limit)
        
        # 2. Generate Answer (Master Analyst Synthesis)
        answer = "No relevant neural fragments passed the sensitivity threshold for this query."
        if context_docs:
            context_parts = [f"Source [{i+1}]: {d['content']}" for i, d in enumerate(context_docs)]
            context_text = "\n\n".join(context_parts)
            
            prompt = ChatPromptTemplate.from_template(
                "You are an AI Lead Engineer performing a technical synthesis for the Admin Dashboard. "
                "Based on these {count} fragments: \n\n"
                "{context}\n\n"
                "Synthesize a full technical report answering: {question}"
            )
            try:
                # Groq Cloud Synthesis via Llama 3.3 70B
                answer = await ai_manager.call_llm(None, {
                    "context": context_text, 
                    "question": query,
                    "count": len(context_docs)
                })
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                answer = "[Neural Synthesis Offline - Using raw Retrieval-Only mode]"

        return {
            "query": query,
            "context": context_docs,
            "answer": answer
        }
    except Exception as e:
        logger.exception(f"Dashboard search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
