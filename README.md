# 🧠 Enterprise Neural Fusion RAG (v3.0.0)

A state-of-the-art, industrial-grade Retrieval-Augmented Generation (RAG) and Agentic Search system. Featuring a **3,712-dimension hybrid dense ensemble**, a **32,768-dimension sparse index** for exact keyword matching, and a **Milvus + Neo4j GraphRAG fusion** with an active **Truth Guard** anti-hallucination layer.

---

## 🚀 Key Architectural Features

*   **Ensemble Dense/Sparse Embeddings**: Combines 5 local models (`bge-m3`, `canine-s` character-level ID shield, `codebert-base` for code, `mpnet`, and `minilm`) into a 3,712-D dense vector and a 32,768-D sparse matrix to guarantee high recall of exact technical identifiers, acronyms, and source code.
*   **Multimodal Layout Parsing (Docling & Llama 3.2 Vision)**: Converts PDFs, Word docs, and Excel sheets into structured Markdown using Docling, harvests embedded images with PyMuPDF, and auto-describes them using Llama 3.2 Vision to enable text-based diagram retrieval alongside CLIP vector indexing.
*   **Knowledge Graph Synergy (Neo4j GraphRAG)**: Extracts Subject-Relation-Object triplets during ingestion and queries Neo4j via Cypher at search time, injecting structural relationships directly into the LLM context window.
*   **Active Truth Guard Audit**: Dynamically extracts facts from the generated LLM response draft and validates them against the Neo4j graph. If contradictions (e.g. port mismatch, status mismatch) are found, the guard automatically forces a correction rewrite before delivery.
*   **Model Context Protocol (MCP) Server**: Exposes the RAG system directly to external AI assistants (like Claude Code, Cursor, or Claude Desktop) using standard SSE (Server-Sent Events) or stdio transport pipes.

---

## 📁 Project Directory Layout

```text
/project-root
├── app/
│   ├── api/                # FastAPI routers (chat, search, feedback) and assets
│   ├── core/               # Singleton clients (Milvus, Neo4j, Redis) & AIManager
│   ├── services/           # Ingestion pipelines (Docling, splitters, retrieval, Celery)
│   └── mcp/                # Model Context Protocol (MCP) server endpoints & catalog
├── scripts/                # Self-healing diagnostics, backup, and restore tools
├── backups/                # Target folder for automated DB snapshots
├── media/                  # Image assets harvested from ingested documents
├── models_cache/           # Local folder caching SentenceTransformers & CLIP
├── logs/                   # Standard output log files
├── tests/                  # Pytest unit, splitter, and integration tests
├── Dockerfile              # Containerization definition
├── docker-compose.yml      # Multi-container orchestrator (FastAPI, Redis, Milvus, Neo4j)
└── requirements.txt        # Python dependency manifest
```

---

## ⚙️ Setting Up the System

### 1. Pre-requisites & Installation
Ensure you have Python 3.10+ and Docker installed. Install local dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root workspace directory:
```env
# Large Language Model APIs
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Databases & Infrastructure Links
REDIS_URL=redis://localhost:6379/0
MILVUS_URI=http://localhost:19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password_here

# Scaling Embeddings (Optional)
OFFLOAD_EMBEDDINGS=false
EMBEDDING_SERVER_URL=http://localhost:8000
```

### 3. Launching Services
Start the full local microservices stack (FastAPI Backend + Redis + Milvus Lite + Neo4j + Celery Worker) using Docker Compose:
```bash
# Launch background services
docker compose up -d
```

For high-throughput document onboarding queues, scale up the dedicated OCR/Docling Celery worker containers:
```bash
docker compose --profile scale up -d --scale worker-heavy=3
```

---

## 🔌 Model Context Protocol (MCP) Server

Our built-in MCP server (`app/mcp/mcp_server.py`) enables Claude Code, Cursor, or Claude Desktop to interface directly with your corporate knowledge base.

### Running the MCP Server
*   **SSE Transport Mode** (Standard HTTP/Uvicorn, defaults to port `9382`):
    ```bash
    python app/mcp/mcp_server.py --transport sse --port 9382
    ```
*   **stdio Transport Mode** (Direct command-line stream pipe):
    ```bash
    python app/mcp/mcp_server.py --transport stdio
    ```

### Exposed Capabilities
1.  **Core RAG Tools**:
    *   `ingest_company_document`: Submits files, text, or web links to background processing queues.
    *   `search_and_execute`: Queries the vector-graph index and maps matches to available corporate diagnostics tools.
2.  **Dynamic Catalog Tools**: Dynamically loaded from `service_catalog.json` (e.g. `get_pof_report`, `get_rca_report` for SRE diagnostics).
3.  **Active Learning Feedback**: Whenever a client executes a tool via MCP, the execution output is converted into a markdown `ACTION TRACE` and async-indexed back into Milvus, allowing the agent to remember its diagnostic history.
4.  **Resources**: Access live system data via resources like `company://reports/knowledge-gaps` (queries yielding low-confidence results).

---

## 🛠️ Operations & Maintenance

### 🩺 System Self-Healing (Neural Doctor)
Validate system configuration, python/node packages, environment credentials, and active port connections:
```bash
# Check system status
python scripts/doctor.py

# Check status and auto-fix configuration/dependency issues
python scripts/doctor.py --fix
```

### 💾 Snapshot Backups & Recovery
Snapshots are exported as timestamped compressed tarballs in `backups/`:
```bash
# Take system snapshot
python scripts/backup_system.py

# Restore database states from backup archive
python scripts/restore_system.py backups/backup_2026-07-01_14-00-00.tar.gz
```

### 🧪 Running the Test Suite
Ensure the logical splitter boundaries, layout parsers, RRF fusion, and retrieval logic are fully operational:
```bash
# Run tests
pytest -v
```

### ⚡ Performance Tuning & Memory Optimization

*   **Model Loading & Execution Strategy**: 
    *   *Parallel Initialization*: Local SentenceTransformers and CLIP layers are loaded concurrently using `ThreadPoolExecutor` inside `AIManager` at startup to minimize CPU wait times.
    *   *Lazy Loading*: The Cross-Encoder reranking model is lazy-loaded on demand during the first retrieval run.
    *   *Offloaded Mode*: When `OFFLOAD_EMBEDDINGS=true` is configured, local models are bypassed entirely, enabling instant API startup.
*   **Celery Queue Partitioning**: 
    *   `default` queue: Processes fast query updates, user votes, and action memory traces.
    *   `heavy_ingest` queue: Handles Docling structural parsing, Groq Vision extraction, and parallel multi-model vector calculations.
