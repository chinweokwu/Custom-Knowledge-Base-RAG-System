# Enterprise Neural Fusion RAG (v3.0.0)

A high-performance, industrial-grade RAG system featuring a 3,712-dimension hybrid dense ensemble, 32,768-dimension sparse matrix, native Milvus + Neo4j fusion, and background worker orchestration.

---

## 📁 Project Structure

```text
/project-root
├── app/
│   ├── api/                # FastAPI routes, schemas, and static frontend assets
│   ├── core/               # Configuration, DB client singletons, and AI managers
│   ├── services/           # Heavy business logic (retrieval, Docling loaders, tasks)
│   └── mcp/                # Model Context Protocol (MCP) server endpoints
├── scripts/                # Operations, backups, diagnostics, and recovery scripts
├── backups/                # Storage directory for automated database backups
├── media/                  # Extracted document image elements
├── models_cache/           # Local LLM/Embedding cache directory
├── logs/                   # System execution logs
├── tests/                  # System, splitter, and integration tests
├── Dockerfile              # Application Docker image configuration
├── docker-compose.yml      # Orchestration (FastAPI + Workers + Redis + Milvus + Neo4j)
└── requirements.txt        # Backend dependencies
```

---

## 🧠 Core Production Architecture

The system uses a decoupled, asynchronous, worker-driven architecture:
*   **Decoupled Async Workload**: Celery handles heavy layout extraction, embedding generation, and triplet extraction in background processes.
*   **Ensemble Dense/Sparse Embeddings**: Multi-model dense vector fusion (3,712-D) coupled with lexical sparse representation (32,768-D) via BGE-M3.
*   **Just-in-Time (JIT) Purging**: Wipes previous document versions *immediately* before database commits, ensuring zero search blackouts during processing.

---

## 🚀 Getting Started

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
# AI APIs
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Databases
REDIS_URL=redis://localhost:6379/0
MILVUS_URI=http://localhost:19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Model Offloading (Optional - for high-throughput scaling)
OFFLOAD_EMBEDDINGS=false
EMBEDDING_SERVER_URL=http://localhost:8000
```

### 3. Launching

Start the local RAG microservices stack using Docker Compose:
```bash
# Start standard services (FastAPI + Redis + Milvus + Neo4j + Default Worker)
docker compose up -d
```

For heavy ingestion queues, spin up heavy-concurrency workers using the scale profile:
```bash
docker compose --profile scale up -d --scale worker-heavy=3
```

---

## 🛠️ Operations & Maintenance

### 💾 Automated Backup & Recovery
Automated backup and restore scripts handle SQLite snapshots, Redis configurations, and Milvus/Neo4j volume exports:

```bash
# Run a live snapshot backup (creates compressed tarball in backups/)
python scripts/backup_system.py

# Restore the system to a previous state from an archive
python scripts/restore_system.py backups/backup_2026-06-06_20-00-00.tar.gz
```

### ⚡ Performance Tuning & Memory Optimization
*   **Lazy-Loaded Models**: Cross-Encoder and SentenceTransformers models are lazy-loaded on demand to ensure FastAPI boots up in <1 second.
*   **Worker Queue Partitioning**: 
    *   `default` queue: Processes fast query synthesis and memory traces.
    *   `heavy_ingest` queue: Handles OCR parsing, structural splitting, and vector generation.

---


