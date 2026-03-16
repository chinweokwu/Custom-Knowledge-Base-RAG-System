# AI Knowledge Base Beyond (Custom-Built Intelligence Engine)

A high-precision, zero-footprint Knowledge Base system engineered for technical autonomy. This project represents a shift from standard "off-the-shelf" RAG solutions to a custom-engineered **Hybrid Neural Fusion** architecture designed for **collaborative teams** who prioritize **data privacy and sovereign intelligence**.

## 🛤️ The Trail: Our Evolutionary Journey

The system has matured through three distinct architectural epochs:

1.  **Phase I: Cloud Dependency**: Relied on external APIs (OpenAI/Pinecone) with high latency and privacy risks.
2.  **Phase II: Heavy Local (Ollama + Postgres)**: Moved to local hosting, but was bogged down by the heavy footprint of PostgreSQL (pgvector) and the CPU overhead of Ollama.
3.  **Phase III: Hybrid Neural Fusion (Current)**: A custom-built, optimized architecture that utilizes **ChromaDB** for zero-footprint portability, a **5x MiniLM Ensemble** for blazing-fast local vectorization, and **Groq Cloud (Llama-3.3 70B)** for high-speed agentic reasoning.

---

## 🧠 Core Intelligence: Hybrid Neural Fusion

Unlike standard RAG systems that use a single embedding model, this system employs a **5x Neural Fusion Ensemble**.

### The 1920-Dimension Vector
Every document fragment is passed through five specialized models simultaneously:
*   **all-MiniLM-L6-v2**: General semantic mapping.
*   **paraphrase-MiniLM-L6-v2**: Semantic clarity and paraphrasing.
*   **multi-qa-MiniLM-L6-cos-v1**: Technical QA and code-specific indexing.
*   **all-MiniLM-L12-v2**: Deep hierarchical reasoning.
*   **msmarco-MiniLM-L6-cos-v5**: High-precision search ranking.

**Result**: A massive **1,920-dimension vector fingerprint** that captures technical nuances (Site IDs, Alphanumeric codes, Protocol logs) that standard GPT-4 embeddings often miss.

---

## 🚀 Advanced Cognitive Phases

The engine operates on a sophisticated retrieval pipeline, featuring:

### Phase 15: HyDE (Hypothetical Document Embeddings)
The system doesn't just search for your question. It uses the LLM to generate a **hypothetical perfect answer** (a "fake" manual page), vectorizes *that*, and searches for "Answer-to-Answer" matches. This bypasses the keyword mismatch problem.

### Phase 17: Agentic Reasoning (The Researcher)
The retrieval is self-correcting. If the initial search results are deemed "insufficient" by the AI, it automatically spawns a follow-up research task with a new query to fill the knowledge gaps before presenting the final answer.

### Phase 18: GraphRAG (Knowledge Graph)
The system extracts (Subject | Relation | Object) triplets during ingestion, building a persistent Knowledge Graph. This allows the AI to traverse relationships (e.g., "System A *uses* Protocol B") even if they aren't mentioned in the same document chunk.

---

## 🛠️ The Tech Stack

*   **Logic Core**: FastAPI (Asynchronous Python 3.11+)
*   **Vector Matrix**: ChromaDB (Zero-Footprint, Portable)
*   **Background Ops**: Celery + Redis (Asynchronous Ingestion)
*   **Neural Ensemble**: 5x Sentence-Transformers (Local CPU/GPU)
*   **Cognitive Layer**: Groq Cloud API (Llama-3.3-70b-versatile)
*   **UI Foundation**: Modern Vanilla JS + CSS (Admin Dashboard)

---

## ✨ Key Features

*   **Shielded ID Protection**: X-Algo boosters ensure that technical IDs (e.g., `HUB402`, `SW-12345`) are never lost in semantic noise.
*   **Structural Content Mapping**: Specialized loaders for Excel and PDF that preserve tables and row-level relationships.
*   **Heavy Parsing Mode**: OCR-capable ingestion for scanned technical diagrams and complex documentation.
*   **Real-Time Health Monitoring**: Live dashboard tracking Redis heartbeat, ChromaDB counts, and Neural Ensemble readiness.

---

## 🏁 Getting Started

Choose your preferred deployment method below.

### Option 1: Docker Deployment (Recommended)

Quickest way to get the full stack running with cross-platform compatibility.

**Prerequisites:** Docker and Docker Compose installed.

1.  **Configure Environment**:
    Ensure your `.env` file has the `GROQ_API_KEY`. The D-drive paths in `.env` are ignored by Docker in favor of internal volumes.

2.  **Launch the Stack**:
    ```bash
    docker-compose up -d
    ```

3.  **Access the Admin Dashboard**:
    Open [http://localhost:8000](http://localhost:8000)
    
    The dashboard provides a visual interface for:
    - **Neural Queries**: Real-time RAG searching and AI synthesis.
    - **Pulse Center**: Drag-and-drop ingestion of technical documents.
    - **Memory Explorer**: Live view of vector database fragments.
    - **System Health**: Real-time monitoring of Redis, ChromaDB, and Groq.

---

### Option 2: Local Manual Setup

Best for development or debugging.

**Prerequisites:** Python 3.11+, Redis Server.

1.  **Environment Configuration**:
    Create a `.env` file in the root directory (refer to `.env.example`):
    ```env
    GROQ_API_KEY=your_key_here
    REDIS_URL=redis://localhost:6380/0
    HF_HOME=D:\AI_Models_Cache
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start Services**:
    If on Windows, you can use the provided orchestrator:
    ```powershell
    ./start_app.bat
    ```
    
    Or start manually:
    - **Redis**: Start on port 6380.
    - **Worker**: `celery -A app.services.tasks worker --loglevel=info -P solo`
    - **API**: `python -m app.api.main`
    - **MCP**: `python -m app.mcp.mcp_server --transport sse --port 9382`

4.  **Access the Admin UI**:
    Open [http://localhost:8000](http://localhost:8000)

---

## 📂 Project Structure & Persistence

When running via Docker, the following host directories are used for persistence:
*   `./chroma_db`: Vector database storage.
*   `./models_cache`: AI model weights (HuggingFace).
*   `./logs`: Application and service logs.
*   `./uploads`: Temporary file storage for ingestion.

---

## 👨‍💻 Developed By

**Morah Paul**
Senior Software and Automation Engineer

This system was originally engineered for the **Knowledge Base Copilot project** at **Huawei NOC Nigeria** to be utilized as a secure, in-house intelligence asset.

📧 **Email**: [paulcmorah@gmail.com](mailto:paulcmorah@gmail.com)
📱 **Phone**: [+234 813 227 0970](tel:+2348132270970)
🔗 **LinkedIn**: [Morah Paul](https://www.linkedin.com/in/morah-paul/)

---
⭐ **Give me a star!** If you find this project useful, please give it a star on GitHub.

