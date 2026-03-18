# AI Knowledge Base Beyond (Custom-Built Intelligence Engine)

A high-precision, zero-footprint Knowledge Base system engineered for technical autonomy. This project represents a shift from standard "off-the-shelf" RAG solutions to a custom-engineered **Hybrid Neural Fusion** architecture designed for **collaborative teams** who prioritize **data privacy and sovereign intelligence**.

## 🛤️ The Trail: Our Evolutionary Journey

The system has matured through three distinct architectural epochs:

1.  **Cloud Dependency**: Relied on external APIs (OpenAI/Pinecone) with high latency and privacy risks.
2.  **Heavy Local (Ollama + Postgres)**: Moved to local hosting, but was bogged down by the heavy footprint of PostgreSQL (pgvector) and the CPU overhead of Ollama.
3.  **Enterprise Neural Fusion (Current)**: A production-ready architecture using **Milvus Standalone** for multi-container orchestration, a **5x MiniLM Ensemble** for ultra-fast vectorization, and **Groq Cloud** for high-speed agentic reasoning.

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

### HyDE (Hypothetical Document Embeddings)
The system doesn't just search for your question. It uses the LLM to generate a **hypothetical perfect answer** (a "fake" manual page), vectorizes *that*, and searches for "Answer-to-Answer" matches. This bypasses the keyword mismatch problem.

### Agentic Reasoning (The Researcher)
The retrieval is self-correcting. If the initial search results are deemed "insufficient" by the AI, it automatically spawns a follow-up research task with a new query to fill the knowledge gaps before presenting the final answer.

### GraphRAG (Knowledge Graph)
The system extracts (Subject | Relation | Object) triplets during ingestion, building a persistent Knowledge Graph. This allows the AI to traverse relationships (e.g., "System A *uses* Protocol B") even if they aren't mentioned in the same document chunk.

---

## 🛠️ The Tech Stack

*   **Logic Core**: FastAPI (Asynchronous Python 3.11+)
*   **Vector Matrix**: **Milvus Standalone** (Enterprise-grade, Multi-container safe)
*   **Storage Glue**: Minio (Object Storage) & Etcd (Metadata)
*   **Background Ops**: Celery + Redis (Asynchronous Ingestion)
*   **Neural Ensemble**: 5x Sentence-Transformers (Local CPU/GPU)
*   **Cognitive Layer**: Groq Cloud API (Llama-3.3-70b-versatile)
*   **UI Foundation**: Modern Vanilla JS + CSS (Admin Dashboard)

---

## 📚 Detailed Documentation

For a comprehensive breakdown of how the system works, including the **Write** and **Read** flows and more specialized features, check out our technical deep-dive:

👉 **[Features & Workflows Guide](docs/features_and_workflows.md)**

---

## ✨ Key Features

*   **Hot-Reloading (Live Mode)**: Your local code is mounted directly into Docker. Edit your code on your laptop, and the containers update instantly.
*   **Shielded ID Protection**: X-Algo boosters ensure that technical IDs (e.g., `HUB402`, `SW-12345`) are never lost in semantic noise.
*   **Structural Content Mapping**: Specialized loaders for Excel and PDF that preserve tables and row-level relationships.
*   **Heavy Parsing Mode**: OCR-capable ingestion for scanned technical diagrams and complex documentation.
*   **Milvus Standalone Architecture**: Optimized for parallel access from the API, Worker, and MCP server.

---

## 🏁 Getting Started (Linux/Docker)

Follow these steps to get your private intelligence engine running in minutes.

### 1. Environment Configuration
Ensure your `.env` file contains your Groq API key. All paths have been pre-configured to be relative and Linux-compatible.
```env
GROQ_API_KEY=gsk_xxxx...
REDIS_URL=redis://redis:6379/0
MILVUS_URI=http://milvus:19530
HF_HOME=./models_cache
```

### 2. Launch the Stack
Run the following command in the root directory:
```bash
docker-compose up -d
```

### 3. Monitor the "Neural Boot Sequence"
The first time you start, the system downloads and loads 5 heavy AI models into memory. Monitor this in real-time:
```bash
docker-compose logs -f
```
Wait for the line: `Uvicorn running on http://0.0.0.0:8000`.

### 4. Access the Frontend Dashboard
Once the boot sequence is complete, open your browser to:

👉 **[http://localhost:8000](http://localhost:8000)**

*The source files for this dashboard are located at: `app/api/static/index.html`*

## 🕹️ How to Use the Frontend Dashboard

Once the containers are running, you can access the full intelligence suite in your browser.

1.  **Open the Admin UI**: Navigate to **[http://localhost:8000](http://localhost:8000)**.
2.  **Ingest Documents**:
    *   Find the **Pulse Ingestion** or **Upload Section**.
    *   Drag and drop your PDFs, Excel files, or technical documents.
    *   The **Worker** container will automatically begin the 5-layer vectorization process.
3.  **Perform Neural Queries**:
    *   Type your question in the search bar.
    *   The system will use **HyDE** and **Reranking** to find the most technically accurate matches.
4.  **Monitor System Health**:
    *   The dashboard includes a live **System Pulse** section.
    *   It tracks the connectivity of **Redis**, **Milvus**, and **Groq** in real-time.

---

## 📂 Project Structure & Persistence

Your data is safely persisted on your host machine in these directories:
*   `./milvus_data`: Vector database and metadata (Milvus/Etcd/Minio).
*   `./models_cache`: AI model weights (HuggingFace).
*   `./logs`: Application logs for debugging.
*   `./uploads`: Temporary storage for document ingestion.
*   `./media`: Extracted images from PDFs.

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

