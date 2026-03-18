# Features and Workflows: AI Knowledge Base

This document provides a detailed technical overview of every feature in the AI Knowledge Base, how data flows through the system during ingestion (Write) and retrieval (Read), and the primary use cases.

---

## 🧠 Core Intelligence Features

### 1. Hybrid Neural Fusion (1,920-Dimension Vector)
Unlike conventional RAG systems that rely on a single embedding model, our engine uses a **5x Neural Fusion Ensemble**. 
- **The Process**: Every text fragment is processed by five specialized models (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-cos-v1`, `all-MiniLM-L12-v2`, `msmarco-MiniLM-L6-cos-v5`) simultaneously.
- **The Result**: A massive 1,920-dimension vector fingerprint.
- **Benefit**: This deep granularity ensures that technical IDs, alphanumeric codes, and subtle protocol variations are captured with perfect precision.

### 2. Dual-LLM Resonance Strategy (Groq + LangChain)
The system split between two specialized LLMs to maximize speed and intelligence.

| Role | Model | Technology | Purpose |
| :--- | :--- | :--- | :--- |
| **Reasoning Engine** | `Llama-3.3-70B-versatile` | LangChain `ChatGroq` | Final synthesis, HyDE, and Agentic Reasoning. |
| **Ingestion Engine** | `Llama-3.1-8B-instant` | LangChain `ChatGroq` | Background bulk tasks (Synthetic Qs, GraphRAG triplets). |

- **LangChain Integration**: Built-in robust retry logic (6x for 70B, 3x for 8B) to handle API rate limits gracefully.

### 3.HyDE (Hypothetical Document Embeddings)
- **The Process**: The **Llama-3.3-70B** model generates a factual hypothetical answer to your query *before* searching.
- **The Search**: We vectorize this "fake" answer and use it to search the database.
- **Benefit**: It performs "Answer-to-Answer" matching, significantly improving technical retrieval accuracy.

### 4. Hierarchical Context Retrieval (Parent Context)
- **The Process**: The search finds specific fragments, but the system automatically retrieves the larger "Parent Context" (surrounding text) if available.
- **Benefit**: Prevents the LLM from losing the "big picture" of a technical explanation when searching across narrow chunks.

### 5. Agentic Reasoning (The Research Loop)
- **The Process**: After retrieval, the **Llama-3.3-70B** acts as a "Technical Auditor," evaluating if the found context is sufficient.
- **The Loop**: If information is missing, it automatically generates a targeted search query and performs a second-pass search.

### 6. GraphRAG & Expansion
- **Extraction**: The **8B model** parses ingestion text into (Subject | Relation | Object) triplets.
- **Expansion**: During search, the system identifies entities in your query and pulls related facts directly from the Knowledge Graph to enrich the context.

---

## 🚀 Advanced Retrieval Logic (X-Algo)

The system doesn't just "match" vectors; it uses a proprietary **X-Algo** scoring system to prioritize the right data.

### 1. Reciprocal Rank Fusion (RRF)
Combines results from multiple search branches (raw query, expanded queries, and HyDE) into a single, mathematically optimized list.

### 2. X-Algo Boosters (Shields)
- **ID-Protection Shield**: Automatically identifies technical IDs (e.g., `HUB402`) using regex and applies a **2x priority boost** to matching fragments.
- **Semantic enrichment Boost**: If a search matches a "Synthetic Question" we predicted during ingestion, the result receives a **2.5x priority boost**.
- **Recency Bias**: Prioritizes newer documents using a Time Decay factor to ensure you see the most up-to-date manual pages.
- **Source Authority**: Weighting based on document importance (e.g., official manuals vs. chat logs).

### 3. Cross-Encoder Reranking
The top 50 candidates from RRF are passed through a **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) for a final, high-precision relevance check.

### 4. Document Diversity Selector
Ensures you don't get buried in results from a single document. It applies a penalty if more than 2 chunks originate from the same source, forcing the AI to consider diverse information.

---

## 🛠️ Workflows: How it Works

### 🖋️ The "Write" Flow (Ingestion Pipeline)

When a document enters the system, it undergoes a high-precision transformation before being committed to the vector matrix:

1.  **Source Parsing & Authority Tuning**:
    *   The system identifies the file type and applies specialized loaders (e.g., Structural Excel loading).
    *   **X-Algo Booster**: It assigns an **Authority Weight** (e.g., **1.2x** for official manuals, **0.8x** for chat logs) to prioritize verified data.
2.  **Sub-Batching**: Chunks are processed in sub-batches of 100 via **Celery + Redis** to maintain high throughput without blocking the UI.
3.  **Local Neural Fusion**: Every chunk is vectorized into a **1,920-dimension fingerprint** using the 5-model local ensemble.
4.  **Semantic Enrichment (The 8B Model)**:
    *   **Synthetic Query Generation**: Generates 3 human-like questions per chunk to ensure "plain English" searchability.
    *   **Knowledge Graph Mapping**: Extracts technical relationships (triplets) to build connections between documents.
5.  **Hierarchical Context Logic**: If the document is structured, the system stores the **Parent Context** alongside the fragment to ensure the AI never loses the "big picture."
6.  **Atomic Milvus Commit**:
    *   Metadata is cleaned, flattened, and enriched with timestamps and model IDs.
    *   A single high-speed commit sends the vectors and enriched metadata into **Milvus Lite** for instant indexing.

### 🔍 The "Read" Flow (Retrieval Pipeline)

When you query the system, it triggers a multi-stage cognitive search process:

1.  **Technical Expansion (LangChain + 70B)**: The system turns vague user queries (e.g., "fix it") into descriptive technical variations to ensure deep resonance.
2.  **HyDE & Graph Expansion**:
    *   **HyDE**: The **70B model** writes a hypothetical "perfect answer" to search against.
    *   **Graph Facts**: Entities are identified, and related facts are pulled from the Knowledge Graph.
3.  **Multi-Branch Vector Search**: The system performs parallel searches in **Milvus Lite** for the raw query, technical variations, and the HyDE document.
4.  **Reciprocal Rank Fusion (RRF)**: All search results are merged into a single list based on their relative rankings across branches.
5.  **X-Algo Scoring & Neural Boosting**:
    *   **ID-Protection Shield**: Specific technical IDs (regex-matched) grant a **2x priority boost**.
    *   **Predictive Match Shield**: Matching a pre-generated synthetic question grants a **2.5x priority boost**.
    *   **Time Decay**: Newer logs/manuals are automatically weighted higher.
6.  **Hierarchical Recall**: Individual fragments are swapped for their larger **Parent Context** blocks to preserve technical meaning.
7.  **Cross-Encoder Reranking**: The top 50 candidates are re-scored by a specialized reranker model for absolute relevance.
8.  **Agentic Audit (The Researcher)**: The **70B model** audits the retrieved context. If it finds a "Knowledge Gap," it automatically triggers a second research pass.
9.  **Final Neural Synthesis**: The **70B model** synthesizes the final report, citing sources and formatting technical details for the dashboard.

---

## 🖥️ System Infrastructure

- **Milvus Lite (Flat Index)**: Configured with a `FLAT` index for **100% mathematical precision**—essential for local retrieval where every ID digit matters.
- **Dynamic Schema**: Uses dynamic fields to store arbitrary JSON metadata alongside vectors without performance loss.
- **System Pulse**: Live health monitoring for Redis, Milvus, and Groq connections on the Admin Dashboard.
- **Local Persistence**: All model weights and vector data are stored in host-mounted directories (`./models_cache`, `./milvus_data`).
