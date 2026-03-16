import os
import sys
import asyncio
import re
from datetime import datetime, timezone
# Add project root to path
sys.path.append(os.getcwd())

from app.core.database import pool
from app.core.ai_manager import ai_manager
from sentence_transformers import CrossEncoder

# Constants
K = 60
THRESHOLD = 0.2
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def diag_search(query_text):
    print(f"--- DIAGNOSTIC SEARCH: '{query_text}' ---")
    
    # 1. ID Extraction
    ids = re.findall(r'[a-zA-Z]{2,3}\d{6,8}', query_text)
    print(f"EXTRACTED_IDS: {ids}")
    
    id_filters = " OR ".join([f"content ILIKE '%%{id}%%'" for id in ids])
    if id_filters:
        id_filters = f"OR {id_filters}"
    
    # 2. Vector
    query_vector = await ai_manager.get_embeddings(query_text)
    model_name = ai_manager.get_model_name()
    print(f"MODEL: {model_name} | DIMS: {len(query_vector)}")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # 2a. Vector Search
            cur.execute("""
                SELECT id, content, metadata, created_at
                FROM ai_memory
                WHERE (metadata->>'embedding_model' = %s OR metadata->>'embedding_model' IS NULL)
                ORDER BY embedding <=> cast(%s as vector)
                LIMIT 100;
            """, (model_name, str(query_vector).replace(' ', ''),))
            vector_results = cur.fetchall()
            print(f"VECTOR_HITS: {len(vector_results)}")

            # 2b. Keyword Search
            cur.execute(f"""
                SELECT id, content, metadata, created_at
                FROM ai_memory
                WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
                   OR content ILIKE %s
                   {id_filters}
                LIMIT 100;
            """, (query_text, f"%{query_text}%"))
            text_results = cur.fetchall()
            print(f"TEXT_HITS: {len(text_results)}")

    # 3. RRF with Boosting
    def calculate_boosted_score(rank, content, metadata, created_at):
        base_rrf = 1.0 / (K + rank + 1)
        id_boost = 1.0
        for uid in ids:
            if uid.lower() in content.lower():
                id_boost = 2.0
                break
        
        now = datetime.now(timezone.utc)
        age_days = (now - created_at).days
        decay_factor = 1.0 / (1.0 + 0.05 * age_days)
        return base_rrf * id_boost * decay_factor

    scores = {}
    for rank, (doc_id, content, meta, created_at) in enumerate(vector_results):
        scores[doc_id] = [content, meta, created_at, calculate_boosted_score(rank, content, meta, created_at)]
    
    for rank, (doc_id, content, meta, created_at) in enumerate(text_results):
        boosted = calculate_boosted_score(rank, content, meta, created_at)
        if doc_id in scores:
            scores[doc_id][3] += boosted
        else:
            scores[doc_id] = [content, meta, created_at, boosted]

    top_candidates = sorted(scores.values(), key=lambda x: x[3], reverse=True)[:50]
    print(f"CANDIDATES_FOR_RERANK: {len(top_candidates)}")
    
    if not top_candidates:
        print("FAIL: No candidates reached the re-ranker.")
        return

    # 4. Re-ranking
    pairs = [[query_text, doc[0]] for doc in top_candidates]
    rerank_scores = reranker.predict(pairs)
    
    print("\n--- TOP 10 RERANK SCORES ---")
    for i, doc in enumerate(top_candidates[:10]):
        raw_score = float(rerank_scores[i])
        found_id = next((uid for uid in ids if uid.lower() in doc[0].lower()), None)
        id_protection = 5.0 if found_id else 0.0
        score = raw_score + id_protection
        print(f"Rank {i+1} | Score: {score:.4f} (Raw: {raw_score:.4f}) | {'MATCH ('+found_id+')' if found_id else 'NO_ID'}")
        print(f"  CONTENT: {doc[0][:300]}...")

if __name__ == "__main__":
    asyncio.run(diag_search("i want summary of chat from jwx1369347"))
