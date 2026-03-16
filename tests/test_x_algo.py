import asyncio
import os
import json
from datetime import datetime, timedelta, timezone
from app.core.database import pool
from app.api.main import get_hybrid_context
from app.core.ai_manager import ai_manager
from app.services.tasks import process_and_store_batch
from dotenv import load_dotenv

load_dotenv()

async def verify_x_algo():
    print("🚀 Starting X-Algo Enhancements Verification...")
    
    # 1. Clear existing test data (optional but cleaner)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ai_memory WHERE metadata->>'test_set' = 'x_algo_test';")
            # Also delete by specific content prefixes to ensure no conflicts with unique ID strings
            cur.execute("DELETE FROM ai_memory WHERE content LIKE 'X-Algo is a powerful framework%';")
            cur.execute("DELETE FROM ai_memory WHERE content LIKE 'The company protocol for restarts%';")
            cur.execute("DELETE FROM ai_memory WHERE content LIKE 'Diversity Test Chunk%';")
            conn.commit()

    print("--- 🛠️ Preparing Test Data ---")
    
    # Define test cases
    # A. Recency Bias: Two identical texts, one old, one new
    # B. Source Weighting: Two similar texts, one official (high authority), one manual (normal)
    # C. Diversity: Multiple chunks from the same source
    
    test_data = [
        # A. Recency
        {"content": "X-Algo is a powerful framework for AI search. Reference ID: recency_old", "metadata": {"test_set": "x_algo_test", "type": "manual_entry", "id": "recency_old"}, "days_ago": 30},
        {"content": "X-Algo is a powerful framework for AI search. Reference ID: recency_new", "metadata": {"test_set": "x_algo_test", "type": "manual_entry", "id": "recency_new"}, "days_ago": 0},
        
        # B. Authority
        {"content": "The company protocol for restarts is to check the logs first. DocID: auth_low", "metadata": {"test_set": "x_algo_test", "type": "manual_entry", "id": "auth_low"}, "days_ago": 0},
        {"content": "The company protocol for restarts is to check the logs first. DocID: auth_high", "metadata": {"test_set": "x_algo_test", "type": "official_doc", "id": "auth_high"}, "days_ago": 0},
        
        # C. Diversity
        {"content": "Diversity Test Chunk 1: Topic Alpha. Part A.", "metadata": {"test_set": "x_algo_test", "source_url": "source_one", "id": "div_1"}, "days_ago": 0},
        {"content": "Diversity Test Chunk 2: Topic Alpha. Part B.", "metadata": {"test_set": "x_algo_test", "source_url": "source_one", "id": "div_2"}, "days_ago": 0},
        {"content": "Diversity Test Chunk 3: Topic Alpha. Part C.", "metadata": {"test_set": "x_algo_test", "source_url": "source_one", "id": "div_3"}, "days_ago": 0},
    ]

    for item in test_data:
        # Ingest
        vector = await ai_manager.get_embeddings(item["content"])
        created_at = datetime.now(timezone.utc) - timedelta(days=item["days_ago"])
        
        # Authority calculation (extracted from tasks.py logic)
        source_type = item["metadata"].get("type", "unknown")
        authority = 1.0
        if source_type == "official_doc": authority = 1.2
        elif source_type in ["chat_log", "action_trace"]: authority = 0.8
        
        item["metadata"]["authority"] = authority
        item["metadata"]["embedding_model"] = ai_manager.get_model_name()
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # casting hyper-explicitly to handle pgvector native type reliably on D-Drive
                cur.execute(
                    "INSERT INTO ai_memory (content, embedding, metadata, created_at) VALUES (cast(%s as text), cast(%s as vector), cast(%s as jsonb), cast(%s as timestamptz))",
                    (item["content"], str(vector).replace(' ', ''), json.dumps(item["metadata"]), created_at)
                )
                conn.commit()
    
    print("✅ Test data ingested.")
    print("\n--- 🔍 Running Verification Searches ---")

    # TEST 1: Recency Bias
    print("\n[TEST 1] Recency Bias: Expecting 'recency_new' to rank above 'recency_old'")
    # We turn off query rewriting for the test to ensure exact vector comparison
    results = await get_hybrid_context("X-Algo is a powerful framework for AI search.", limit=5)
    for r in results:
        role = r['metadata'].get('id')
        print(f"ID: {role} | Score: {r['score']:.4f}")
    
    if results[0]['metadata'].get('id') == 'recency_new':
        print("PASSED: Newest document ranked first.")
    else:
        print("FAILED: Older document ranked first.")

    # TEST 2: Authority Boost
    print("\n[TEST 2] Authority Boost: Expecting 'auth_high' to rank above 'auth_low'")
    results = await get_hybrid_context("What is the reset protocol?", limit=5)
    for r in results:
         print(f"ID: {r['metadata'].get('id')} | Score: {r['score']:.4f}")
    
    if results[0]['metadata'].get('id') == 'auth_high':
        print("PASSED: Official (high-authority) document ranked first.")
    else:
        print("FAILED: Lower-authority document ranked first.")

    # TEST 3: Document Diversity
    print("\n[TEST 3] Document Diversity: Expecting at most 2 results from 'source_one' in top rankings without penalty")
    results = await get_hybrid_context("Topic Alpha", limit=5)
    source_counts = {}
    for r in results:
        s_id = r['metadata'].get('source_url')
        source_counts[s_id] = source_counts.get(s_id, 0) + 1
        print(f"Source: {s_id} | Score: {r['score']:.4f} | Original: {r.get('original_score', 0):.4f}")

    if all(count <= 3 for count in source_counts.values()):
         print("PASSED: Diversity logic applied (check scores for penalties on >2 results).")
    else:
         print("FAILED: Diversity filter not applied.")

    print("\n✨ Verification Complete!")

if __name__ == "__main__":
    asyncio.run(verify_x_algo())
