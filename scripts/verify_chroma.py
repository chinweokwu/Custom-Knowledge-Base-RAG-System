import asyncio
import os
import sys

# Ensure project root is in sys.path for portable environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.main import get_hybrid_context, process_and_store_batch

async def main():
    print("🚀 Running ChromaDB Vector Migration Test...")
    
    # 1. Insert Mock Data via Celery function directly (bypass redis for testing)
    mock_chunks = [
        "The server JWX1369 is experiencing high latency on the primary database nodes.",
        "To resolve the JWX1369 connectivity issue, restart the main redis cache pool.",
        "ChromaDB is a modern, fast vector database built for local RAG applications."
    ]
    mock_meta = {
        "source": "test_script.py",
        "type": "official_doc",
        "author": "System"
    }
    
    print("\n--- Phase 1: Ingestion ---")
    print("Sending 3 chunks to process_and_store_batch...")
    # Call the synchronous part of the celery task
    try:
        result = process_and_store_batch(mock_chunks, mock_meta)
        print(f"✅ Ingestion Result: {result}")
    except Exception as e:
        print(f"❌ Ingestion Failed: {e}")
        return
        
    # 2. Query Data using standard Hybrid Search
    print("\n--- Phase 2: Hybrid Retrieval ---")
    query = "How do I fix the server JWX1369 latency issue?"
    print(f"Querying: '{query}'")
    
    results = await get_hybrid_context(query, limit=5)
    
    print(f"\n✅ Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\nResult [{i+1}] Score: {res['score']:.4f}")
        print(f"Content: {res['content']}")
        print(f"Metadata: {res['metadata']}")
        
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    asyncio.run(main())
