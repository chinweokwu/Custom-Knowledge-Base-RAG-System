import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set unique URI for verification to avoid collision
# Ensure MILVUS_URI is UNSET to prevent pymilvus from attempting a remote gRPC connection
if "MILVUS_URI" in os.environ:
    del os.environ["MILVUS_URI"]

# Using APP_MILVUS_URI to avoid conflict with pymilvus internal env var handling
os.environ["APP_MILVUS_URI"] = "test_milvus_lite.db"

from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME
from app.core.ai_manager import ai_manager

async def test_milvus_flow():
    print("🚀 Starting Milvus Lite Verification Test...")
    
    # 1. Initialize
    init_milvus_collection()
    
    # 2. Generate a test embedding
    test_text = "The quick brown fox jumps over the lazy dog. Technical code: SW-999888."
    embedding = await ai_manager.get_embeddings(test_text)
    
    print(f"✅ Generated embedding (Dim: {len(embedding)})")
    
    # 3. Insert data
    test_id = str(uuid.uuid4())
    data = [{
        "embedding": embedding,
        "content": test_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": "test_verification",
        "synthetic_questions": "What does the fox do? | What is the technical code?"
    }]
    
    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    print("✅ Inserted test record into Milvus Lite.")
    
    # 4. Search
    print("🔍 Performing vector search...")
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=1,
        output_fields=["content", "type"]
    )
    
    if search_res and len(search_res[0]) > 0:
        hit = search_res[0][0]
        print(f"✅ Search successful! Found content: {hit['entity'].get('content')[:50]}...")
        print(f"✅ Metadata 'type': {hit['entity'].get('type')}")
    else:
        print("❌ Search failed to find the inserted record.")
        return False

    # 5. Query
    print("🔍 Performing metadata query...")
    query_res = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="type == 'test_verification'",
        limit=1
    )
    
    if query_res:
        print(f"✅ Query successful! Found {len(query_res)} records.")
    else:
        print("❌ Query failed to find record by metadata.")
        return False

    print("\n✨ Milvus Lite Migration Verified Successfully! ✨")
    return True

if __name__ == "__main__":
    asyncio.run(test_milvus_flow())
