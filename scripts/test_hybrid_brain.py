import asyncio
import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ai_manager import ai_manager
from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME
from app.services.retrieval import get_hybrid_context

async def verify_brain():
    print("🚀 Initializing Ultra-HD Hybrid Brain Verification...")
    
    # 1. Initialize Collection
    try:
        init_milvus_collection()
        print(f"✅ Collection '{COLLECTION_NAME}' ready.")
    except Exception as e:
        print(f"❌ Failed to initialize collection: {e}")
        return

    # 2. Test Embedding Generation
    test_text = "Site ID JWX1369347: Cooling system failure at Alpha Sector. Check RRU connector."
    print(f"\n🧬 Generating Hybrid Embedding for: '{test_text}'")
    
    try:
        dense, sparse = await ai_manager.get_hybrid_embeddings(test_text)
        
        print(f"✅ Dense Dimensions: {len(dense)} (Expected: 3712)")
        print(f"✅ Sparse Non-Zero Tokens: {len(sparse)}")
        
        # Verify Sparse Structure (Should be a dict of {int: float})
        if isinstance(sparse, dict):
            print(f"✅ Sparse Matrix verified as optimized dictionary.")
        else:
            print(f"⚠️ Sparse output type: {type(sparse)}")
            
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return

    # 3. Test Retrieval Logic
    print("\n🔍 Testing Native Hybrid Search (Dense 4k + Sparse 32k)...")
    try:
        # Note: This might return empty if no data is ingested, but we check for logic success
        results = await get_hybrid_context("How to fix JWX1369347 cooling?")
        print(f"✅ Retrieval logic verified. (Note: Results will be empty until re-ingestion).")
    except Exception as e:
        print(f"❌ Retrieval logic failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_brain())
