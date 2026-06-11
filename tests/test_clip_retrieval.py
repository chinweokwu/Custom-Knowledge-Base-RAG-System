import asyncio
import os
import sys
from PIL import Image
import io

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ai_manager import ai_manager
from app.api.main import get_hybrid_context
from app.core.milvus_client import milvus_client, VISUAL_COLLECTION_NAME, init_milvus_collection

async def test_clip_flow():
    print("🚀 Starting Multi-Modal CLIP Verification...")
    
    # 1. Initialize
    init_milvus_collection()
    
    # 2. Create a dummy image (Blue Square)
    img = Image.new('RGB', (100, 100), color = 'blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    
    print("--- 🛠️ Ingesting Visual Test Data ---")
    # Generate CLIP embedding
    visual_vector = await ai_manager.get_clip_embedding(img)
    
    # Store in Milvus
    test_data = {
        "embedding": visual_vector,
        "content": "[TEST_VISUAL] A blue square representing a network terminal.",
        "media_url": "/media/test_blue_square.png",
        "is_visual": True,
        "meaning_type": "test_image",
        "created_at": "2026-04-08T12:00:00Z"
    }
    
    milvus_client.insert(
        collection_name=VISUAL_COLLECTION_NAME,
        data=[test_data]
    )
    print("✅ Visual test data ingested into Milvus.")

    print("\n--- 🔍 Running Multi-Modal Search ---")
    
    # Search for something visually similar (text query)
    query = "Show me the blue square terminal"
    print(f"Query: '{query}'")
    
    results = await get_hybrid_context(query, limit=5)
    
    found = False
    for r in results:
        is_visual = r.get("metadata", {}).get("is_visual", False)
        print(f"Result: {r['content'][:60]}... | Score: {r['score']:.4f} | Visual: {is_visual}")
        if is_visual and "blue square" in r['content']:
            found = True
            
    if found:
        print("\n🏆 PASSED: Text query successfully retrieved the visual fragment via CLIP!")
    else:
        print("\n❌ FAILED: Visual fragment not found in results.")

if __name__ == "__main__":
    asyncio.run(test_clip_flow())
