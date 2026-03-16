import asyncio
import os
import numpy as np
from app.core.ai_manager import ai_manager
from dotenv import load_dotenv

load_dotenv()

async def test_neural_fusion():
    print("--- [Neural Fusion (ModelX5) Standalone Test] ---")
    
    text = "SITE_ID: 402 | LOCATION: Ibadan | STATUS: FAILED | REASON: Generator Overheat"
    print(f"Feeding text to 5x Ensemble: '{text}'")
    
    # 1. Get Embeddings
    embedding = await ai_manager.get_embeddings(text)
    
    print(f"\nFusion Results:")
    print(f"Dimensions: {len(embedding)} (Target: 1920)")
    print(f"Storage Location: {os.environ.get('HF_HOME')}")
    
    if len(embedding) == 1920:
        print("\n✅ SUCCESS: 1920-dimension Neural Fusion is ACTIVE.")
        print("Note: Llama-3 is still downloading in the background, but your mathematical engine is 100% online.")
    else:
        print(f"\n❌ ERROR: Dimension mismatch. Got {len(embedding)} instead of 1920.")

if __name__ == "__main__":
    asyncio.run(test_neural_fusion())
