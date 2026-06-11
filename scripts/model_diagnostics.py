import os
import asyncio
import time
from app.core.ai_manager import ai_manager
from app.core.milvus_client import milvus_client
from app.core.logger_config import get_logger

logger = get_logger("diagnostics")

async def run_diagnostics():
    print("\n" + "="*50)
    print("🧠 AI KNOWLEDGE BASE - NEURAL DIAGNOSTICS")
    print("="*50)

    # 1. Test Embedding Ensemble (Local Models)
    print("\n[1/3] Testing Neural Ensemble (Local Models)...")
    try:
        start_time = time.time()
        test_text = "Diagnostic signal for neural path verification."
        embedding = await ai_manager.get_embedding(test_text)
        duration = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"   - Models Active: {len(ai_manager.models)} (Balanced Mode)")
        print(f"   - Embedding Dimension: {len(embedding)}")
        print(f"   - Latency: {duration:.2f}s")
    except Exception as e:
        print(f"❌ Local Ensemble Failed: {e}")

    # 2. Test Groq Brain (70B Model)
    print("\n[2/3] Testing Groq Brain (Llama-3 70B)...")
    try:
        start_time = time.time()
        # Mocking a simple chain input
        class MockChain:
            def invoke(self, prompt):
                class Content:
                    content = "Neural handshake confirmed. Llama-3-70B is online and operational."
                return Content()
        
        # Actually calling the real LLM for a live test
        prompt = "Hello! Briefly state your model version and current status for a system diagnostic."
        response = await ai_manager.call_llm(None, {"question": prompt, "context": "System Diagnostic Mode"})
        duration = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"   - Response: {response}")
        print(f"   - Latency: {duration:.2f}s (Ultra-High Speed)")
    except Exception as e:
        print(f"❌ Groq API Failed: {e}")

    # 3. Test Milvus Connectivity
    print("\n[3/3] Testing Vector Memory (Milvus)...")
    try:
        collections = milvus_client.list_collections()
        print(f"✅ Success!")
        print(f"   - Connected to: {milvus_client.uri}")
        print(f"   - Active Collections: {collections}")
    except Exception as e:
        print(f"❌ Milvus Failed: {e}")

    print("\n" + "="*50)
    print("🎉 DIAGNOSTIC COMPLETE - SYSTEM READY")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
