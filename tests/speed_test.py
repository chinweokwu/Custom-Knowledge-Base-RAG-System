import asyncio
import time
from app.services.retrieval import get_hybrid_context
from app.core.ai_manager import ai_manager
from dotenv import load_dotenv

load_dotenv()

async def run_speed_test():
    print("🚀 Starting RAG Latency Speed Test...")
    print(f"Model Stack: {ai_manager.get_model_name()}")
    
    # Test query
    query = "What is the reset protocol for technical identifiers?"
    
    print(f"\n--- Testing Query: '{query}' ---")
    
    # Run once to warm up models (especially if it's the first run)
    print("Warming up models...")
    start_warm = time.time()
    await get_hybrid_context(query, limit=5)
    print(f"Warm-up complete in {time.time() - start_warm:.2f}s")
    
    # Measurement Run
    print("\nMeasuring Concurrent Execution Speed...")
    latencies = []
    for i in range(3):
        start = time.time()
        results = await get_hybrid_context(query, limit=5)
        end = time.time()
        latencies.append(end - start)
        print(f"Run {i+1}: {end - start:.4f}s | Results found: {len(results)}")
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n✅ Average Latency: {avg_latency:.4f}s")
    
    if avg_latency < 2.0:
        print("⚡ PERFORMANCE: EXCELLENT (Sub-2s for 5x Ensemble + LLM agents)")
    elif avg_latency < 4.0:
        print("🚗 PERFORMANCE: GOOD (Acceptable for agentic retrieval)")
    else:
        print("🐢 PERFORMANCE: NEEDS ATTENTION (Check hardware/GPU availability)")

if __name__ == "__main__":
    asyncio.run(run_speed_test())
