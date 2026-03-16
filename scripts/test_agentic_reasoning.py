import asyncio
import os
import sys
import requests
import time
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.append(os.getcwd())
load_dotenv()

from app.services.loaders import extract_chunks_from_source
from app.services.tasks import process_and_store_batch

async def test_agentic_loop():
    print("🚀 Initializing Agentic Reasoning Verification...")
    
    # 1. Create two fragmented technical notes
    note_1 = "D:\\AI knowledge Based\\tests\\agent_note_1.txt"
    note_2 = "D:\\AI knowledge Based\\tests\\agent_note_2.txt"
    
    with open(note_1, "w") as f:
        f.write("SYSTEM DESIGN: Project 'Nebula' is an advanced neural gateway that uses the 'Cortex-X' protocol for inter-module synchronization.")
        
    with open(note_2, "w") as f:
        f.write("COOPERATIVE PROTOCOLS: The 'Cortex-X' protocol (used by Nebula) is strictly limited to a maximum latency of 15ms per hop to ensure real-time stability.")

    print("✅ Created fragmented knowledge notes.")

    # 2. Ingest notes into the Vector DB
    print("📥 Ingesting notes...")
    for note in [note_1, note_2]:
        chunks = extract_chunks_from_source(note, hierarchical=True)
        # We call the task logic locally for simplicity in test
        process_and_store_batch.delay(chunks, {"source": note, "type": "technical_note"})
    
    print("⏳ Waiting for Celery ingestion (5 seconds)...")
    await asyncio.sleep(5)

    # 3. Call the Agentic Chat Endpoint
    url = "http://localhost:8000/chat"
    query = "What is the maximum latency allowed for Project Nebula's synchronization protocol?"
    
    print(f"📡 Sending Agentic Query: '{query}'")
    
    try:
        response = requests.post(url, json={"message": query, "limit": 3})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            thought_process = data.get("thought_process", [])
            
            print("\n--- AGENTIC LOGS (Thought Process) ---")
            for step in thought_process:
                print(f"[{step}]")
                
            print("\n--- FINAL ANSWER ---")
            print(answer)
            
            # 4. Validation
            has_reasoning = any("Initial search insufficient" in s or "Searching" in s for s in thought_process)
            has_facts = "15ms" in answer and "Cortex-X" in answer
            
            if has_reasoning and has_facts:
                print("\n✅ SUCCESS: Agentic Loop correctly identifying missing info and resolving multi-hop query.")
            elif has_facts:
                print("\n⚠️ PARTIAL SUCCESS: Fact found, but Agentic Loop might have been bypassed (Found in first search?).")
            else:
                print("\n❌ FAILED: Facts not found or reasoning failed.")
        else:
            print(f"❌ API ERROR: {response.text}")
            
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_agentic_loop())
