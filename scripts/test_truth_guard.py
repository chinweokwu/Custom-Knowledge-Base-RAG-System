import requests
import time
import json
from app.core.graph_manager import graph_manager

API_URL = "http://localhost:8000"

def run_test():
    print("🚀 Starting Factual Conflict Test...")
    
    # 1. Ingest a document with a "Base Fact"
    # We use a unique ID 'TX-9000' to avoid existing data
    content = "The TX-9000 communication unit is configured to use Management Port 80 by default."
    metadata = {"source": "legacy_manual.pdf", "type": "manual"}
    
    print("📝 Step 1: Ingesting legacy document (Port 80)...")
    # We hit the upload endpoint (simulating a file upload or direct ingest)
    # Using the /ingest/file style or direct task
    response = requests.post(f"{API_URL}/upload", files={
        "file": ("tx9000.txt", content),
        "metadata_json": (None, json.dumps(metadata))
    })
    
    if response.status_code != 200:
        print(f"❌ Ingestion failed: {response.text}")
        return
        
    task_id = response.json().get("task_id")
    print(f"⏳ Task {task_id} started. Waiting for ingestion to finish...")
    
    # Poll for completion
    for _ in range(10):
        res = requests.get(f"{API_URL}/task/{task_id}")
        state = res.json().get("status")
        if state == "SUCCESS":
            break
        time.sleep(2)
    
    print("✅ Step 1 Complete: Document indexed.")

    # 2. Insert a CONFLICTING Fact into Neo4j (Ground Truth)
    print("🕸️ Step 2: Inserting CONFLICTING fact into Neo4j (Port 9999)...")
    graph_manager.add_relationship("TX-9000", "USES_MANAGEMENT_PORT", "9999", {"verification": "overridden_by_admin"})
    print("✅ Step 2 Complete: Graph Ground-Truth set to 9999.")

    # 3. Ask the AI the question
    print("❓ Step 3: Asking the AI about TX-9000 Port...")
    query = "What is the management port for the TX-9000 unit?"
    response = requests.post(f"{API_URL}/chat", json={"message": query})
    
    if response.status_code == 200:
        answer = response.json().get("answer")
        print("\n" + "="*50)
        print(f"🤖 AI RESPONSE:\n{answer}")
        print("="*50 + "\n")
        
        if "9999" in answer and "80" not in answer:
            print("✨ TEST SUCCESS: Truth Guard successfully prioritized the Graph (9999) over the Document (80)!")
        elif "9999" in answer and "80" in answer:
            print("⚠️ TEST PARTIAL: AI mentioned both, but should have corrected to 9999.")
        else:
            print("❌ TEST FAILED: AI still thinks it's Port 80.")
    else:
        print(f"❌ Chat request failed: {response.text}")

if __name__ == "__main__":
    run_test()
