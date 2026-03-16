import asyncio
import os
import requests
import time
from dotenv import load_dotenv

# Ensure project root is in sys.path
import sys
sys.path.append(os.getcwd())
load_dotenv()

def test_graph_expansion():
    print("Starting GraphRAG Expansion Verification...")
    
    # Reset Graph for clean test (optional, but good for isolated verification)
    from app.core.graph_manager import graph_manager
    graph_manager.graph.clear()
    graph_manager.save_graph()
    
    # 1. Simulate Ingestion of related facts
    # Fact 1: Nebula uses Cortex-X
    # Fact 2: Cortex-X has 15ms latency
    print("📥 Directly populating Knowledge Graph with test relations...")
    graph_manager.add_relationship("Project Nebula", "uses", "Cortex-X Protocol")
    graph_manager.add_relationship("Cortex-X Protocol", "has limit of", "15ms Latency")
    
    print("✅ Relationships Added to Graph.")

    # 2. Call the Chat Endpoint with a query about 'Nebula'
    url = "http://localhost:8000/chat"
    query = "What are the technical specs for Project Nebula's protocol?"
    
    print(f"📡 Sending Graph-Aware Query: '{query}'")
    
    try:
        response = requests.post(url, json={"message": query, "limit": 3})
        if response.status_code == 200:
            data = response.json()
            sources = data.get("sources", [])
            
            print("\n--- RETRIEVED SOURCES (Looking for Graph Expansion) ---")
            found_graph_fact = False
            for s in sources:
                content = s.get("content", "")
                print(f"- {content[:100]}...")
                if "[Graph Fact]" in content:
                    found_graph_fact = True
            
            if found_graph_fact:
                print("\n✅ SUCCESS: Knowledge Graph expansion correctly injected into retrieval context.")
            else:
                print("\n❌ FAILED: Graph expansion facts not found in context.")
                
            print("\n--- AI SYNTHESIS ---")
            print(data.get("answer"))
        else:
            print(f"❌ API ERROR: {response.text}")
            
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    test_graph_expansion()
