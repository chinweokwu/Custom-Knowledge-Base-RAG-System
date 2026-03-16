import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def ingest_text(content, topic):
    print(f"📥 Ingesting Text: {topic}...")
    payload = {
        "content": content,
        "metadata": {"topic": topic, "type": "manual_entry"}
    }
    response = requests.post(f"{BASE_URL}/ingest", json=payload)
    print(f"   Response: {response.json()}\n")
    return response.json()

def ingest_url(url):
    print(f"🌐 Ingesting URL: {url}...")
    payload = {
        "source": url,
        "metadata": {"source_url": url, "type": "web_scrape"}
    }
    response = requests.post(f"{BASE_URL}/ingest/file", json=payload)
    print(f"   Response: {response.json()}\n")
    return response.json()

def search_memory(query):
    print(f"🔍 Searching for: '{query}'...")
    response = requests.get(f"{BASE_URL}/search/stream", params={"query": query, "limit": 3}, stream=True)
    
    print("\n--- Top Results (RRF + Re-ranked) ---")
    for line in response.iter_lines():
        if line:
            result = json.loads(line)
            content_preview = result['content'][:100] + "..."
            score = result.get('score', 0)
            print(f"📄 Content: {content_preview}")
            print(f"   [Precision Score: {score:.4f}]")
            print("-" * 30)

def chat_mode():
    print("\n💬 Entering Chat Mode (Type 'exit' to quit)")
    while True:
        query = input("\nYOU: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        print("AI is thinking...", end="\r")
        response = requests.post(f"{BASE_URL}/chat", json={"message": query, "limit": 3})
        data = response.json()
        
        print("AI: " + data['answer'])
        if data['sources']:
            print(f"\n   (Sources used: {len(data['sources'])} memories)")

if __name__ == "__main__":
    print("🚀 AI Memory & Knowledge Base Demo\n")
    
    # Check if we should skip ingestion for quick testing
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        chat_mode()
        sys.exit()

    # 1. Ingest sample data
    ingest_text(
        "Python 3.12 introduced isolated subinterpreters and per-interpreter GIL.",
        "python_updates"
    )
    
    # 2. Ingest a website
    ingest_url("https://www.postgresql.org/about/")
    
    print("⏳ Waiting for processing...")
    time.sleep(3)
    
    # 3. Test Search
    search_memory("What's new in Python 3.12?")
    
    # 4. Start Interactive Chat
    chat_mode()
    
    print("\n✨ Demo Complete!")
