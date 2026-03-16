import requests
import json

def diag_api():
    url = "http://localhost:8000/chat"
    msg = "What is the relation between Nebula and its protocol?"
    print(f"Querying: {msg}")
    
    resp = requests.post(url, json={"message": msg, "limit": 5})
    if resp.status_code == 200:
        data = resp.json()
        print("\n--- RESPONSE JSON (PREVIEW) ---")
        print(f"Answer: {data.get('answer')}")
        print("\n--- SOURCES ---")
        for i, s in enumerate(data.get("sources", [])):
            print(f"Source {i}: {s.get('content')}")
        
        print("\n--- THOUGHT PROCESS ---")
        for t in data.get("thought_process", []):
            print(f"- {t}")
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    diag_api()
