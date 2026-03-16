import requests
import json

def test_api_search():
    url = "http://localhost:8000/search"
    query = "i want summary of chat from jwx1369347"
    params = {"query": query, "limit": 5}
    
    try:
        print(f"TESTING_URL: {url}")
        print(f"QUERY: {query}")
        response = requests.get(url, params=params)
        print(f"STATUS_CODE: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"RESULTS_FOUND: {len(results)}")
            for i, res in enumerate(results):
                print(f"Result {i+1} | Score: {res.get('score')} | Content: {res.get('content')[:100]}...")
            
            if not results:
                print("DEBUG: No results returned. System likely reported KNOWLEDGE GAP.")
        else:
            print(f"ERROR: {response.text}")
            
    except Exception as e:
        print(f"CONNECTION_ERROR: {e}")

if __name__ == "__main__":
    test_api_search()
