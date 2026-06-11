from pymilvus import MilvusClient
import os

APP_MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "ai_memory_v2"
VISUAL_COLLECTION_NAME = "ai_visual_memory"

def drop_collections():
    try:
        client = MilvusClient(uri=APP_MILVUS_URI)
        for c in [COLLECTION_NAME, VISUAL_COLLECTION_NAME]:
            if client.has_collection(c):
                print(f"Dropping collection {c}...")
                client.drop_collection(c)
                print(f"Dropped {c}.")
            else:
                print(f"Collection {c} does not exist.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    drop_collections()
