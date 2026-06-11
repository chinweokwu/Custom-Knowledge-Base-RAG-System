from pymilvus import MilvusClient
import os

APP_MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "ai_memory_v2"

def get_stats():
    try:
        client = MilvusClient(uri=APP_MILVUS_URI)
        if client.has_collection(COLLECTION_NAME):
            stats = client.get_collection_stats(COLLECTION_NAME)
            print(f"Stats for {COLLECTION_NAME}: {stats}")
            
            # Check partitions
            partitions = client.list_partitions(COLLECTION_NAME)
            print(f"Partitions: {partitions}")
            
            # Check index
            indexes = client.list_indexes(COLLECTION_NAME)
            print(f"Indexes: {indexes}")
        else:
            print("Collection not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_stats()
