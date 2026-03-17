from pymilvus import MilvusClient
import os

try:
    print("Attempting to connect to Milvus Lite...")
    client = MilvusClient("test_connection.db")
    print("✅ Successfully connected to Milvus Lite.")
    print("Creating collection test...")
    if client.has_collection("test"):
        client.drop_collection("test")
    client.create_collection("test", dimension=128)
    print("✅ Successfully created collection.")
    client.drop_collection("test")
    print("✅ Cleanup complete.")
    os.remove("test_connection.db")
except Exception as e:
    print(f"❌ Connection failed: {e}")
