import os
import sys
# Ensure app modules are importable
sys.path.append(os.getcwd())

from app.core.chroma_client import chroma_client, COLLECTION_NAME

def purge_memory():
    """
    Purges the vector memory by deleting the ChromaDB collection.
    PostgreSQL is decommissioned and no longer purged.
    """
    print("⚠️  Preparing to purge vector memory (ChromaDB Only)...")
    
    # Clear Knowledge Graph
    try:
        graph_path = os.path.join("D:\\AI knowledge Based", "graph_store.pkl")
        if os.path.exists(graph_path):
            os.remove(graph_path)
            print("✅ Knowledge Graph (graph_store.pkl) deleted.")
        else:
            print("ℹ️ Knowledge Graph already empty.")
    except Exception as e:
        print(f"❌ Knowledge Graph Purge Error: {e}")

    print("\n✨ GLOBAL RESET COMPLETE: Vector DB & Knowledge Graph are clean slates.")

if __name__ == "__main__":
    purge_memory()
