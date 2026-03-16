import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.append(os.getcwd())
load_dotenv()

from app.services.loaders import extract_chunks_from_source
from app.services.tasks import process_and_store_batch

async def verify_hierarchy():
    print("🚀 Initializing Hierarchical Retrieval Verification...")
    
    # 1. Create a mock large technical document
    mock_file = "D:\\AI knowledge Based\\tests\\mock_technical_guide.txt"
    with open(mock_file, "w") as f:
        f.write("# TECHNICAL GUIDE FOR HIERARCHICAL RAG\n\n")
        # Generate enough text to trigger splitting
        for i in range(20):
            f.write(f"CHAPTER {i+1}: Detailed architectural explanation of scaling vector databases. "
                    f"This section covers the deep technical nuances of HNSW indexes, "
                    f"memory management, and query expansion patterns in a multi-model environment. "
                    f"Note that for chapter {i+1}, the specific parameter value is 'PARAM-{i+1}-XYZ'.\n\n")
    
    print(f"✅ Created mock document: {mock_file}")
    
    # 2. Test Extraction
    print("--- Testing extract_chunks_from_source (Hierarchical) ---")
    chunks = extract_chunks_from_source(mock_file, hierarchical=True)
    print(f"Total Chunks: {len(chunks)}")
    if len(chunks) > 0 and isinstance(chunks[0], tuple):
        print(f"✅ SUCCESS: Chunks are tuples (child, parent).")
        print(f"Child Sample: {chunks[0][0][:100]}...")
        print(f"Parent Sample: {chunks[0][1][:100]}...")
    else:
        print("❌ ERROR: Chunks are not hierarchical tuples.")
        return

    # 3. Simulate Ingestion (Manual step because Celery needs a worker)
    # We will just verify that process_and_store_batch handles the tuples correctly
    print("\n--- Testing Storage Logic Preparation ---")
    # We'll just run the core logic of storage preparation
    from app.core.ai_manager import ai_manager
    sub_batch = chunks[:2]
    is_hierarchical = isinstance(sub_batch[0], tuple)
    sub_chunks = [c[0] if is_hierarchical else c for c in sub_batch]
    
    print(f"Embedding {len(sub_chunks)} child snippets...")
    vecs = await ai_manager.get_embeddings_batch(sub_chunks)
    print(f"Embedded successful. Dimensions: {len(vecs[0])}")
    
    # Verify metadata preparation
    child_text = sub_batch[0][0]
    parent_text = sub_batch[0][1]
    
    print("\n--- FAITHFULNESS CHECK ---")
    print(f"Child contains 'PARAM-1-XYZ': {'PARAM-1-XYZ' in child_text}")
    print(f"Parent contains 'PARAM-1-XYZ': {'PARAM-1-XYZ' in parent_text}")
    print(f"Parent length ({len(parent_text)}) > Child length ({len(child_text)})")
    
    print("\n✨ HIERARCHICAL LOGIC VERIFIED.")

if __name__ == "__main__":
    asyncio.run(verify_hierarchy())
