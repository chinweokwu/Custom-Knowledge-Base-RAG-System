import asyncio
import os
from dotenv import load_dotenv

# Ensure we're running from the right directory
os.environ["PYTHONPATH"] = os.getcwd()
load_dotenv()

from app.core.ai_manager import ai_manager

async def test_hyde():
    print("Initializing HyDE Test...")
    query = "What is the overarching goal of this project?"
    print(f"User Query: '{query}'")
    
    hyde_doc = await ai_manager.generate_hyde_document(query)
    
    print("\n--- HYDE DOCUMENT (FAKE ANSWER) ---")
    print(hyde_doc)
    print("-----------------------------------")
    
if __name__ == "__main__":
    asyncio.run(test_hyde())
