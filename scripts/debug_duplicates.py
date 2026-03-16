import asyncio
from app.core.database import pool
from app.core.logger_config import get_logger

logger = get_logger("debug_duplicates")

async def check_duplicates():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                print("\n--- 🕵️ Searching for Redundant Knowledge Fragments ---")
                
                # Find identical content across different embeddings
                cur.execute("""
                    SELECT content, COUNT(*), array_agg(DISTINCT metadata->>'embedding_model') 
                    FROM ai_memory 
                    GROUP BY content 
                    HAVING COUNT(*) > 1 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10;
                """)
                rows = cur.fetchall()
                
                if not rows:
                    print("✅ No duplicate content detected. The core is clean.")
                    return

                for content, count, models in rows:
                    print(f"\n[DUPLICATE FOUND] x{count} copies")
                    print(f"Models involved: {models}")
                    print(f"Snippet: {content[:150].strip()}...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_duplicates())
