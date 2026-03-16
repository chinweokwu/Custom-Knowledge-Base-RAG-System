import asyncio
from app.core.database import pool
from app.core.logger_config import get_logger

logger = get_logger("db_monitor")

async def check_stats():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Count total rows
                cur.execute("SELECT COUNT(*) FROM ai_memory;")
                total = cur.fetchone()[0]
                
                # Count by dimensions (1920 is Neural Fusion, 1536 is OpenAI)
                cur.execute("""
                    SELECT 
                        metadata->>'dimensions' as dims, 
                        metadata->>'embedding_model' as model,
                        COUNT(*) 
                    FROM ai_memory 
                    GROUP BY dims, model;
                """)
                rows = cur.fetchall()
                
                print("\n--- 🧠 Neural Fusion Ingestion Stats ---")
                print(f"Total Fragments in DB: {total}")
                print("-" * 40)
                for dims, model, count in rows:
                    status = "✅ ACTIVE" if dims == "1920" else "🕒 LEGACY/OPENAI"
                    print(f"Model: {model:<30} | Dims: {dims:<5} | Count: {count:<6} | {status}")
                print("-" * 40)
                print("\nTip: Look for 'Task SUCCESS' in your celery logs for final completion.")

    except Exception as e:
        print(f"Error checking stats: {e}")

if __name__ == "__main__":
    asyncio.run(check_stats())
