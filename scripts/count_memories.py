import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from app.core.database import pool

def count_memories():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM ai_memory;")
                count = cur.fetchone()[0]
                print(f"TOTAL_MEMORIES: {count}")
                
                cur.execute("SELECT max(id) FROM ai_memory;")
                max_id = cur.fetchone()[0]
                print(f"MAX_ID: {max_id}")
                
                cur.execute("SELECT created_at FROM ai_memory ORDER BY created_at DESC LIMIT 1;")
                latest = cur.fetchone()
                if latest:
                    print(f"LATEST_INGESTION: {latest[0]}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    count_memories()
