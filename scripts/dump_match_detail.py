import os
import sys
sys.path.append(os.getcwd())
from app.core.database import pool

def dump_match_107(user_id):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM ai_memory WHERE content ILIKE %s OFFSET 106 LIMIT 1;", (f"%{user_id}%",))
                row = cur.fetchone()
                if row:
                    print(f"FULL_CONTENT_FOR_{user_id}_MATCH_107:")
                    print("-" * 50)
                    print(row[0])
                    print("-" * 50)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    dump_match_107("jwx1369347")
