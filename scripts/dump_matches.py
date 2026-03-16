import os
import sys
sys.path.append(os.getcwd())
from app.core.database import pool

def dump_all_id_matches(user_id):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM ai_memory WHERE content ILIKE %s;", (f"%{user_id}%",))
                rows = cur.fetchall()
                print(f"DUMP_FOR_ID: {user_id}")
                print(f"COUNT: {len(rows)}")
                for i, row in enumerate(rows):
                    print(f"MATCH {i+1}: {repr(row[0][:500])}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    dump_all_id_matches("jwx1369347")
