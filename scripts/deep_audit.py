import os
import sys
sys.path.append(os.getcwd())
from app.core.database import pool

def deep_audit_id(user_id):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Search for any record containing the user_id
                cur.execute("SELECT content, metadata FROM ai_memory WHERE content ILIKE %s LIMIT 10;", (f"%{user_id}%",))
                rows = cur.fetchall()
                print(f"AUDIT_REPORT_FOR_ID: {user_id}")
                print(f"TOTAL_MATCHES: {len(rows)}")
                for i, row in enumerate(rows):
                    print(f"\n--- MATCH {i+1} ---")
                    print(f"METADATA: {row[1]}")
                    print(f"FULL_CONTENT: {row[0]}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    deep_audit_id("jwx1369347")
