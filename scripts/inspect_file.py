import os
import sys
sys.path.append(os.getcwd())
from app.core.database import pool

def inspect_file_content(filename):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content, metadata 
                    FROM ai_memory 
                    WHERE metadata->>'filename' = %s 
                    LIMIT 20;
                """, (filename,))
                rows = cur.fetchall()
                print(f"FILE_INSPECTION_REPORT: {filename}")
                for i, row in enumerate(rows):
                    print(f"\n--- FRAGMENT {i+1} ---")
                    print(f"CONTENT: {row[0]}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    inspect_file_content("rnoc_llm_c_chat_log_export_20260309164536.xlsx")
