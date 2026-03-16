import os
import sys
sys.path.append(os.getcwd())
from app.core.database import pool

def check_dims():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT vector_dims(embedding), count(*) FROM ai_memory GROUP BY 1;")
                rows = cur.fetchall()
                print("EMBEDDING_DIMENSIONS_REPORT:")
                for dims, count in rows:
                    print(f"Dims: {dims} | Count: {count}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    check_dims()
