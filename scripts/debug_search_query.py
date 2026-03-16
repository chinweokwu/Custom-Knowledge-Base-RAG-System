import os
import psycopg
from dotenv import load_dotenv
import json

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def debug_query():
    print(f"Connecting to: {DATABASE_URL}")
    model_name = "local_minilm"
    query_vector = [0.1] * 384
    limit = 5
    
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                print("Testing search query...")
                cur.execute("""
                    SELECT id, content, metadata, created_at
                    FROM ai_memory
                    WHERE (metadata->>'embedding_model' = cast(%s as text) OR metadata->>'embedding_model' IS NULL)
                      AND vector_dims(embedding) = cast(%s as integer)
                    ORDER BY embedding <=> cast(%s as vector)
                    LIMIT cast(%s as integer);
                """, (model_name, len(query_vector), str(query_vector).replace(' ', ''), limit * 3))
                rows = cur.fetchall()
                print(f"Success! Rows: {len(rows)}")
    except Exception as e:
        print(f"❌ Query FAILED: {e}")

if __name__ == "__main__":
    debug_query()
