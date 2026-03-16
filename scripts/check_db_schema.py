import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def check_schema():
    print(f"Connecting to: {DATABASE_URL}")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT column_name, udt_name FROM information_schema.columns WHERE table_name = 'ai_memory' AND column_name = 'embedding';")
                res = cur.fetchone()
                print(f"Column: {res}")
                
                # Check for vector index
                cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'ai_memory' AND indexname = 'idx_memory_hnsw';")
                idx = cur.fetchone()
                print(f"Index: {idx}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
