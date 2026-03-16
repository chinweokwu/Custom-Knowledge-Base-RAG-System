import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def migrate_to_vector():
    print(f"Connecting to: {DATABASE_URL}")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                print("Checking for column type...")
                cur.execute("SELECT udt_name FROM information_schema.columns WHERE table_name = 'ai_memory' AND column_name = 'embedding';")
                res = cur.fetchone()
                if res and res[0] == '_float8':
                    print("Converting float8[] to vector...")
                    # We cast float8[] to vector. Note: This works if pgvector is installed.
                    cur.execute("ALTER TABLE ai_memory ALTER COLUMN embedding TYPE vector USING embedding::vector;")
                    print("Conversion SUCCESS.")
                else:
                    print(f"Column is already {res[0] if res else 'None'}.")
                
                # We won't add HNSW yet because of mixed dimensions (1536 and 384).
                # But native vector search is still faster than PL/pgSQL.
                conn.commit()
                print("✅ Migration to native vector complete.")
    except Exception as e:
        print(f"❌ Migration FAILED: {e}")

if __name__ == "__main__":
    migrate_to_vector()
