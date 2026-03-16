import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def verify_vector():
    print(f"Connecting to: {DATABASE_URL}")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                print("Checking EXTENSION vector...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                
                print("Testing vector cast...")
                cur.execute("SELECT '[1,2,3]'::vector;")
                result = cur.fetchone()
                print(f"Vector Test Result: {result}")
                
                print("✅ pgvector is successfully installed and active!")
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    verify_vector()
