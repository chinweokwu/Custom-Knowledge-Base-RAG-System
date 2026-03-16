import os
import psycopg
from dotenv import load_dotenv
import json
from datetime import datetime, timezone

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def debug_insert():
    print(f"Connecting to: {DATABASE_URL}")
    content = "Debug test content"
    vector = [0.1] * 384
    metadata = {"test": "debug"}
    created_at = datetime.now(timezone.utc)
    
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                print("Testing insert query...")
                cur.execute("DELETE FROM ai_memory WHERE content = %s;", (content,))
                cur.execute(
                    "INSERT INTO ai_memory (content, embedding, metadata, created_at) VALUES (cast(%s as text), cast(%s as vector), cast(%s as jsonb), cast(%s as timestamptz))",
                    (content, str(vector).replace(' ', ''), json.dumps(metadata), created_at)
                )
                conn.commit()
                print("Success!")
    except Exception as e:
        print(f"❌ Insert FAILED: {e}")

if __name__ == "__main__":
    debug_insert()
