import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def check_typmod():
    print(f"Connecting to: {DATABASE_URL}")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = 'ai_memory'::regclass AND attname = 'embedding';
                """)
                res = cur.fetchone()
                print(f"atttypmod (Dimension Constraint): {res[0] if res else 'Column Not Found'}")
                if res and res[0] == -1:
                    print("✅ This is a generic vector (no dimension limit).")
                elif res:
                    print(f"❌ This is a FIXED-LENGTH vector (dimension={res[0]}).")
                
                # Check for existing data dimensions
                cur.execute("SELECT vector_dims(embedding), count(*) FROM ai_memory GROUP BY 1;")
                rows = cur.fetchall()
                print(f"Existing Data Dimensions: {rows}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_typmod()
