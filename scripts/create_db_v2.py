import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

def create_database():
    # Use the password provided by the user
    conn_info = "postgresql://postgres:paul1234@localhost:5432/postgres"
    print(f"Connecting to {conn_info}...")
    
    try:
        # Connect with autocommit to create a database
        conn = psycopg.connect(conn_info, autocommit=True)
        with conn.cursor() as cur:
            # Check if ai_memory exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname='ai_memory'")
            exists = cur.fetchone()
            if not exists:
                print("Creating database 'ai_memory'...")
                cur.execute("CREATE DATABASE ai_memory")
            else:
                print("Database 'ai_memory' already exists.")
        conn.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if create_database():
        print("Done.")
    else:
        print("Failed.")
