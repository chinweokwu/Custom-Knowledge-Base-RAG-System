import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

# Extract connection details but connect to 'postgres' DB first
conn_str = "postgresql://postgres:paul1234@localhost:5432/postgres"

def create_db():
    print("Attempting to create 'ai_memory' database...")
    try:
        # Connect to default postgres database
        conn = psycopg.connect(conn_str, autocommit=True)
        with conn.cursor() as cur:
            # Check if ai_memory exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ai_memory'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("CREATE DATABASE ai_memory")
                print("✅ Database 'ai_memory' created successfully.")
            else:
                print("ℹ️  Database 'ai_memory' already exists.")
        conn.close()
    except Exception as e:
        print(f"❌ Failed to create database: {e}")

if __name__ == "__main__":
    create_db()
