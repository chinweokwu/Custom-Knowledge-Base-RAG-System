import os
import psycopg
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def sanity_check():
    print("🔍 Starting Database Sanity Check...")
    print(f"Connecting to: {DATABASE_URL}")
    
    try:
        # 1. Test Connection
        with psycopg.connect(DATABASE_URL) as conn:
            print("✅ Successfully connected to PostgreSQL!")
            
            with conn.cursor() as cur:
                # 2. Check for pgvector extension
                cur.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
                extension = cur.fetchone()
                
                if extension:
                    installed_version = extension[2] # version is the 3rd column
                    if installed_version:
                        print(f"✅ pgvector extension is INSTALLED (Version: {installed_version})")
                    else:
                        print("⚠️  pgvector is AVAILABLE but NOT installed in this database.")
                        print("👉 Run: CREATE EXTENSION vector;")
                else:
                    print("❌ pgvector extension NOT FOUND in your PostgreSQL installation.")
                    print("👉 You need to install the pgvector binary for Windows.")
                
                # 3. Check for specific table
                cur.execute("SELECT EXIsTS (SELECT FROM information_schema.tables WHERE table_name = 'ai_memory');")
                table_exists = cur.fetchone()[0]
                if table_exists:
                    print("✅ Table 'ai_memory' exists.")
                else:
                    print("ℹ️  Table 'ai_memory' does not exist yet. It will be created when you run database.py.")

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("\nPossible fixes:")
        print("1. Ensure PostgreSQL is running.")
        print("2. Check if the database name in .env actually exists in pgAdmin.")
        print("3. Verify your username and password in .env.")

if __name__ == "__main__":
    sanity_check()
