import os
import sys
from dotenv import load_dotenv

# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import pool
from app.core.logger_config import get_logger

logger = get_logger("clear_db")

def clear_database():
    """
    Safely truncates the ai_memory and knowledge_gaps tables.
    Use this when transitioning to a new intelligence strategy (e.g., 5x Neural Fusion).
    """
    confirm = input("⚠️ WARNING: This will delete ALL stored AI memories and knowledge gaps. Port is irreversible. Type 'DELETE' to proceed: ")
    
    if confirm != "DELETE":
        print("Operation cancelled.")
        return

    logger.info("Cleaning up Vector Database for Neural Refresh...")
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Truncate tables to reset primary keys and clear data
                cur.execute("TRUNCATE TABLE ai_memory RESTART IDENTITY;")
                cur.execute("TRUNCATE TABLE knowledge_gaps RESTART IDENTITY;")
                conn.commit()
                logger.info("✅ Database CLEARED. Ready for High-IQ Re-ingestion.")
                print("\nSuccess! Your D-Drive is now fresh. Re-upload your files to use the 1920-dimension Neural Fusion ensemble.")
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    clear_database()
