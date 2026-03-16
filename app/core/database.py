import os
from dotenv import load_dotenv
from app.core.logger_config import get_logger

# Initialize Logger
logger = get_logger("database")

# Load environment variables
load_dotenv()

# PostgreSQL logic is DECOMMISSIONED in favor of ChromaDB (Zero-Footprint Strategy)
# We keep the 'pool' variable as None to prevent immediate import errors, 
# but all functions are now stubs.
pool = None
HAS_PGVECTOR = False

def check_vector_capability():
    """DEPRECATED: We use ChromaDB now."""
    return False

def init_db():
    """
    Initializes the database. 
    POSTGRESQL DECOMMISSIONED: This system now uses ChromaDB for all vector operations.
    """
    logger.info("Database initialization: PostgreSQL is decommissioned. System is using ChromaDB.")
    return

if __name__ == "__main__":
    init_db()
