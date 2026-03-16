import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from app.core.logger_config import get_logger

logger = get_logger("chroma_client")
load_dotenv()

# We will use a local folder for ChromaDB
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = "ai_memory"

# Initialize the ChromaDB client with persistent local storage
try:
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
        
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    logger.info(f"✅ Connected to ChromaDB at {CHROMA_DB_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to connect to ChromaDB: {e}")
    raise

# Get or create the collection
try:
    # Cosine similarity matches our Postgres pgvector setup perfectly
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"✅ ChromaDB collection '{COLLECTION_NAME}' ready.")
except Exception as e:
    logger.error(f"❌ Failed to initialize ChromaDB collection: {e}")
    raise

def get_chroma_collection():
    """Returns the main ai_memory collection."""
    return collection
