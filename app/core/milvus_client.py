import os
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
from app.core.logger_config import get_logger

logger = get_logger("milvus_client")
load_dotenv()

# We will use a local SQLite-like file for Milvus Lite
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "milvus_local.db")
COLLECTION_NAME = "ai_memory"
DIMENSION = 1920

# Initialize the Milvus Lite client
try:
    milvus_client = MilvusClient(MILVUS_DB_PATH)
    logger.info(f"✅ Connected to Milvus Lite at {MILVUS_DB_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Milvus Lite: {e}")
    raise

def init_milvus_collection():
    """
    Creates the necessary collection and schema in Milvus Lite if it doesn't exist.
    """
    if milvus_client.has_collection(collection_name=COLLECTION_NAME):
        logger.info(f"Milvus collection '{COLLECTION_NAME}' already exists.")
        return

    logger.info(f"Creating Milvus collection '{COLLECTION_NAME}' with dimension {DIMENSION}...")
    
    # 1. Create Schema
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True # This allows us to store arbitrary metadata (like JSONB in Postgres)
    )

    # 2. Add Fields
    # The primary key
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    # The actual vector embedding
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    # The text chunk itself (storing it directly alongside the vector)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 3. Create Collection
    # Index params (how it searches). FLAT is perfect for local 100% precision up to ~1M vectors.
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="FLAT"
    )

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"✅ Milvus collection '{COLLECTION_NAME}' initialized successfully.")

if __name__ == "__main__":
    init_milvus_collection()
