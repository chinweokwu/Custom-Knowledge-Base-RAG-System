from pymilvus import MilvusClient, DataType
from app.core.logger_config import get_logger
from app.core.config import settings
import threading
import os

logger = get_logger("milvus_client")

# Global lock to prevent concurrent load operations
_load_lock = threading.Lock()
_loaded_collections = set()

APP_MILVUS_URI = settings.MILVUS_URI
COLLECTION_NAME = settings.COLLECTION_NAME
VISUAL_COLLECTION_NAME = settings.VISUAL_COLLECTION_NAME
DIMENSION = settings.DENSE_DIMENSION
VISUAL_DIMENSION = 512 # CLIP-ViT-B-32 dimension

class LazyMilvusClient:
    def __init__(self):
        self._client = None

    def _init_client(self):
        if self._client is None:
            try:
                self._client = MilvusClient(uri=APP_MILVUS_URI)
                logger.info(f"✅ Connected to Milvus at {APP_MILVUS_URI}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Milvus: {e}")
                raise
        return self._client

    def __getattr__(self, name):
        client = self._init_client()
        return getattr(client, name)

# Lazy instance of the Milvus client
milvus_client = LazyMilvusClient()

def init_milvus_collection():
    """
    Creates and loads the necessary collections.
    """
    with _load_lock:
        for c_name, dim in [(COLLECTION_NAME, DIMENSION), (VISUAL_COLLECTION_NAME, VISUAL_DIMENSION)]:
            if milvus_client.has_collection(collection_name=c_name):
                if c_name not in _loaded_collections:
                    logger.info(f"Checking load state for {c_name}...")
                    for attempt in range(1, 4): # Wait only 3 seconds for health checks/startup
                        state = milvus_client.get_load_state(collection_name=c_name)
                        state_val = str(state.get('state') if isinstance(state, dict) else getattr(state, 'name', state))
                        
                        if "loaded" in state_val.lower():
                            _loaded_collections.add(c_name)
                            logger.info(f"✅ {c_name} is LOADED.")
                            break
                        
                        # Track progress
                        progress = state.get('progress', 0) if isinstance(state, dict) else 0

                        # If not loading and not loaded, trigger a load
                        if "loading" not in state_val.lower() and "loaded" not in state_val.lower():
                            try: 
                                logger.info(f"Issuing load command for {c_name}...")
                                milvus_client.load_collection(collection_name=c_name)
                            except Exception as e:
                                logger.error(f"Load command failed for {c_name}: {e}")
                        
                        import time
                        time.sleep(1)
                    else:
                        # Fetch final progress for the warning
                        state = milvus_client.get_load_state(collection_name=c_name)
                        progress = state.get('progress', 0) if isinstance(state, dict) else 0
                        logger.warning(f"⚠️ {c_name} failed to load after 30s. Progress: {progress}%. Continuing anyway.")
            else:
                logger.info(f"Creating hybrid collection '{c_name}'...")
                schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
                
                # Add Sparse Vector Field (Lexical Expansion)
                if c_name == COLLECTION_NAME:
                    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

                index_params = MilvusClient.prepare_index_params()
                
                # Dense Index (HNSW for speed)
                index_params.add_index(
                    field_name="embedding", 
                    metric_type="COSINE", 
                    index_type="HNSW",
                    params={"M": 16, "efConstruction": 200}
                )
                
                # Sparse Index (Inverted Index for keywords)
                if c_name == COLLECTION_NAME:
                    index_params.add_index(
                        field_name="sparse_vector",
                        metric_type="IP",
                        index_type="SPARSE_INVERTED_INDEX",
                        params={"drop_ratio_build": 0.2}
                    )

                milvus_client.create_collection(collection_name=c_name, schema=schema, index_params=index_params)
                milvus_client.load_collection(collection_name=c_name)
                _loaded_collections.add(c_name)
                logger.info(f"✅ Hybrid Collection '{c_name}' initialized.")

if __name__ == "__main__":
    init_milvus_collection()
