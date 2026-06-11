import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "Enterprise-Neural-Fusion-RAG"
    VERSION: str = "3.0.0"
    
    # Infrastructure
    MILVUS_URI: str = os.getenv("MILVUS_URI") or os.getenv("APP_MILVUS_URI") or "milvus_lite.db"
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # AI Models
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    LLM_INGEST_MODEL: str = os.getenv("LLM_INGEST_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_DEVICE: str = "cpu" # Switch to 'cuda' if GPU available
    
    # Storage Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MEDIA_DIR: str = os.path.join(BASE_DIR, "media")
    MODELS_CACHE_DIR: str = os.path.join(BASE_DIR, "models_cache")
    
    # Vector DB Config
    COLLECTION_NAME: str = "ai_hybrid_memory_32k"
    VISUAL_COLLECTION_NAME: str = "ai_visual_memory_v2"
    DENSE_DIMENSION: int = 3712
    
    # Model Offloading (Inference Server)
    OFFLOAD_EMBEDDINGS: bool = os.getenv("OFFLOAD_EMBEDDINGS", "False").lower() in ("true", "1", "yes")
    EMBEDDING_SERVER_URL: str = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8000")
    # Retrieval Config
    RRF_K: int = 60
    THRESHOLD: float = 0.1 # Lowered to allow more candidates for re-ranking
    
    # Intelligence Weights (X-Algo)
    AUTHORITY_MANUAL: float = 20.0
    AUTHORITY_CHAT: float = 0.8
    BOOST_SYNTHETIC: float = 2.5
    BOOST_ID_MATCH: float = 5.0
    BOOST_KEYWORD: float = 15.0
    
    # Temporal Decay (Weight lost per year)
    TEMPORAL_DECAY_RATE: float = 0.1 
    
    # Truth Guard Config
    ENABLE_GRAPH_TRUTH_GUARD: bool = True
    TRUTH_GUARD_THRESHOLD: float = 0.7 
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
