import json
import os
from typing import List, Dict, Any
from app.core.logger_config import get_logger

logger = get_logger("session_memory")

class SessionMemoryManager:
    def __init__(self):
        self.local_store: Dict[str, List[Dict[str, str]]] = {}
        self.redis_client = None
        self.use_redis = False
        
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(redis_url, socket_timeout=2, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ SessionMemoryManager using Redis for short-term session storage.")
        except Exception as e:
            logger.warning(f"⚠️ Redis offline or connection failed ({e}). Falling back to in-memory session store.")

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if not session_id:
            return []
        
        if self.use_redis:
            try:
                key = f"session_history:{session_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Error reading session from Redis: {e}")
        
        return self.local_store.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str, max_turns: int = 10):
        if not session_id:
            return
        
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        
        # Limit history to keep it fast and fit within context constraints (max_turns * 2 messages)
        if len(history) > max_turns * 2:
            evicted_count = len(history) - (max_turns * 2)
            evicted = history[:evicted_count]
            history = history[evicted_count:]
            
            # Asynchronously offload summarization & storage of evicted turns to Celery worker
            try:
                from app.services.tasks import summarize_and_store_evicted_turns
                summarize_and_store_evicted_turns.delay(session_id, evicted)
                logger.info(f"Queued background summarization for {len(evicted)} evicted messages (Session: {session_id}).")
            except Exception as e:
                logger.error(f"Failed to queue evicted memory summarization: {e}")
            
        if self.use_redis:
            try:
                key = f"session_history:{session_id}"
                # Expire after 1 hour of inactivity
                self.redis_client.setex(key, 3600, json.dumps(history))
                return
            except Exception as e:
                logger.error(f"Error saving session to Redis: {e}")
                
        self.local_store[session_id] = history

session_memory = SessionMemoryManager()
