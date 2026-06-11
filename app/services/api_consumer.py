import os
import httpx
import asyncio
import base64
import re
from typing import List, Dict, Any, Tuple
from app.core.logger_config import get_logger
from dotenv import load_dotenv

logger = get_logger("api_consumer")
load_dotenv()

class ChatLogConsumer:
    def __init__(self):
        self.api_url = os.getenv("CHAT_LOG_API_URL")
        self.app_id = os.getenv("CHAT_LOG_APP_ID")
        self.app_secret = os.getenv("CHAT_LOG_APP_SECRET")

    async def fetch_chat_logs(self, limit: int = 100, start: int = 0) -> Dict[str, Any]:
        """
        Fetches chat logs from the external API using POST request.
        """
        if not self.api_url or not self.app_id or not self.app_secret:
            logger.error("API configuration missing in environment variables.")
            raise ValueError("API configuration missing.")

        auth_str = f"{self.app_id}:{self.app_secret}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }
        
        body = {
            "limit": limit,
            "start": start
        }

        logger.info(f"Fetching chat logs from {self.api_url} (POST, body={body})")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(self.api_url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise

    def process_logs_to_chunks(self, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the API response and converts chat logs into searchable text chunks with metadata.
        Returns a list of dicts: {"content": str, "external_id": str}
        """
        results = api_response.get("results", [])
        processed_data = []
        
        for log in results:
            log_id = log.get("id")
            if not log_id:
                continue
                
            # Handle potential None values (null in JSON)
            user_q = log.get("user_question") or "N/A"
            ai_think = log.get("ai_think") or "N/A"
            ai_resp = log.get("ai_response") or "N/A"
            
            # Ensure they are strings
            user_q = str(user_q)
            ai_think = str(ai_think)
            ai_resp = str(ai_resp)
            
            # Remove HTML tags if present in ai_response
            clean_ai_resp = re.sub('<[^<]+?>', '', ai_resp)
            
            # Format the chunk to include all relevant context for retrieval
            formatted_chunk = (
                f"--- CHAT LOG ENTRY ---\n"
                f"USER QUESTION: {user_q}\n\n"
                f"AI THINKING PROCESS:\n{ai_think}\n\n"
                f"AI RESPONSE:\n{clean_ai_resp}\n"
                f"----------------------"
            )
            
            processed_data.append({
                "content": formatted_chunk,
                "external_id": str(log_id),
                "create_time": log.get("create_time"),
                "session_id": log.get("session_id")
            })
            
        logger.info(f"Processed {len(processed_data)} chat logs into chunks.")
        return processed_data

# Singleton instance
api_consumer = ChatLogConsumer()
