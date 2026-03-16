import asyncio
import os
import unittest
from unittest.mock import patch, MagicMock
from ai_manager import ai_manager
from main import get_hybrid_context
import time

class TestHybridFallback(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Reset ai_manager state before each test
        ai_manager.openai_active = True
        ai_manager.last_check = 0

    async def test_1_openai_success(self):
        """
        Verify that OpenAI is used when healthy.
        """
        print("\n🧪 Testing OpenAI Success Path...")
        self.assertTrue(ai_manager.is_openai_working())
        self.assertEqual(ai_manager.get_model_name(), "openai")
        self.assertEqual(ai_manager.get_embedding_dimension(), 1536)

    async def test_2_quota_failure_trigger(self):
        """
        Verify that a RateLimitError triggers local fallback.
        """
        print("\n🧪 Testing Quota Failure Trigger...")
        
        # Simulate a 429 Error
        mock_error = Exception("Error code: 429 - {'error': {'message': 'You exceeded your current quota'}}")
        ai_manager.handle_openai_error(mock_error)
        
        self.assertFalse(ai_manager.is_openai_working())
        self.assertEqual(ai_manager.get_model_name(), "local_minilm")
        self.assertEqual(ai_manager.get_embedding_dimension(), 384)
        print("✅ Correctly switched to Local Mode.")

    async def test_3_local_embedding_generation(self):
        """
        Verify that local embeddings are generated correctly and have 384 dimensions.
        """
        print("\n🧪 Testing Local Embedding Generation...")
        ai_manager.openai_active = False # Force local
        
        vector = await ai_manager.get_embeddings("This is a test document for local search.")
        
        self.assertEqual(len(vector), 384)
        print(f"✅ Generated local vector with {len(vector)} dimensions.")

    async def test_4_cooldown_logic(self):
        """
        Verify that the system doesn't probe OpenAI during the cooldown.
        """
        print("\n🧪 Testing Cooldown Logic...")
        ai_manager.openai_active = False
        ai_manager.last_check = time.time() - 100 # Only 100 seconds passed
        
        # Should still be DOWN because 600 seconds haven't passed
        self.assertFalse(ai_manager.is_openai_working())
        print("✅ Cooldown active, OpenAI not probed.")

if __name__ == "__main__":
    unittest.main()
