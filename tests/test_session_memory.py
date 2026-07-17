import pytest
from app.services.session_memory import SessionMemoryManager

def test_session_memory_local_fallback():
    """Verify session memory stores and retrieves conversation history correctly."""
    # Force use of local store for test isolation
    manager = SessionMemoryManager()
    manager.use_redis = False
    
    session_id = "test_session_999"
    assert manager.get_history(session_id) == []
    
    manager.add_message(session_id, "user", "Hello agent")
    manager.add_message(session_id, "assistant", "Hello human")
    
    history = manager.get_history(session_id)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello agent"}
    assert history[1] == {"role": "assistant", "content": "Hello human"}

def test_session_memory_limit():
    """Verify session memory caps history at the max turns limit."""
    manager = SessionMemoryManager()
    manager.use_redis = False
    
    session_id = "test_limit_session"
    
    # Add 12 turns (24 messages)
    for i in range(12):
        manager.add_message(session_id, "user", f"message {i}", max_turns=5)
        manager.add_message(session_id, "assistant", f"reply {i}", max_turns=5)
        
    history = manager.get_history(session_id)
    # With max_turns=5, it should keep the last 5 turns (10 messages)
    assert len(history) == 10
    assert history[0]["content"] == "message 7"
    assert history[-1]["content"] == "reply 11"
