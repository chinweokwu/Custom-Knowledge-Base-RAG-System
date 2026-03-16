import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.api.main import app

# Initialize the FastAPI TestClient
client = TestClient(app)

def test_health_check():
    """Verify the API is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.api.main.process_and_store_batch.delay")
def test_ingest_endpoint(mock_delay):
    """
    Test that the ingestion endpoint correctly triggers 
    a background task and returns a task_id.
    """
    # Mock the Celery task return value
    mock_task = MagicMock()
    mock_task.id = "test-task-123"
    mock_delay.return_value = mock_task

    payload = {
        "content": "This is a test memory about HNSW indexing.",
        "metadata": {"source": "unit_test", "priority": "high"}
    }
    
    response = client.post("/ingest", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert response.json()["task_id"] == "test-task-123"
    
    # Verify the background task was actually called with the right data
    mock_delay.assert_called_once_with(payload["content"], payload["metadata"])

@patch("app.api.main.embeddings.embed_query")
@patch("app.api.main.pool.connection")
def test_streaming_search(mock_conn, mock_embeddings):
    """
    Test the streaming retrieval logic by mocking the 
    OpenAI embedding call and the Database response.
    """
    # 1. Mock OpenAI Embedding response
    mock_embed_response = MagicMock()
    mock_embed_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_embeddings.return_value = mock_embed_response

    # 2. Mock Database Cursor and Rows
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("Memory 1 content", {"tag": "test"}, 0.95),
        ("Memory 2 content", {"tag": "demo"}, 0.88)
    ]
    
    # Setup the context manager chain: pool.connection() -> conn -> cursor() -> cur
    mock_context_conn = MagicMock()
    mock_context_conn.__enter__.return_value = MagicMock()
    mock_context_conn.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.return_value = mock_context_conn

    # 3. Call the streaming endpoint
    # We use stream=True to test the response behavior
    with client.stream("GET", "/search/stream", params={"query": "test query", "limit": 2}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"
        
        # Read the streamed lines
        lines = list(response.iter_lines())
        assert len(lines) == 2
        
        # Parse the first streamed memory
        first_result = json.loads(lines[0])
        assert first_result["content"] == "Memory 1 content"
        assert first_result["score"] == 0.95
        assert first_result["metadata"] == {"tag": "test"}

if __name__ == "__main__":
    # Setup instructions for the user
    print("\n🚀 Starting AI Memory System Unit Tests...")
    print("------------------------------------------")
    pytest.main([__file__])
