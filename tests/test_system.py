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
@patch("app.api.main.extract_chunks_from_source")
def test_ingest_file_endpoint(mock_extract, mock_delay):
    """
    Test that the ingest file endpoint correctly extracts chunks,
    triggers a background batch task, and returns a task_id.
    """
    mock_extract.return_value = ["chunk 1", "chunk 2"]
    
    mock_task = MagicMock()
    mock_task.id = "test-task-file-123"
    mock_delay.return_value = mock_task

    payload = {
        "source": "dummy_path.pdf",
        "metadata": {"source_type": "manual"},
        "heavy_parsing": False
    }
    
    response = client.post("/ingest/file", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert response.json()["task_id"] == "test-task-file-123"
    
    mock_extract.assert_called_once_with("dummy_path.pdf", False)
    mock_delay.assert_called_once_with(["chunk 1", "chunk 2"], {"source_type": "manual"})

@patch("app.api.main.perform_agentic_search")
def test_chat_endpoint(mock_search):
    """Verify the chat endpoint correctly forwards calls to perform_agentic_search."""
    mock_search.return_value = {
        "answer": "Stated fact from knowledge base.",
        "sources": [{"content": "Source fragment 1"}],
        "confidence": "HIGH"
    }

    payload = {
        "message": "What is the network setup?",
        "limit": 10
    }
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["answer"] == "Stated fact from knowledge base."
    assert res_data["confidence"] == "HIGH"
    
    mock_search.assert_called_once_with("What is the network setup?", 10, [])

@patch("app.api.main.synthesize_dashboard_report")
def test_search_endpoint(mock_report):
    """Verify the dashboard search endpoint correctly synthesizes reports."""
    mock_report.return_value = {
        "query": "alarm guide",
        "context": [{"content": "Source alarm docs"}],
        "answer": "Guide details.",
        "confidence": "MEDIUM"
      }
    
    response = client.get("/search", params={"query": "alarm guide", "limit": 5})
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["query"] == "alarm guide"
    assert res_data["answer"] == "Guide details."
    
    mock_report.assert_called_once_with("alarm guide", 5, [])

if __name__ == "__main__":
    pytest.main([__file__])
