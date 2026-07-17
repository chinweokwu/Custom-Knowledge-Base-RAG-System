import pytest
import re
from app.services.loaders import StructuralSplitter

def test_code_block_preservation():
    splitter = StructuralSplitter(chunk_size=200, chunk_overlap=20)
    text = (
        "Some introductory paragraph before the code.\n\n"
        "def test_function(x):\n"
        "    y = x + 1\n"
        "    return y\n\n"
        "Some concluding paragraph after the code."
    )
    chunks = splitter.split_text(text)
    
    # Verify the code block was kept together inside one of the chunks
    code_found = False
    for chunk in chunks:
        if "def test_function" in chunk:
            code_found = True
            assert "return y" in chunk
            
    assert code_found, "Indented code block was not preserved together"

def test_list_block_preservation():
    splitter = StructuralSplitter(chunk_size=300, chunk_overlap=30)
    text = (
        "Here is a technical guide:\n"
        "* Step 1: Open the shell.\n"
        "* Step 2: Run the following setup:\n"
        "      python -m venv venv\n"
        "      source venv/bin/activate\n"
        "* Step 3: Start the worker server.\n"
        "End of document."
    )
    chunks = splitter.split_text(text)
    
    list_found = False
    for chunk in chunks:
        if "Step 1: Open the shell" in chunk:
            list_found = True
            assert "Step 3: Start the worker" in chunk
            assert "python -m venv venv" in chunk
            
    assert list_found, "List block with nested code was fragmented or corrupted"

def test_table_preservation():
    splitter = StructuralSplitter(chunk_size=200, chunk_overlap=20)
    text = (
        "Below is the database state:\n\n"
        "| ID | Status | Message |\n"
        "|----|--------|---------|\n"
        "| 1  | Success| Done    |\n"
        "| 2  | Pending| Waiting |\n\n"
        "Please inspect it carefully."
    )
    chunks = splitter.split_text(text)
    
    table_found = False
    for chunk in chunks:
        if "| ID | Status |" in chunk:
            table_found = True
            assert "| 2  | Pending|" in chunk
            
    assert table_found, "Markdown table was not preserved together"

def test_hierarchical_splitting():
    splitter = StructuralSplitter(chunk_size=150, chunk_overlap=15)
    text = (
        "Top-level introductory description that represents main parent context.\n"
        "This describes a long document layout that needs nested parsing.\n\n"
        "* Detail Item 1: This is a sub-point that goes into details.\n"
        "* Detail Item 2: Another sub-point with separate context details.\n"
        "* Detail Item 3: A third sub-point for completeness.\n"
    )
    hierarchy = splitter.split_text(text, hierarchical=True)
    
    assert len(hierarchy) > 0
    for child, parent in hierarchy:
        assert isinstance(child, str)
        assert isinstance(parent, str)
        assert child in parent or any(p in parent for p in child.split())

def test_oversized_table_splitting():
    splitter = StructuralSplitter(chunk_size=500, chunk_overlap=20)
    rows = [
        "| ID | Status | Message |",
        "|----|--------|---------|"
    ]
    for i in range(1, 40):
        rows.append(f"| {i:02d} | Success| Row number {i:02d} is here to fill up space and make this table extremely long. |")
    text = "\n".join(rows) + "\n"
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert "| ID | Status |" in chunk
        assert "|----|--------|---------|" in chunk
        assert any(f"| {i:02d} |" in chunk for i in range(1, 40))

def test_oversized_code_splitting():
    splitter = StructuralSplitter(chunk_size=500, chunk_overlap=20)
    lines = ["```python"]
    for i in range(1, 40):
        lines.append(f"print('This is line {i:02d} to make the python code block extremely long and exceed the threshold of characters.')")
    lines.append("```")
    text = "\n".join(lines) + "\n"
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().startswith("```python")
        assert chunk.strip().endswith("```")
        assert any(f"line {i:02d}" in chunk for i in range(1, 40))

def test_json_loader(tmp_path):
    import json
    from app.services.loaders import load_document
    
    # 1. Test Small JSON (should fit in a single chunk at "$")
    small_data = {
        "metadata": {
            "title": "Test JSON",
            "version": 1
        }
    }
    
    small_file = tmp_path / "small.json"
    with open(small_file, "w") as f:
        json.dump(small_data, f)
        
    docs_small = load_document(str(small_file))
    assert len(docs_small) == 1
    assert docs_small[0].metadata["json_path"] == "$"
    assert "Test JSON" in docs_small[0].page_content

    # 2. Test Large JSON (should be traversed and split)
    # Create an items list that exceeds 1000 characters
    large_items = []
    for i in range(15):
        large_items.append({
            "id": i,
            "name": f"Item {i}",
            "description": f"This is description for item {i} designed to take up considerable bytes. Let's make sure it is long enough. " * 3
        })
    
    large_data = {
        "metadata": {
            "title": "Large Test JSON",
            "version": 2
        },
        "items": large_items
    }
    
    large_file = tmp_path / "large.json"
    with open(large_file, "w") as f:
        json.dump(large_data, f)
        
    docs_large = load_document(str(large_file))
    
    # Verify we extracted multiple documents due to length
    assert len(docs_large) > 1
    
    # Verify traversal happened
    has_metadata = False
    has_item_element = False
    
    for doc in docs_large:
        path = doc.metadata.get("json_path", "")
        if path.startswith("$.metadata"):
            has_metadata = True
            assert "Large Test JSON" in doc.page_content
        elif "$.items" in path:
            has_item_element = True
            
    assert has_metadata
    assert has_item_element


def test_event_loop_safety(tmp_path):
    import asyncio
    from app.services.loaders import load_document
    
    # Create a small dummy text file
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("Hello event loop safety!")
    
    # Define an async function that runs the sync loader.
    # This simulates a running event loop (e.g. FastAPI request thread).
    async def simulate_fastapi_request():
        # Inside this running async function, we call the sync wrapper.
        # It must NOT throw "RuntimeError: This event loop is already running".
        docs = load_document(str(dummy_file))
        return docs
    
    # Run the simulated FastAPI request using asyncio.run
    docs = asyncio.run(simulate_fastapi_request())
    assert len(docs) == 1
    assert "Hello event loop safety!" in docs[0].page_content




