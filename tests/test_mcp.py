import asyncio
import json
from mcp_server import list_tools, call_tool, catalog

async def test():
    print("--- Listing Registered Tools ---")
    try:
        tools = await list_tools()
        for tool in tools:
            print(f"[REGISTERED] {tool.name}")
    except Exception as e:
        print(f"Error listing tools: {e}")

    print("\n--- Testing 'search_and_execute' (Unified Context) ---")
    # This will use the mock context or actual context if DB is populated
    try:
        result = await call_tool("search_and_execute", {"question": "What is Site 112?"})
        print(f"Response: {result[0].text}")
    except Exception as e:
        print(f"Error calling search_and_execute: {e}")

    print("\n--- Testing Dynamic Discovery (get_rca_report) ---")
    try:
        result = await call_tool("get_rca_report", {"incident_id": "INC-999", "site_name": "London"})
        print(f"Response: {result[0].text}")
    except Exception as e:
        print(f"Error calling dynamic tool: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    asyncio.run(test())
