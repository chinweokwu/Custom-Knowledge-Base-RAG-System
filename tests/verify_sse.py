import asyncio
import httpx
from httpx_sse import connect_sse
import subprocess
import time
import os

async def test_sse_connection():
    print("🚀 Starting MCP SSE Verification...")
    
    # 1. Start the MCP Server in a background process
    env = os.environ.copy()
    env["PYTHONPATH"] = "d:\\AI knowledge Based"
    server_process = subprocess.Popen(
        ["python", "d:\\AI knowledge Based\\app\\mcp\\mcp_server.py", "--transport", "sse", "--port", "9382"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    print("⏳ Waiting for server to start...")
    time.sleep(5) # Give it time to bind the port
    
    try:
        url = "http://localhost:9382/sse"
        print(f"🔗 Connecting to {url}...")
        
        async with httpx.AsyncClient(timeout=10) as client:
            async with connect_sse(client, "GET", url) as event_source:
                print("✅ SSE Connection Established!")
                
                # We expect an initial 'endpoint' event from MCP SSE transport
                async for event in event_source.aiter_sse():
                    print(f"📥 Received Event: {event.event}")
                    print(f"📦 Data: {event.data}")
                    
                    if event.event == "endpoint":
                        post_url = f"http://localhost:9382{event.data}"
                        print(f"📝 Found message endpoint: {post_url}")
                        
                        # Test a basic JSON-RPC call (initialize)
                        init_payload = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "verify-client", "version": "1.0.0"}
                            }
                        }
                        
                        resp = await client.post(post_url, json=init_payload)
                        print(f"📤 Sent initialize request. Status: {resp.status_code}")
                        
                        if resp.status_code == 202 or resp.status_code == 200:
                            print("✅ SSE Transport verified successfully!")
                        break
    
    except Exception as e:
        print(f"❌ SSE Verification FAILED: {e}")
    finally:
        print("🛑 Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    asyncio.run(test_sse_connection())
