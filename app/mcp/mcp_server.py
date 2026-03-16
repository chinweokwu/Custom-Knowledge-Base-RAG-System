import os
import sys
import json
import logging
import asyncio

# Ensure project root is in sys.path for portable environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import httpx
from typing import List, Dict, Any, Optional
from mcp.server.lowlevel import Server
import mcp.types as types
from dotenv import load_dotenv
from app.core.logger_config import get_logger
from langchain_core.prompts import ChatPromptTemplate
from app.core.ai_manager import ai_manager

# Initialize Logger
logger = get_logger("mcp_server")

# Import our core logic from other files
from app.api.main import get_hybrid_context
from app.services.loaders import structural_splitter
from app.core.ai_manager import ai_manager, llm
from app.core.database import pool
from app.services.tasks import process_and_store_batch

load_dotenv()

# Initialize MCP Server
server = Server("company-knowledge-server")

# Load Service Catalog for Dynamic Tools
CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "service_catalog.json")

def load_catalog():
    logger.info(f"Loading service catalog from: {CATALOG_PATH}")
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r") as f:
                data = json.load(f)
                logger.info(f"Catalog loaded successfully with {len(data.get('tools', []))} dynamic tools.")
                return data
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
    else:
        logger.warning("Service catalog file not found.")
    return {"tools": []}

catalog = load_catalog()

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """
    Lists the available tools, combining static RAG tools and dynamic API tools.
    """
    logger.info("Listing available tools to MCP client.")
    tools = [
        types.Tool(
            name="ingest_company_document",
            description="Ingest a new document (text, PDF, or URL) into the company's private knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The raw text content or a URL/Path to ingest."},
                    "source_type": {"type": "string", "enum": ["text", "file", "url"], "description": "The type of content being provided."},
                    "metadata": {"type": "object", "description": "Optional metadata like author or department."}
                },
                "required": ["content", "source_type"]
            }
        ),
        types.Tool(
            name="search_and_execute",
            description="Searches company knowledge and executes the most relevant internal tools based on findings. Best for complex requests like 'Find RCA for site X'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's query or problem descripiton."},
                    "context_limit": {"type": "number", "description": "Number of memory chunks to retrieve.", "default": 5}
                },
                "required": ["question"]
            }
        )
    ]

    # Add Dynamic Tools from Catalog
    for dynamic_tool in catalog.get("tools", []):
        tools.append(types.Tool(
            name=dynamic_tool["name"],
            description=dynamic_tool["description"],
            inputSchema=dynamic_tool["parameters"]
        ))

    return tools

@server.list_resources()
async def list_resources() -> List[types.Resource]:
    """
    Lists available resources, like the knowledge gaps report.
    """
    return [
        types.Resource(
            uri="company://reports/knowledge-gaps",
            name="Knowledge Gaps Report",
            description="A real-time report of unanswered queries and low-confidence searches.",
            mimeType="text/markdown"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """
    Reads a specific resource by URI.
    """
    if uri == "company://reports/knowledge-gaps":
        logger.info("Generating Knowledge Gaps Report resource.")
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT query, best_score, created_at FROM knowledge_gaps ORDER BY created_at DESC LIMIT 20;")
                rows = cur.fetchall()
                if not rows:
                    return "# Knowledge Gaps Report\n\n✅ No knowledge gaps detected yet!"
                
                report = "# Knowledge Gaps Report\n\n"
                report += "| Query | Best Score | Detected At |\n| :--- | :--- | :--- |\n"
                for r in rows:
                    report += f"| {r[0]} | {r[1]:.4f} | {r[2]} |\n"
                return report
    
    raise ValueError(f"Resource not found: {uri}")

@server.list_prompts()
async def list_prompts() -> List[types.Prompt]:
    """
    Lists available pre-defined prompts for the AI.
    """
    return [
        types.Prompt(
            name="analyze_company_expert",
            description="Expert-level analysis of company state and knowledge base.",
            arguments=[
                types.PromptArgument(
                    name="focus_area",
                    description="The specific department or topic to analyze.",
                    required=False
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> types.GetPromptResult:
    """
    Returns the content of a pre-defined prompt.
    """
    if name == "analyze_company_expert":
        focus = (arguments or {}).get("focus_area", "General")
        return types.GetPromptResult(
            description="Company Expert Analysis Prompt",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"You are a Senior Company Expert analyzing the state of '{focus}'. "
                             f"Use the company memory tools to discover recent actions, gaps, and documents. "
                             f"Provide a strategic summary and recommended next steps."
                    )
                )
            ]
        )
    raise ValueError(f"Prompt not found: {name}")

async def record_action_trace(action_name: str, parameters: dict, result: Any, summary: str):
    """
    Stores the result of an MCP action back into the Vector DB for "Action Memory".
    """
    logger.info(f"Recording action trace for tool: {action_name}")
    try:
        trace_content = f"ACTION TRACE: {action_name}\nSUMMARY: {summary}\nRESULT: {json.dumps(result)[:500]}..."
        metadata = {
            "type": "action_trace",
            "action": action_name,
            "parameters": parameters,
            "timestamp": "now",
            "authority": 1.0 # Actions taken by server are high authority
        }
        
        # Split and Send to Celery
        chunks = text_splitter.split_text(trace_content)
        process_and_store_batch.delay(chunks, metadata)
        logger.info(f"Action trace for {action_name} queued for vector DB storage.")
    except Exception as e:
        logger.error(f"Failed to record action trace: {e}")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """
    Handles tool execution.
    """
    logger.info(f"MCP Tool Call: {name} with arguments: {arguments}")
    
    # 1. Handle Ingestion
    if name == "ingest_company_document":
        try:
            content = arguments["content"]
            meta = arguments.get("metadata", {})
            chunks = structural_splitter.split_text(content)
            process_and_store_batch.delay(chunks, meta)
            logger.info(f"Ingestion tool triggered for {len(chunks)} chunks.")
            return [types.TextContent(type="text", text=f"Ingestion started for {len(chunks)} chunks.")]
        except Exception as e:
            logger.exception(f"Error in ingest_company_document: {e}")
            raise

    # 2. Handle Unified Search & Execute
    if name == "search_and_execute":
        try:
            question = arguments["question"]
            limit = arguments.get("context_limit", 5)
            
            logger.info(f"Executing search_and_execute for: {question}")
            # Search Vector DB
            context = await get_hybrid_context(question, limit)
            context_str = "\n".join([d['content'] for d in context])
            
            # 2. Analyze context to suggest next tool
            tools_list = [t['name'] for t in catalog.get('tools', [])]
            decision_prompt_template = ChatPromptTemplate.from_template("""
                User Question: {question}
                Retrieved Context: {context}
                Available Corporate Tools: {tools}
                
                Based on the context, which specific corporate tool should be called next to solve the user's problem? 
                Requirements:
                - If a tool matches the need, return ONLY the tool name.
                - If no tool matches, return 'none'.
                - Do not provide explanation.
            """)
            
            decision_chain = decision_prompt_template | llm
            suggested_tool_content = await ai_manager.call_llm(decision_chain, {
                "question": question,
                "context": context_str,
                "tools": ", ".join(tools_list)
            })
            
            suggested_tool = suggested_tool_content.strip().lower()
            
            suggestion_text = ""
            # Case-insensitive match for the suggested tool
            matching_tool = next((t for t in tools_list if t.lower() == suggested_tool), None)
            
            if matching_tool:
                 suggestion_text = f"\n\n👉 **Recommended Next Action:** Call the `{matching_tool}` tool for detailed execution."
            else:
                 suggestion_text = "\n\n👉 **Recommended Next Action:** Continue reasoning or ask for more specific info."

            logger.info(f"LangChain suggested tool: {suggested_tool}")
            return [types.TextContent(
                type="text", 
                text=f"Found {len(context)} relevant facts in company memory:\n\n{context_str}\n" + suggestion_text
            )]
        except Exception as e:
            logger.exception(f"Error in search_and_execute: {e}")
            raise

    # 3. Handle Dynamic Tools from Catalog
    for dynamic_tool in catalog.get("tools", []):
        if name == dynamic_tool["name"]:
            try:
                endpoint = dynamic_tool["endpoint"]
                logger.info(f"Executing dynamic tool {name} at endpoint: {endpoint}")
                # Perform the API Call (Simulated for this demo environment)
                result = {"status": "success", "data": f"Simulated data from {endpoint} for {arguments}"}
                
                # Summary for Action Memory
                summary = f"Executed {name} with parameters {arguments} to retrieve company data."
                await record_action_trace(name, arguments, result, summary)
                
                logger.info(f"Dynamic tool {name} execution successful.")
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.exception(f"Error in dynamic tool {name}: {e}")
                # Log failure as well
                await record_action_trace(name, arguments, {"error": str(e)}, f"FAILED execution of {name}")
                raise

    logger.warning(f"Tool {name} not found in catalog or static tools.")
    raise ValueError(f"Tool {name} not found.")

def create_app():
    """
    Creates a Starlette app to serve the MCP server over SSE (Server-Sent Events).
    """
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        logger.info("New SSE client connecting...")
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(
                streams[0], 
                streams[1], 
                server.create_initialization_options()
            )

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run the Company MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse", help="Transport to use")
    parser.add_argument("--port", type=int, default=9382, help="Port for SSE transport")
    args = parser.parse_args()

    if args.transport == "stdio":
        from mcp.server.stdio import stdio_server
        async def run_stdio():
            logger.info("Starting MCP Server in stdio mode...")
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())
        asyncio.run(run_stdio())
    else:
        logger.info(f"Starting MCP SSE Server on port {args.port}...")
        uvicorn.run(create_app(), host="0.0.0.0", port=args.port)
