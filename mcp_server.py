"""
MCP Server for Universal Adapter (stdio transport for local clients)
For remote access, use server.py (FastAPI) instead.
"""

import logging
from mcp.server.fastmcp import FastMCP
from services.llm import Agent
from services.env import FIRECRAWL_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("Universal Adapter")


@mcp.tool()
async def chat(
    message: str,
    conversation_id: str | None = None,
    model: str = "anthropic/claude-haiku-4.5",
    max_iterations: int = 25,
) -> dict:
    """
    Chat with the Universal Adapter agent.

    The agent has access to all marketplace tools including:
    - File operations (read, write, list)
    - Web scraping (firecrawl search, scrape, crawl)
    - Tool generation (generate new tools from API docs)
    - Tool search (semantic search of marketplace)
    - All dynamically generated tools (crypto prices, APIs, etc.)

    Args:
        message: Your question or command
        conversation_id: Optional conversation ID for multi-turn chat
        model: LLM model to use (default: claude-haiku-4.5)
        max_iterations: Maximum agent iterations (default: 25)

    Returns:
        dict: Response with agent output, conversation_id, usage stats
    """
    try:
        logger.info(f"MCP chat request: {message[:100]}")

        # Create agent with all tools
        agent = Agent(
            model=model,
            save_conversations=True,
            firecrawl_api_key=FIRECRAWL_API_KEY,
        )

        # Use existing conversation if provided
        if conversation_id:
            agent.conversation_id = conversation_id

        # Run agent (it has access to all marketplace tools)
        result = await agent.run(message, max_iterations=max_iterations)

        # Extract tool calls from agent messages
        tool_calls = []
        for msg in agent.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls.append(tc["function"]["name"])

        return {
            "response": result.output,
            "conversation_id": agent.conversation_id,
            "usage": result.usage,
            "tool_calls": list(set(tool_calls)),
            "success": True,
        }

    except Exception as e:
        logger.exception(f"Error in chat tool: {e}")
        return {
            "response": f"Error: {str(e)}",
            "conversation_id": conversation_id,
            "usage": {},
            "tool_calls": [],
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_marketplace_tools(limit: int = 50) -> dict:
    """
    List available tools in the marketplace.

    Args:
        limit: Maximum number of tools to return

    Returns:
        dict: List of tools with names and descriptions
    """
    from services.db import Database

    try:
        db = Database()
        tools = db.list_tools()[:limit]
        return {
            "count": len(tools),
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "category": t.get("category", "general"),
                }
                for t in tools
            ],
            "success": True,
        }
    except Exception as e:
        return {"error": str(e), "success": False}


@mcp.tool()
async def health() -> dict:
    """Check if the MCP server is healthy"""
    return {
        "status": "healthy",
        "service": "universal-adapter-mcp",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport (for local clients)")
    logger.info("For remote access, use server.py instead")
    mcp.run()
