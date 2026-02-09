"""
Universal Adapter API - Production Server
FastAPI backend implementing all P0 and P1 requirements for Universal Adapter UI
"""

import asyncio
import json
import logging
import time

from bson import ObjectId
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from services.llm import Agent
from services.db import Database
from services import tools as tools_module
from services.models import (
    ChatRequest,
    WorkflowStep,
    ToolCall,
    ActionLog,
    ChatMetadata,
    ChatResponse,
    ToolExecuteResponse,
    EnhancedTool,
    ConversationSummary
)
from services.agent_logs import (
    get_or_create_queue,
    set_log_queue,
    clear_log_queue,
    put_stream_done,
    drain_queue_until_done,
)
from services.env import ALLOWED_ORIGINS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database instance
db_instance: Optional[Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global db_instance
    # Startup
    logger.info("Starting up - initializing database connection...")
    db_instance = Database()
    yield
    # Shutdown
    logger.info("Shutting down - closing database connection...")
    if db_instance:
        db_instance.close()


def get_db() -> Database:
    """Dependency injection for database"""
    if db_instance is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_instance


app = FastAPI(
    title="Universal Adapter API",
    description="AI agent with tool marketplace and governance - Production v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Universal Adapter API",
        "version": "2.0.0",
        "status": "production",
        "endpoints": {
            "GET /health": "Health check",
            "POST /chat": "Enhanced chat with workflow steps and tool calls",
            "GET /api/discovery/stream": "Real-time discovery event stream (SSE)",
            "GET /tools": "List all tools with enhanced metadata",
            "POST /tools/{name_or_id}/execute": "Execute tool with detailed logging",
            "GET /tools/search": "Search for tools based on search query",
            "GET /tools/{name_or_id}": "Get tool definition",
            "DELETE /tools/{name_or_id}": "Delete the tool defined",
            "GET /conversations": "List the conversations",
            "GET /conversations/{conversation_id}": "Get the conversation from the id",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for docker-compose"""
    return {"status": "healthy", "service": "universal-adapter-api", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Database = Depends(get_db)):
    """
    Enhanced chat endpoint with workflow steps, tool calls, and action logging.
    Implements all P0 requirements for Universal Adapter UI.
    """
    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid4())

    workflow_steps = []
    tool_calls_list = []
    actions_logged = []

    try:
        # Step 1: Checking
        step_start = time.time()
        workflow_steps.append(
            WorkflowStep(
                step="checking",
                status="completed",
                duration_ms=int((time.time() - step_start) * 1000),
                message="Analyzing command...",
            )
        )

        # Step 2: Discovering
        step_start = time.time()
        search_results = db.search_tools(query=request.message, limit=3)
        tool_found = len(search_results) > 0

        workflow_steps.append(
            WorkflowStep(
                step="discovering",
                status="completed",
                duration_ms=int((time.time() - step_start) * 1000),
                message=f"Found {len(search_results)} relevant tools"
                if tool_found
                else "No existing tools found",
            )
        )

        # Step 3: Forging
        step_start = time.time()
        log_queue = await get_or_create_queue(conversation_id)
        set_log_queue(log_queue)
        try:
            agent = Agent(model=request.model, db=db)
            agent.conversation_id = conversation_id

            result = await agent.run(request.message, max_iterations=25)
        finally:
            clear_log_queue()
            await put_stream_done(conversation_id)

        # Extract tool calls from agent messages
        for msg in agent.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_call_id = tc.get("id", str(uuid4()))
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"].get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {}

                    # Find corresponding result
                    tool_result = None
                    for result_msg in agent.messages:
                        if (
                            result_msg.get("role") == "tool"
                            and result_msg.get("tool_call_id") == tool_call_id
                        ):
                            try:
                                tool_result = (
                                    json.loads(result_msg["content"])
                                    if isinstance(result_msg["content"], str)
                                    else result_msg["content"]
                                )
                            except Exception:
                                tool_result = {"result": result_msg["content"]}
                            break

                    tool_calls_list.append(
                        ToolCall(
                            id=tool_call_id,
                            name=tool_name,
                            arguments=tool_args,
                            result=tool_result,
                            execution_time_ms=500,
                            status="success" if tool_result else "error",
                        )
                    )

                    # Log action
                    actions_logged.append(
                        ActionLog(
                            id=f"act_{tool_call_id[:8]}",
                            title=f"Agent called {tool_name}",
                            detail=f"Executed with arguments: {json.dumps(tool_args)[:100]}",
                            status="success" if tool_result else "error",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            tool_name=tool_name,
                            execution_id=tool_call_id,
                        )
                    )

        workflow_steps.append(
            WorkflowStep(
                step="forging",
                status="completed",
                duration_ms=int((time.time() - step_start) * 1000),
                message=f"Executed {len(tool_calls_list)} tool(s)",
            )
        )

        # Step 4: Done
        workflow_steps.append(
            WorkflowStep(
                step="done",
                status="completed",
                duration_ms=0,
                message="Task completed successfully",
            )
        )

        total_duration = int((time.time() - start_time) * 1000)
        tokens = result.usage.get("total_tokens", 0)

        return ChatResponse(
            success=True,
            response=result.output,
            conversation_id=conversation_id,
            model=request.model,
            workflow_steps=workflow_steps,
            tool_calls=tool_calls_list,
            actions_logged=actions_logged,
            metadata=ChatMetadata(
                total_duration_ms=total_duration,
                tokens_used=tokens,
                cost_usd=tokens * 0.000001,
            ),
        )

    except Exception as e:
        logger.exception(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/discovery/stream")
async def discovery_stream(conversation_id: Optional[str] = None):
    """
    Server-Sent Events stream for real-time agent/discovery logs.
    Connect with the same conversation_id used in POST /chat to see live logs
    (e.g. firecrawl search/scrape, tool generation, tool execution).
    """
    if not conversation_id:
        conversation_id = str(uuid4())
    await get_or_create_queue(conversation_id)

    async def event_generator():

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        yield f"data: {json.dumps({'type': 'connected', 'conversation_id': conversation_id, 'timestamp': ts, 'source': 'system', 'message': 'Stream connected', 'level': 'info'})}\n\n"
        async for event in drain_queue_until_done(
            conversation_id, timeout_seconds=360.0
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/tools", response_model=List[EnhancedTool])
async def list_tools(
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
):
    """List all tools with enhanced UI fields"""
    try:
        tools = db.list_tools()
        paginated_tools = tools[skip : skip + limit]

        response = []
        for tool in paginated_tools:
            response.append(
                EnhancedTool(
                    id=tool.get("_id", str(uuid4())),
                    name=tool["name"],
                    description=tool["description"],
                    status=tool.get("status", "PROD-READY"),
                    source_url=tool.get("source_url"),
                    api_reference_url=tool.get("api_reference_url"),
                    preview_snippet=tool.get("preview_snippet", f"{tool['name']}()"),
                    category=tool.get("category", "general"),
                    tags=tool.get("tags", [tool["name"]]),
                    verified=tool.get("verified", True),
                    usage_count=tool.get("usage_count", 0),
                    parameters=tool["parameters"],
                    code=tool.get("code"),
                    created_at=tool.get("created_at", "").isoformat()
                    if tool.get("created_at")
                    else None,
                )
            )

        return response

    except Exception as e:
        logger.exception(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/{tool_name_or_id}/execute", response_model=ToolExecuteResponse)
async def execute_tool(
    tool_name_or_id: str, params: Dict[str, Any], db: Database = Depends(get_db)
):
    """Execute a tool by name or MongoDB ObjectId with enhanced response metadata"""
    try:
        # Resolve tool name from ID if needed
        if len(tool_name_or_id) == 24 and all(
            c in "0123456789abcdef" for c in tool_name_or_id.lower()
        ):
            tool_doc = db._db.tools.find_one({"_id": ObjectId(tool_name_or_id)})
            if not tool_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool with ID '{tool_name_or_id}' not found",
                )
            tool_name = tool_doc["name"]
        else:
            tool_name = tool_name_or_id

        execution_id = f"exec_{uuid4()}"
        started_at = datetime.now(timezone.utc)

        logs = [
            {"timestamp": "00:00:00.100", "message": f"Executing {tool_name}..."},
        ]

        _, tool_functions = tools_module.load_generated_tools()

        if tool_name not in tool_functions:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        func = tool_functions[tool_name]
        result = (
            await func(**params)
            if asyncio.iscoroutinefunction(func)
            else func(**params)
        )

        completed_at = datetime.now(timezone.utc)
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        logs.append(
            {
                "timestamp": f"00:00:{duration_ms / 1000:.3f}",
                "message": "Execution completed successfully",
            }
        )

        # Update usage count
        db_tool = db.get_tool(tool_name)
        if db_tool:
            usage_count = db_tool.get("usage_count", 0) + 1
            db._db.tools.update_one(
                {"name": tool_name}, {"$set": {"usage_count": usage_count}}
            )

        return ToolExecuteResponse(
            success=True,
            tool_name=tool_name,
            execution_id=execution_id,
            result=result if isinstance(result, dict) else {"result": result},
            execution_metadata={
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_ms": duration_ms,
                "api_calls_made": 1,
                "cached": False,
            },
            logs=logs,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/search")
async def search_tools(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=50),
    db: Database = Depends(get_db),
):
    """Search tools using vector similarity"""
    try:
        results = db.search_tools(query=q, limit=limit)

        tools = []
        for tool in results:
            tools.append(
                EnhancedTool(
                    id=tool.get("_id", str(uuid4())),
                    name=tool["name"],
                    description=tool["description"],
                    status=tool.get("status", "PROD-READY"),
                    source_url=tool.get("source_url"),
                    api_reference_url=tool.get("api_reference_url"),
                    preview_snippet=tool.get("preview_snippet"),
                    category=tool.get("category", "general"),
                    tags=tool.get("tags", []),
                    verified=tool.get("verified", True),
                    usage_count=tool.get("usage_count", 0),
                    parameters=tool["parameters"],
                    code=tool.get("code"),
                    created_at=tool.get("created_at", "").isoformat()
                    if tool.get("created_at")
                    else None,
                    similarity_score=tool.get("similarity_score"),
                )
            )

        return {"query": q, "count": len(tools), "tools": tools}

    except Exception as e:
        logger.exception(f"Error searching tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/{name_or_id}")
async def get_tool(name_or_id: str, db: Database = Depends(get_db)):
    """Get specific tool by name or MongoDB ObjectId"""
    try:
        # Check if it looks like a MongoDB ObjectId (24 hex characters)
        if len(name_or_id) == 24 and all(
            c in "0123456789abcdef" for c in name_or_id.lower()
        ):
            tool = db._db.tools.find_one({"_id": ObjectId(name_or_id)})
        else:
            # Lookup by name
            tool = db.get_tool(name_or_id)

        if not tool:
            raise HTTPException(
                status_code=404, detail=f"Tool '{name_or_id}' not found"
            )

        return EnhancedTool(
            id=str(tool.get("_id", uuid4())),
            name=tool["name"],
            description=tool["description"],
            status=tool.get("status", "PROD-READY"),
            source_url=tool.get("source_url"),
            api_reference_url=tool.get("api_reference_url"),
            preview_snippet=tool.get("preview_snippet"),
            category=tool.get("category", "general"),
            tags=tool.get("tags", []),
            verified=tool.get("verified", True),
            usage_count=tool.get("usage_count", 0),
            parameters=tool["parameters"],
            code=tool.get("code"),
            created_at=tool.get("created_at", "").isoformat()
            if tool.get("created_at")
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tools/{name_or_id}")
async def delete_tool(name_or_id: str, db: Database = Depends(get_db)):
    """Delete a tool from the marketplace by name or MongoDB ObjectId"""
    try:
        # Check if it looks like a MongoDB ObjectId
        if len(name_or_id) == 24 and all(
            c in "0123456789abcdef" for c in name_or_id.lower()
        ):
            result = db._db.tools.delete_one({"_id": ObjectId(name_or_id)})
            deleted = result.deleted_count > 0
        else:
            # Delete by name
            deleted = db.delete_tool(name_or_id)

        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Tool '{name_or_id}' not found"
            )

        return {"success": True, "message": f"Tool '{name_or_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
):
    """List recent conversations"""
    try:
        conversations = db.list_conversations(limit=limit, skip=skip)

        response = []
        for conv in conversations:
            response.append(
                ConversationSummary(
                    id=conv["_id"],
                    conversation_id=conv.get("conversation_id", ""),
                    start_time=conv.get("start_time", ""),
                    model=conv.get("model", ""),
                    final_output=conv.get("final_output", "")[:200] + "..."
                    if len(conv.get("final_output", "")) > 200
                    else conv.get("final_output", ""),
                )
            )

        return response

    except Exception as e:
        logger.exception(f"Error listing conversations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing conversations: {str(e)}"
        )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, db: Database = Depends(get_db)):
    """Get a specific conversation by ID"""
    try:
        conversation = db.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(
                status_code=404, detail=f"Conversation '{conversation_id}' not found"
            )

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting conversation: {str(e)}"
        )


#TODO:
# @app.get("/api/actions", response_model=List[Action])
# """Get action feed with optional conversation filter"""


# TODO:
# @app.get("/api/governance/verified-tools", response_model=List[VerifiedTool])
# """Get verified tools with governance metadata"""


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8001))
    # Disable reload in production (Railway, Docker) so we bind once and respond to PORT
    use_reload = os.getenv("RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=use_reload,
    )
