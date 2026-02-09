from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class ChatRequest(BaseModel):
    """Chat request with enhanced features"""

    message: str = Field(description="User message/command")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for continuation"
    )
    model: str = Field(
        default="google/gemini-3-flash-preview", description="LLM model to use"
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    context: Optional[Dict[str, str]] = Field(
        default=None, description="UI context metadata"
    )


class WorkflowStep(BaseModel):
    """Workflow progress step"""

    step: str = Field(description="Step name: checking, discovering, forging, done")
    status: str = Field(description="Step status: completed, failed")
    duration_ms: int = Field(description="Duration in milliseconds")
    message: str = Field(description="Human-readable message")


class ToolCall(BaseModel):
    """Tool execution record"""

    id: str = Field(description="Tool call ID")
    name: str = Field(description="Tool name")
    arguments: Dict[str, Any] = Field(description="Tool arguments")
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution result"
    )
    execution_time_ms: int = Field(description="Execution time in ms")
    status: str = Field(description="Status: success, error")


class ActionLog(BaseModel):
    """Action log entry for action feed"""

    id: str = Field(description="Action ID")
    title: str = Field(description="Short action description")
    detail: str = Field(description="Detailed description")
    status: str = Field(description="Status: success, pending, error")
    timestamp: str = Field(description="ISO timestamp")
    github_pr_url: Optional[str] = Field(default=None)
    tool_name: Optional[str] = Field(default=None)
    execution_id: Optional[str] = Field(default=None)


class ChatMetadata(BaseModel):
    """Chat response metadata"""

    total_duration_ms: int
    tokens_used: int
    cost_usd: float


class ChatResponse(BaseModel):
    """Enhanced chat response with workflow tracking"""

    success: bool = True
    response: str = Field(description="Agent response text")
    conversation_id: str = Field(description="Conversation ID")
    model: str = Field(description="Model used")
    workflow_steps: List[WorkflowStep] = Field(
        default_factory=list, description="Workflow progress"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools executed"
    )
    actions_logged: List[ActionLog] = Field(
        default_factory=list, description="Actions for feed"
    )
    metadata: ChatMetadata


class ToolExecuteResponse(BaseModel):
    """Tool execution response with metadata"""

    success: bool
    tool_name: str
    execution_id: str
    result: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    logs: List[Dict[str, str]]


class EnhancedTool(BaseModel):
    """Enhanced tool model with UI fields"""

    id: str
    name: str
    description: str
    status: str = Field(
        default="PROD-READY", description="PROD-READY, BETA, DEPRECATED"
    )
    source_url: Optional[str] = Field(default=None, description="Original API docs URL")
    api_reference_url: Optional[str] = Field(
        default=None, description="API documentation URL used to generate tool"
    )
    preview_snippet: Optional[str] = Field(
        default=None, description="Type signature preview"
    )
    category: Optional[str] = Field(default="general", description="Tool category")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    verified: bool = Field(default=False, description="Verification status")
    usage_count: int = Field(default=0, description="Number of executions")
    parameters: Dict[str, Any]
    code: Optional[str] = None
    created_at: Optional[str] = None
    similarity_score: Optional[float] = None


class ConversationSummary(BaseModel):
    """Conversation summary"""

    id: str
    conversation_id: str
    start_time: str
    model: str
    final_output: str

