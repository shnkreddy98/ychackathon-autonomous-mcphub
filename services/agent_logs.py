"""
Real-time agent log streaming for the frontend.
Uses a per-conversation queue and context so tools/agent can emit events
that GET /api/discovery/stream streams as SSE.
"""

import asyncio
import contextvars
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Per-conversation queues: conversation_id -> asyncio.Queue
_discovery_queues: Dict[str, asyncio.Queue] = {}
_queue_lock = asyncio.Lock()

# Context var so the current request's queue is visible to agent and tools
_log_queue_ctx: contextvars.ContextVar[Optional[asyncio.Queue]] = (
    contextvars.ContextVar("agent_log_queue", default=None)
)

# Sentinel to signal stream end
STREAM_DONE = {"done": True, "level": "info", "source": "system"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def emit_log(
    source: str,
    message: str,
    level: str = "info",
    **metadata: Any,
) -> None:
    """
    Emit a discovery log event to the current conversation's stream (if any).
    Sources: firecrawl, mcp, agent, system.
    """
    queue = _log_queue_ctx.get()
    if queue is None:
        return
    event = {
        "timestamp": _timestamp(),
        "source": source,
        "message": message,
        "level": level,
        **metadata,
    }
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        pass


async def get_or_create_queue(conversation_id: str) -> asyncio.Queue:
    async with _queue_lock:
        if conversation_id not in _discovery_queues:
            _discovery_queues[conversation_id] = asyncio.Queue(maxsize=512)
        return _discovery_queues[conversation_id]


def set_log_queue(queue: asyncio.Queue) -> None:
    _log_queue_ctx.set(queue)


def clear_log_queue() -> None:
    _log_queue_ctx.set(None)


async def put_stream_done(conversation_id: str) -> None:
    """Signal that no more events will be sent for this conversation."""
    async with _queue_lock:
        q = _discovery_queues.get(conversation_id)
    if q is not None:
        try:
            q.put_nowait(STREAM_DONE)
        except asyncio.QueueFull:
            pass


async def drain_queue_until_done(
    conversation_id: str,
    timeout_seconds: float = 300.0,
) -> Any:
    """
    Async generator that yields events from the conversation's queue until STREAM_DONE.
    Used by GET /api/discovery/stream.
    """
    async with _queue_lock:
        q = _discovery_queues.get(conversation_id)
    if q is None:
        return
    while True:
        try:
            event = await asyncio.wait_for(q.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            yield {
                "timestamp": _timestamp(),
                "source": "system",
                "message": "Stream timeout",
                "level": "warn",
            }
            continue
        if event.get("done"):
            yield event
            return
        yield event
