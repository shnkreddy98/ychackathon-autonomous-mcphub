import base64
import httpx
import inspect
import json
import logging
import os

from pathlib import Path
from typing import Optional, Any, Callable
from pydantic import BaseModel

from services.db import Database
from services.agent_logs import emit_log

logger = logging.getLogger(__name__)

# Lazy singleton for database connection
_db_instance: Optional[Database] = None


def get_db_instance() -> Database:
    """Get or create the database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

# MongoDB BSON document size limit (16MB)
MAX_MONGODB_DOC_SIZE = 16 * 1024 * 1024  # 16 MB
# Use 10MB as safe threshold to account for other fields
SAFE_CONTENT_SIZE = 10 * 1024 * 1024  # 10 MB


def _check_document_size(doc: dict, max_size: int = SAFE_CONTENT_SIZE) -> dict:
    """
    Check if document size is within MongoDB limits.
    Returns dict with size info and warnings.

    Args:
        doc: Document to check
        max_size: Maximum safe size in bytes

    Returns:
        dict with size, is_safe, and warnings
    """
    import sys
    doc_size = sys.getsizeof(json.dumps(doc))

    return {
        "size_bytes": doc_size,
        "size_mb": round(doc_size / (1024 * 1024), 2),
        "is_safe": doc_size < max_size,
        "limit_mb": round(max_size / (1024 * 1024), 2)
    }


# ============================================
# File Operations (artifacts directory)
# ============================================

ARTIFACTS_DIR = "artifacts"


def _get_safe_path(filepath: str) -> Path:
    """
    Get a safe path within the artifacts directory.
    Prevents directory traversal attacks.
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Get absolute path and ensure it's within artifacts
    base_path = Path(ARTIFACTS_DIR).resolve()
    target_path = (base_path / filepath).resolve()

    # Check if target is within base directory
    if not str(target_path).startswith(str(base_path)):
        raise ValueError(
            f"Access denied: Path '{filepath}' is outside artifacts directory"
        )

    return target_path


async def file_read(
    filepath: str, mode: str = "r", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Read a file from the artifacts directory.
    Handles both text and binary data.

    Args:
        filepath: Path relative to artifacts directory
        mode: Read mode - 'r' (text) or 'rb' (binary, returns base64)
        encoding: Text encoding (default: utf-8), ignored for binary mode
    """
    try:
        emit_log("agent", f"Reading file: {filepath}", level="info", filepath=filepath)
        safe_path = _get_safe_path(filepath)

        if not safe_path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}

        if not safe_path.is_file():
            return {"success": False, "error": f"Not a file: {filepath}"}

        # Validate mode
        if mode not in ["r", "rb"]:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'r' (text) or 'rb' (binary)",
            }

        # Read binary data
        if mode == "rb":
            binary_data = safe_path.read_bytes()
            content = base64.b64encode(binary_data).decode("ascii")
            content_type = "binary_base64"
        # Read text data
        else:
            content = safe_path.read_text(encoding=encoding)
            content_type = "text"

        return {
            "success": True,
            "filepath": filepath,
            "content": content,
            "content_type": content_type,
            "size": safe_path.stat().st_size,
        }

    except Exception as e:
        return {"success": False, "error": f"Error reading file: {str(e)}"}


async def file_write(
    filepath: str, content: str, mode: str = "w", encoding: str = "utf-8"
) -> dict[str, Any]:
    """
    Write content to a file in the artifacts directory.
    Handles both text and binary data automatically.

    Args:
        filepath: Path relative to artifacts directory
        content: Content to write (string for text, base64 string for binary)
        mode: Write mode - 'w' (overwrite), 'a' (append), 'wb' (binary overwrite), 'ab' (binary append)
        encoding: Text encoding (default: utf-8), ignored for binary mode
    """
    try:
        safe_path = _get_safe_path(filepath)

        # Create parent directories if they don't exist
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate mode
        valid_modes = ["w", "a", "wb", "ab"]
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'w', 'a', 'wb', or 'ab'",
            }

        is_binary = mode in ["wb", "ab"]
        is_append = mode in ["a", "ab"]

        # Handle binary data
        if is_binary:
            # Content should be base64 encoded for binary
            try:
                binary_data = base64.b64decode(content)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Binary mode requires base64 encoded content: {str(e)}",
                }

            if is_append:
                with open(safe_path, "ab") as f:
                    f.write(binary_data)
            else:
                safe_path.write_bytes(binary_data)

        # Handle text data
        else:
            if is_append:
                with open(safe_path, "a", encoding=encoding) as f:
                    f.write(content)
            else:
                safe_path.write_text(content, encoding=encoding)

        return {
            "success": True,
            "filepath": filepath,
            "absolute_path": str(safe_path),
            "size": safe_path.stat().st_size,
            "mode": "binary" if is_binary else "text",
            "operation": "appended" if is_append else "overwritten",
        }

    except Exception as e:
        return {"success": False, "error": f"Error writing file: {str(e)}"}


async def file_list(directory: str = ".") -> dict[str, Any]:
    """
    List files and directories in the artifacts directory.
    """
    try:
        safe_path = _get_safe_path(directory)

        if not safe_path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        if not safe_path.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        items = []
        for item in safe_path.iterdir():
            items.append(
                {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )

        return {
            "success": True,
            "directory": directory,
            "items": items,
            "count": len(items),
        }

    except Exception as e:
        return {"success": False, "error": f"Error listing directory: {str(e)}"}


# ============================================
# Firecrawl API Client
# ============================================


class Firecrawl:
    def __init__(self, api_key: str):
        self.base_url = "https://api.firecrawl.dev/v2"
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 5, timeout: int = 60000
    ) -> dict[str, Any]:
        """
        Search the web using Firecrawl and get full page content.
        """
        emit_log(
            "firecrawl",
            f"Searching web: {query[:80]}...",
            level="info",
            query=query,
            limit=limit,
        )
        url = f"{self.base_url}/search"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "limit": limit, "timeout": timeout}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            count = (
                len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
            )
            emit_log(
                "firecrawl",
                f"Found {count} result(s) for search",
                level="info",
                count=count,
            )
            return data

    async def scrape(
        self,
        url: str,
        formats: list[str] = None,
        only_main_content: bool = True,
        timeout: int = 30000,
    ) -> dict[str, Any]:
        """
        Scrape a single URL and extract content in specified formats.
        """
        emit_log("firecrawl", f"Crawling {url[:80]}...", level="info", url=url)
        api_url = f"{self.base_url}/scrape"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "url": url,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
            "timeout": timeout,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            emit_log(
                "firecrawl",
                f"Scraped content from {url[:50]}...",
                level="info",
                url=url,
            )
            return data

    async def crawl(
        self, url: str, limit: int = 10, max_depth: int = 2, timeout: int = 120000
    ) -> dict[str, Any]:
        """
        Crawl an entire website starting from a URL.
        """
        emit_log(
            "firecrawl",
            f"Crawling site {url[:80]}...",
            level="info",
            url=url,
            limit=limit,
        )
        api_url = f"{self.base_url}/crawl"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "url": url,
            "limit": limit,
            "maxDepth": max_depth,
            "timeout": timeout,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            emit_log(
                "firecrawl", f"Crawl completed for {url[:50]}...", level="info", url=url
            )
            return data


# ============================================
# Tool Storage (MongoDB)
# ============================================


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    code: Optional[str] = None  # Python function code as string
    dependencies: Optional[list[str]] = None  # Required packages
    api_reference_url: Optional[str] = (
        None  # API documentation URL used to generate this tool
    )


def _validate_tool_code(name: str, code: str) -> dict:
    """
    Validate that the generated tool code is syntactically correct
    and contains the expected function definition.

    Args:
        name: Name of the tool/function
        code: Python code to validate

    Returns:
        dict with success status and error message if validation fails
    """
    if not code:
        return {"success": True}  # No code to validate (schema-only tool)

    try:
        # Create execution environment
        exec_globals = _get_safe_exec_environment(code)
        exec_locals = {}

        # Try to compile and execute the code
        compile(code, '<string>', 'exec')
        exec(code, exec_globals, exec_locals)

        # Check if the function with the expected name exists
        if name not in exec_locals:
            return {
                "success": False,
                "error": f"Function '{name}' not found in generated code. The function name must match the tool name."
            }

        # Check if it's actually a function
        if not callable(exec_locals[name]):
            return {
                "success": False,
                "error": f"'{name}' is not a callable function in the generated code."
            }

        return {"success": True}

    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error in generated code: {str(e)}",
            "type": "SyntaxError"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error validating code: {str(e)}",
            "type": type(e).__name__
        }


def save_tool(tool_definition: dict) -> dict:
    """
    Save a generated tool definition to MongoDB.
    Validates and stores both schema and executable code.
    Normalizes parameter schema to ensure JSON Schema compliance.
    Validates that code is syntactically correct before saving.
    """
    # Normalize parameter schema before validation
    if "parameters" in tool_definition:
        tool_definition["parameters"] = normalize_tool_schema(
            tool_definition["parameters"]
        )

    # Validate tool definition
    tool = ToolDefinition(**tool_definition)

    # Validate the generated code BEFORE saving to database
    if tool.code:
        validation_result = _validate_tool_code(tool.name, tool.code)
        if not validation_result.get("success"):
            error_msg = validation_result.get("error", "Code validation failed")
            emit_log(
                "mcp",
                f"Tool '{tool.name}' validation failed: {error_msg}",
                level="error",
                tool_name=tool.name,
                error=error_msg
            )
            return {
                "success": False,
                "error": error_msg,
                "type": validation_result.get("type", "ValidationError")
            }

    # Remove large fields before saving to avoid DocumentTooLarge errors
    # MongoDB has a 16MB BSON document limit
    tool_dict = tool.model_dump()

    # If there's api_docs_content, remove it (we only need hash for change detection)
    if "api_docs_content" in tool_dict:
        import hashlib
        # Store only hash, not full content
        content = tool_dict.pop("api_docs_content")
        tool_dict["api_docs_hash"] = hashlib.sha256(content.encode()).hexdigest()
        logger.info(f"Removed large api_docs_content ({len(content)} bytes), stored hash instead")

    # Check document size before saving
    size_info = _check_document_size(tool_dict)
    if not size_info["is_safe"]:
        logger.warning(
            f"Tool document is large ({size_info['size_mb']}MB). "
            f"Limit: {size_info['limit_mb']}MB. May cause issues."
        )
        # Try to truncate large code if present
        if "code" in tool_dict and len(tool_dict["code"]) > 1_000_000:
            logger.warning("Code field is very large, this may cause DocumentTooLarge errors")

    # Save to MongoDB only if validation passed
    out = get_db_instance().save_tool(tool_dict)
    if out.get("success"):
        emit_log(
            "mcp",
            f"Tool '{tool.name}' registered in marketplace",
            level="success",
            tool_name=tool.name,
        )
    return out


def load_tool(name: str) -> dict:
    """
    Load a tool definition from MongoDB.
    """
    tool = get_db_instance().get_tool(name)

    if tool is None:
        return {"success": False, "error": f"Tool '{name}' not found"}

    return tool


def list_tools() -> list[dict]:
    """
    List all saved tool definitions from MongoDB.
    """
    return get_db_instance().list_tools()


async def search_tools(query: str, limit: int = 10) -> dict:
    """
    Search for tools using vector similarity.
    Returns the top N most relevant tools based on the query.

    This is much more efficient than loading all tools when you have many tools.
    Use this to find tools relevant to a specific task.

    Args:
        query: Description of what you want to do (e.g., "download images", "get weather data")
        limit: Maximum number of tools to return (default: 10)

    Returns:
        Dictionary with success status and list of relevant tools
    """
    try:
        emit_log(
            "agent", f"Searching tools: {query[:60]}...", level="info", query=query
        )
        tools = get_db_instance().search_tools(query, limit)
        emit_log(
            "agent",
            f"Found {len(tools)} relevant tool(s)",
            level="info",
            count=len(tools),
        )
        return {
            "success": True,
            "query": query,
            "count": len(tools),
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "similarity_score": tool.get("similarity_score", 0),
                }
                for tool in tools
            ],
        }
    except Exception as e:
        return {"success": False, "error": f"Error searching tools: {str(e)}"}


def _get_safe_exec_environment(code: str) -> dict:
    """
    Create a safe execution environment with only whitelisted modules.
    Automatically detects needed imports from code and provides them.
    """
    import re
    import asyncio
    import time
    from datetime import datetime, timezone, timedelta

    # Whitelist of safe modules that generated tools can use
    # These are automatically provided when detected in the code
    import hashlib
    import urllib.parse as urllib_parse
    import re as regex_module

    SAFE_MODULES = {
        # HTTP and networking
        "httpx": httpx,
        # Data formats
        "json": json,
        "base64": base64,
        # System and files
        "os": os,
        # Async
        "asyncio": asyncio,
        # Time handling
        "time": time,
        "datetime": datetime,
        "timezone": timezone,
        "timedelta": timedelta,
        # Utilities
        "hashlib": hashlib,
        "urllib": urllib_parse,
        "re": regex_module,
        # File operations (custom)
        "file_write": file_write,
        "file_read": file_read,
        "file_list": file_list,
    }

    # Start with builtins
    exec_globals = {"__builtins__": __builtins__}

    # Extract imports from code using regex
    # Matches: import x, from x import y, from x import y as z
    import_pattern = r"(?:from\s+(\w+)\s+import\s+[\w\s,]+|import\s+([\w\s,]+))"
    matches = re.findall(import_pattern, code)

    # Also look for direct usage of whitelisted modules
    for module_name in SAFE_MODULES.keys():
        if module_name in code:
            if module_name in SAFE_MODULES:
                exec_globals[module_name] = SAFE_MODULES[module_name]

    # Add extracted imports if they're in the whitelist
    for from_module, import_module in matches:
        module = from_module or import_module
        if module and module.strip() in SAFE_MODULES:
            exec_globals[module.strip()] = SAFE_MODULES[module.strip()]

    logger.debug(f"Exec environment provides: {list(exec_globals.keys())}")
    return exec_globals


def normalize_tool_schema(schema: dict) -> dict:
    """
    Normalize tool parameter schema for JSON Schema compliance.
    Converts uppercase type values (STRING, INTEGER, etc.) to lowercase.
    Recursively handles nested objects and arrays.

    Args:
        schema: Tool parameter schema dictionary

    Returns:
        Normalized schema with lowercase type values
    """
    if not isinstance(schema, dict):
        return schema

    normalized = {}

    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            # Normalize type to lowercase (STRING -> string, OBJECT -> object, etc.)
            normalized[key] = value.lower()
        elif key == "properties" and isinstance(value, dict):
            # Recursively normalize nested properties
            normalized[key] = {
                prop_name: normalize_tool_schema(prop_value)
                for prop_name, prop_value in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            # Recursively normalize array item schema
            normalized[key] = normalize_tool_schema(value)
        elif isinstance(value, dict):
            # Recursively normalize any nested dicts
            normalized[key] = normalize_tool_schema(value)
        elif isinstance(value, list):
            # Handle lists (like enum values, required fields)
            normalized[key] = value
        else:
            # Keep other values as-is
            normalized[key] = value

    return normalized


async def execute_generated_tool(tool_name: str, arguments: dict) -> dict:
    """
    Execute a generated tool by loading its code and running it.
    Uses exec() to run the stored Python code string.
    """
    # Load the tool
    tool_data = load_tool(tool_name)

    if not tool_data.get("success", True):
        return {"error": f"Tool '{tool_name}' not found"}

    code = tool_data.get("code")
    if not code:
        return {
            "error": f"Tool '{tool_name}' has no executable code. It's just a schema definition.",
            "suggestion": "You need to generate the implementation code for this tool first.",
        }

    try:
        # Create a safe execution environment with auto-detected dependencies
        exec_globals = _get_safe_exec_environment(code)
        exec_locals = {}

        # Execute the code to define the function
        exec(code, exec_globals, exec_locals)

        # Get the function from locals (assume it's the tool name)
        if tool_name not in exec_locals:
            return {"error": f"Function '{tool_name}' not found in generated code"}

        func = exec_locals[tool_name]

        # Execute the function with arguments
        if inspect.iscoroutinefunction(func):
            result = await func(**arguments)
        else:
            result = func(**arguments)

        return {"success": True, "result": result}

    except Exception as e:
        return {
            "error": f"Error executing generated tool '{tool_name}': {str(e)}",
            "type": type(e).__name__,
        }


def load_generated_tools() -> tuple[list[dict], dict[str, Callable]]:
    """
    Load all generated tools and return their schemas and execution functions.

    Returns:
        (tool_schemas, tool_functions)
    """
    tools = list_tools()

    tool_schemas = []
    tool_functions = {}

    def make_async_wrapper(tool_name: str):
        """Create an async wrapper for a tool with proper closure"""

        async def wrapper(**kwargs):
            return await execute_generated_tool(tool_name, kwargs)

        return wrapper

    for tool in tools:
        # Convert to OpenRouter format with normalized schema
        schema = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": normalize_tool_schema(tool["parameters"]),
            },
        }
        tool_schemas.append(schema)

        # Create async execution wrapper
        tool_name = tool["name"]
        tool_functions[tool_name] = make_async_wrapper(tool_name)

    return tool_schemas, tool_functions


# ============================================
# Tool Registry - Convert to OpenRouter Format
# ============================================


def get_base_tools(firecrawl_api_key: str) -> tuple[list[dict], dict[str, Callable]]:
    """
    Returns base tools in OpenRouter format and their execution functions.

    Returns:
        (tool_schemas, tool_functions)
    """
    firecrawl = Firecrawl(firecrawl_api_key)

    # Tool schemas in OpenRouter format
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read content from a file in the artifacts directory. Supports both text and binary files. Binary files (images, PDFs) are returned as base64 encoded strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file relative to artifacts directory (e.g., 'data.txt', 'images/cat.jpg')",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Read mode: 'r' for text files, 'rb' for binary files (returns base64)",
                            "enum": ["r", "rb"],
                            "default": "r",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (only for text mode, default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file in the artifacts directory. Supports both text and binary data (images, PDFs, etc.). For binary data, use mode 'wb' and provide base64 encoded content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file relative to artifacts directory (e.g., 'output.txt', 'cat/image.jpg')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write. For text: plain string. For binary (images, etc.): base64 encoded string",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Write mode: 'w' (text overwrite), 'a' (text append), 'wb' (binary overwrite), 'ab' (binary append)",
                            "enum": ["w", "a", "wb", "ab"],
                            "default": "w",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (only for text mode, default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["filepath", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_list",
                "description": "List files and directories in the artifacts directory. Use this to explore what files exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path relative to artifacts (default: '.' for root)",
                            "default": ".",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_search",
                "description": "Search the web and get full page content. Use this to find information across the internet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (1-100)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_scrape",
                "description": "Scrape content from a single webpage. Use this to extract content from a specific URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to scrape"},
                        "formats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Content formats to extract (markdown, html, links, screenshot)",
                            "default": ["markdown"],
                        },
                        "only_main_content": {
                            "type": "boolean",
                            "description": "Extract only main content, excluding headers/footers/nav",
                            "default": True,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "firecrawl_crawl",
                "description": "Crawl an entire website starting from a URL. Use this to explore and extract content from multiple pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The starting URL to crawl",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of pages to crawl",
                            "default": 10,
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth of crawling",
                            "default": 2,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_tool",
                "description": "Generate and save a new tool definition with executable code. Use this when you need a capability that doesn't exist yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the new tool (snake_case, must match function name)",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the tool does",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "JSON schema for the tool's parameters",
                        },
                        "code": {
                            "type": "string",
                            "description": "Python function code as a string. Must be a complete async function definition.",
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of required Python packages (e.g., ['httpx', 'beautifulsoup4'])",
                        },
                        "api_reference_url": {
                            "type": "string",
                            "description": "The URL of the API documentation that was used to generate this tool (from firecrawl_scrape)",
                        },
                    },
                    "required": ["name", "description", "parameters", "code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": "Search the tool marketplace for relevant tools using semantic search. Use this to find existing tools before generating new ones. Returns top 10 most relevant tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Description of what you want to do (e.g., 'download images', 'get weather data', 'parse JSON')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tools to return (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    # Tool execution functions
    tool_functions = {
        "file_read": file_read,
        "file_write": file_write,
        "file_list": file_list,
        "firecrawl_search": firecrawl.search,
        "firecrawl_scrape": firecrawl.scrape,
        "firecrawl_crawl": firecrawl.crawl,
        "generate_tool": lambda **kwargs: save_tool(kwargs),
        "search_tools": search_tools,
    }

    return tool_schemas, tool_functions
