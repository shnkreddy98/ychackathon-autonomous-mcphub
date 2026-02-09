import httpx
import inspect
import json
import logging

from services.logging import get_logger
from services.env import OPENROUTER_API_KEY, FIRECRAWL_API_KEY
from services.tools import get_base_tools, load_generated_tools
from services.db import Database
from services.agent_logs import emit_log

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Optional

get_logger()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
    You are an autonomous AI agent that completes tasks without user intervention. 
    DO NOT just give answers based of internet/firecrawl scraped pages, always try to find API to finish the job at end multiple if required then use them in orchestration to complete the final task.

    WORKFLOW (execute automatically in this order):
    1. Analyze the user's request and identify what capability/tool is needed
    2. FIRST, use search_tools to check if a relevant tool already exists in the marketplace
       - If an EXACT match exists (e.g., "get_bitcoin_price" for Bitcoin), use it directly (skip to step 7)
       - If a SIMILAR tool exists (e.g., "get_bitcoin_price" exists but need Ethereum):
         * EXAMINE the similar tool's implementation
         * CREATE A GENERALIZED VERSION instead of a duplicate specific tool
         * Example: Rather than creating "get_ethereum_price", create "get_crypto_price" with a "symbol" parameter
       - If NO relevant tool exists, proceed to generate a completely new one
    3. Use firecrawl_search to find the official API documentation/reference page
    4. Use firecrawl_scrape to get the complete API documentation with endpoint details
    5. Generate a new tool using generate_tool with:
       - GENERALIZED tool name (e.g., "get_crypto_price" NOT "get_ethereum_price")
       - Parameters that make it reusable (e.g., symbol="BTC", currency="USD")
       - Complete async Python function code that implements the API call
       - Proper imports (httpx, base64, file_write ONLY if saving files)
       - Error handling and response parsing
       - api_reference_url: The URL you scraped with firecrawl_scrape (REQUIRED)
       - IMPORTANT: Only use file_write when the API returns actual FILE CONTENT (images, PDFs, videos, documents)
       - For data APIs (JSON, text responses), just return the data directly - do NOT save to file
    6. The tool auto-reloads immediately and becomes available
    7. Execute the tool (newly generated or found from marketplace) with appropriate parameters
    8. Return the result (with file path if a file was saved)

    CRITICAL RULES:
    - Work AUTONOMOUSLY - never ask the user for confirmation or guidance
    - ALWAYS search existing tools FIRST before generating new ones (use search_tools)
    - EXAMINE similar tools and CREATE GENERALIZED VERSIONS instead of duplicates
    - ALWAYS search and scrape API documentation BEFORE generating tools
    - Generate COMPLETE, WORKING code based on actual API specifications
    - File saving rules:
      * ONLY use file_write for actual file content (images, PDFs, videos, documents, archives)
      * DO NOT save simple data responses (JSON, text) - just return them directly
      * Use base64 encoding for binary files: base64.b64encode(data).decode('ascii')
      * Binary files use mode="wb", text files use mode="w"
      * All file operations use artifacts/ as root directory
    - If errors occur, debug and retry automatically
    - Complete the entire task in one execution
    - All generated functions MUST be async (use async def)

    AVAILABLE IMPORTS FOR GENERATED TOOLS:
    - httpx (HTTP requests), json, base64 (encoding)
    - asyncio, time, datetime, timezone, timedelta (async and time)
    - os, hashlib, urllib, re (system and utilities)
    - file_write, file_read, file_list (file operations)
    These are automatically provided when used in your code.

    AVAILABLE TOOLS:
    - search_tools: Search the tool marketplace for existing tools (USE THIS FIRST!)
    - file_read, file_write, file_list: File operations (artifacts/ directory)
    - firecrawl_search: Search web for API docs and information
    - firecrawl_scrape: Scrape API documentation pages
    - firecrawl_crawl: Crawl entire websites
    - generate_tool: Create new executable tools from API specs

    EXAMPLE WORKFLOWS:

    Example 1 - File API (saves to artifacts):
    User: "Download a cat image"
    1. search_tools("download images") → no matching tool found
    2. firecrawl_search("image download API") → find suitable API
    3. firecrawl_scrape(api_docs_url) → get endpoint details
    4. generate_tool(download_image) → create tool that downloads and SAVES image file
    5. download_image(url="https://...") → download and save to artifacts/
    6. Return: "Image saved to artifacts/cat_image.jpg"

    Example 2 - Data API (returns data directly):
    User: "Get Ethereum price"
    1. search_tools("ethereum price") → finds "get_bitcoin_price" tool
    2. Examine get_bitcoin_price code → sees it uses CoinGecko API with hardcoded "bitcoin"
    3. firecrawl_scrape(coingecko_docs) → confirm API supports multiple cryptocurrencies
    4. generate_tool("get_crypto_price") → create tool that returns price data (NO file_write)
    5. get_crypto_price(symbol="ethereum") → execute and return JSON data directly
    6. Return: "Ethereum price is $X,XXX.XX"

    Example 3 - Reuse exact match:
    User: "Get Bitcoin price"
    1. search_tools("bitcoin price") → finds "get_crypto_price" tool
    2. get_crypto_price(symbol="bitcoin") → execute directly
    3. Return: "Bitcoin price is $X,XXX.XX"

    Complete the user's request fully and autonomously.
"""


class Result(BaseModel):
    usage: Dict = Field(default={})
    output: str = Field(default="")
    tool_calls: Optional[list] = Field(default=None)


class Agent:
    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        system_prompt: str = SYSTEM_PROMPT,
        firecrawl_api_key: str = FIRECRAWL_API_KEY,
        save_conversations: bool = True,
        db: Optional[Database] = None,
    ):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        self.model = model
        self.system_prompt = system_prompt
        self.save_conversations = save_conversations
        self.db = db  # Database instance for saving conversations

        # Load base tools
        base_schemas, base_functions = get_base_tools(firecrawl_api_key)

        # Load generated tools
        gen_schemas, gen_functions = load_generated_tools()

        # Merge tools
        self.tool_schemas = base_schemas + gen_schemas
        self.tool_functions = {**base_functions, **gen_functions}

        logger.info(
            f"Loaded {len(base_schemas)} base tools and {len(gen_schemas)} generated tools"
        )

        self.messages = None
        self.result = Result()
        self.conversation_id = None
        self.conversation_start_time = None

    async def init_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # Generate conversation ID
        self.conversation_start_time = datetime.now()
        self.conversation_id = self.conversation_start_time.strftime("%Y%m%d_%H%M%S")

    def save_conversation_history(self, final_result: Result):
        """Save the complete conversation history to MongoDB"""
        if not self.save_conversations or not self.db:
            return

        conversation_data = {
            "conversation_id": self.conversation_id,
            "start_time": self.conversation_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "final_output": final_result.output,
            "usage": final_result.usage,
            "tool_calls_made": [
                msg
                for msg in self.messages
                if msg.get("role") == "assistant" and msg.get("tool_calls")
            ],
            "tool_results": [msg for msg in self.messages if msg.get("role") == "tool"],
        }

        conversation_id = self.db.save_conversation(conversation_data)
        logger.info(f"Conversation saved to MongoDB with ID: {conversation_id}")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """
        Execute a tool by name with given arguments.
        Handles both base tools and dynamically generated tools.
        Auto-reloads tools after generate_tool is called.
        """
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        if tool_name not in self.tool_functions:
            return {
                "error": f"Tool '{tool_name}' not found in registry",
                "available_tools": list(self.tool_functions.keys()),
            }

        try:
            func = self.tool_functions[tool_name]

            # Check if function is async
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            logger.info(f"Tool {tool_name} executed successfully")

            # If we just generated a tool, reload to make it immediately available
            if tool_name == "generate_tool" and result.get("success"):
                logger.info("Auto-reloading tools after generate_tool execution")
                self.reload_tools()
                logger.info(
                    f"Tools reloaded. Now have {len(self.tool_schemas)} total tools"
                )

            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "type": type(e).__name__}

    def reload_tools(self, firecrawl_api_key: str = FIRECRAWL_API_KEY):
        """
        Reload all tools (useful after generating new tools).
        """
        # Reload base tools
        base_schemas, base_functions = get_base_tools(firecrawl_api_key)

        # Reload generated tools
        gen_schemas, gen_functions = load_generated_tools()

        # Merge tools
        self.tool_schemas = base_schemas + gen_schemas
        self.tool_functions = {**base_functions, **gen_functions}

        logger.info(
            f"Reloaded {len(base_schemas)} base tools and {len(gen_schemas)} generated tools"
        )

    async def run(self, question: str, max_iterations: int = 25) -> Result:
        if self.messages is None:
            await self.init_messages()

        self.messages.append({"role": "user", "content": question})

        # Emit agent start event
        emit_log(
            "agent",
            f"Starting agent with question: {question[:100]}",
            level="info",
            type="agent_start",
            question=question,
            model=self.model,
            max_iterations=max_iterations,
        )

        # Agent loop - handle tool calls iteratively
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{max_iterations}")
            emit_log(
                "agent",
                f"Iteration {iteration + 1}/{max_iterations}",
                level="info",
                iteration=iteration + 1,
            )

            payload = {
                "model": self.model,
                "messages": self.messages,
                "tools": self.tool_schemas,
                "tool_choice": "auto",
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.debug(f"Sending to LLM with {len(self.messages)} messages")
                response = await client.post(
                    self.url, headers=self.headers, json=payload
                )

                logger.info(f"Status: {response.status_code}")

                if 200 <= response.status_code <= 299:
                    result = response.json()
                    usage = result.get("usage", {})
                    choices = result.get("choices", [])[0]
                    message = choices.get("message", {})

                    output = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])

                    # Add assistant message to history
                    self.messages.append(message)

                    # Emit assistant message if present
                    if output:
                        emit_log(
                            "agent",
                            output,
                            level="info",
                            type="assistant_message",
                            iteration=iteration + 1,
                            content=output,
                        )

                    # If no tool calls, we're done
                    if not tool_calls:
                        logger.info("No tool calls, returning response")
                        emit_log(
                            "agent",
                            "Task completed - no more tool calls",
                            level="info",
                            type="completion",
                        )
                        self.result.output = output or ""
                        self.result.usage = usage
                        self.save_conversation_history(self.result)
                        return self.result

                    # Execute tool calls
                    logger.info(f"Processing {len(tool_calls)} tool call(s)")
                    emit_log(
                        "agent",
                        f"Processing {len(tool_calls)} tool call(s)",
                        level="info",
                        count=len(tool_calls),
                        type="tool_calls_start",
                    )
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_id = tool_call["id"]

                        # Parse arguments safely
                        tool_args_raw = tool_call["function"]["arguments"]
                        try:
                            if isinstance(tool_args_raw, str):
                                # Handle empty string or whitespace
                                if not tool_args_raw.strip():
                                    tool_args = {}
                                else:
                                    tool_args = json.loads(tool_args_raw)
                            else:
                                tool_args = tool_args_raw

                            # Emit tool call event with arguments
                            emit_log(
                                "agent",
                                f"Calling {tool_name} with arguments: {json.dumps(tool_args)[:100]}",
                                level="info",
                                type="tool_call",
                                tool_name=tool_name,
                                tool_id=tool_id,
                                arguments=tool_args,
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse tool arguments for {tool_name}: {e}"
                            )
                            logger.error(f"Raw arguments: {tool_args_raw}")
                            # Return error to LLM
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": json.dumps(
                                        {
                                            "error": f"Invalid JSON in tool arguments: {str(e)}",
                                            "raw_arguments": str(tool_args_raw)[:200],
                                        }
                                    ),
                                }
                            )
                            continue

                        # Execute the tool
                        tool_result = await self.execute_tool(tool_name, tool_args)

                        # Emit tool result event
                        if "error" in tool_result:
                            emit_log(
                                "agent",
                                f"Tool {tool_name} failed: {str(tool_result.get('error', ''))[:100]}",
                                level="error",
                                type="tool_result",
                                tool_name=tool_name,
                                tool_id=tool_id,
                                status="error",
                                error=str(tool_result.get("error", ""))[:200],
                            )
                        else:
                            # Truncate result for display
                            result_str = str(tool_result)
                            result_preview = (
                                result_str[:200] + "..."
                                if len(result_str) > 200
                                else result_str
                            )
                            emit_log(
                                "agent",
                                f"Tool {tool_name} completed successfully",
                                level="info",
                                type="tool_result",
                                tool_name=tool_name,
                                tool_id=tool_id,
                                status="success",
                                result_preview=result_preview,
                            )

                        # Add tool result to messages
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(tool_result),
                            }
                        )

                    # Continue loop to get next response

                else:
                    error_msg = f"Error {response.status_code} with Openrouter call: {response.text}"
                    logger.error(error_msg)
                    emit_log(
                        "agent",
                        f"API error: {error_msg[:200]}",
                        level="error",
                        type="api_error",
                        status_code=response.status_code,
                        error=error_msg[:500],
                    )
                    self.result.output = error_msg
                    self.save_conversation_history(self.result)
                    return self.result

        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        emit_log(
            "agent",
            f"Max iterations ({max_iterations}) reached without completion",
            level="warn",
            type="max_iterations",
            max_iterations=max_iterations,
        )
        self.result.output = "Max iterations reached without final response"
        self.save_conversation_history(self.result)
        return self.result
