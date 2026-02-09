# YC Hack 2 - Serverless MCP Marketplace

## Overview
An autonomous AI agent that can dynamically generate and execute tools by searching API documentation, creating Python code, and running it on-demand. Built for hackathons - no database setup required, works out of the box with just API keys.

## Key Features
- 🤖 **Fully Autonomous**: Give it a task, it completes it end-to-end without user intervention
- 🔧 **Dynamic Tool Generation**: Searches for API docs, generates working Python code, and executes it
- 📁 **File Operations**: Can read, write, and manage files in the `artifacts/` directory
- 🔄 **Auto-Reload**: Generated tools are immediately available in the same conversation
- 💾 **MongoDB Storage**: All conversations and tools stored in MongoDB for persistence
- 🌐 **Web Integration**: Built-in web search, scraping, and crawling via Firecrawl

## Quick Start

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key ([get one here](https://openrouter.ai/keys))
- Firecrawl API key ([get one here](https://firecrawl.dev))

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd yc-hack2

# Install dependencies
uv sync
```

### 3. Configuration

Create a `dev.env` file in the project root:

```bash
cp .env.example dev.env
```

Edit `dev.env` and add your API keys:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Optional - for semantic tool search (app works without it)
VOYAGE_API_KEY=your_voyage_api_key_here

# Optional - defaults to localhost:27017 if not set
MONGODB_URI=mongodb://localhost:27017/agent_db
```

### 4. Run the Agent

Pass your question as a command-line argument:

```bash
uv run main.py "Download an image of a cat and save it to the cat directory"
```

Or try other examples:

```bash
# Get real-time data
uv run main.py "Get me the real-time stream flow data for the Mississippi River"

# Fetch weather data
uv run main.py "Get current weather for Austin, TX and save it to a file"

# Download and process content
uv run main.py "Scrape the latest news from HackerNews and summarize it"
```

The agent will autonomously:
1. Search for relevant API documentation
2. Scrape the API specs
3. Generate a Python tool with working code
4. Execute the tool to complete your request
5. Save results to the `artifacts/` directory

## MongoDB Storage

The application uses MongoDB to store:
- **Conversations**: Complete conversation history with all messages, tool calls, and results
- **Tools**: Generated tool definitions with executable Python code and vector embeddings

### Vector Search for Tools

The system uses **vector embeddings** to intelligently search through the tool marketplace. Instead of showing the agent all 1000+ tools (which would confuse it), it uses semantic search to find the top 10 most relevant tools for each task.

**How it works:**
1. When a tool is saved, the system generates a vector embedding from its name and description
2. Embeddings are generated using Voyage AI's `voyage-4` model (1024 dimensions)
3. When searching, the query is embedded and compared using cosine similarity
4. The top 10 most semantically similar tools are returned

**Benefits:**
- Agent sees only relevant tools, reducing confusion
- Faster tool discovery with semantic understanding
- Scales to thousands of tools without performance issues
- More accurate than keyword matching

### Local Development with MongoDB

If running locally (without Docker), you'll need MongoDB installed:

```bash
# Install MongoDB (macOS)
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Set MongoDB URI in dev.env
MONGODB_URI=mongodb://localhost:27017/agent_db
```

**Note**: When using Docker Compose, MongoDB is automatically set up and configured.

### MongoDB Atlas Vector Search (Optional)

For production deployments, you can use MongoDB Atlas with native vector search:

1. Create a free MongoDB Atlas cluster at https://www.mongodb.com/cloud/atlas
2. Create a vector search index on the `tools` collection:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 1024,
         "similarity": "cosine"
       }
     ]
   }
   ```
3. Update `MONGODB_URI` in your `dev.env` to use Atlas connection string

With Atlas Vector Search, queries will use MongoDB's optimized Hierarchical Navigable Small Worlds algorithm for even faster similarity search.

## Architecture

### Core Components

1. **LLM Agent** (`services/llm.py`)
   - OpenRouter-powered (Claude Haiku 4.5 by default)
   - Autonomous workflow: search → scrape → generate → execute
   - Multi-turn tool calling with auto-reload
   - Conversation logging and error recovery

2. **Base Tools** (`services/tools.py`)
   - **file_read/write/list**: File operations in `artifacts/` directory
   - **firecrawl_search**: Search the web for API docs
   - **firecrawl_scrape**: Extract content from webpages
   - **firecrawl_crawl**: Crawl entire websites
   - **generate_tool**: Create executable Python tools

3. **Tool Execution**
   - Generated tools stored in MongoDB
   - Code executed using Python's `exec()` with base64 support
   - Auto-reload makes new tools immediately available
   - Supports both text and binary file operations

4. **Storage**
   - **MongoDB**: Conversations and tool definitions
   - `artifacts/` - Agent's workspace for all file operations
   - `logs/` - Application logs

## How It Works

### Example: "Download an image of a cat"

```
1. Agent searches for image APIs
   → firecrawl_search("cat image API download")

2. Agent scrapes API documentation
   → firecrawl_scrape("https://thecatapi.com/docs")

3. Agent generates a download tool
   → generate_tool(name="download_cat_image", code="...", ...)
   → Tool auto-reloads, immediately available

4. Agent executes the new tool
   → download_cat_image(output_path="cat/image.jpg")

5. Agent saves the result
   → file_write("cat/image.jpg", base64_data, mode="wb")

6. Done! Image saved to artifacts/cat/image.jpg
```

All of this happens **autonomously** in one execution, no user intervention required.

## API Server

### Run as a REST API

Start the FastAPI server:

```bash
uv run server.py
```

The API will be available at `http://localhost:8001` (default)

**Interactive API docs:** http://localhost:8001/docs

**Change port:** Set the `PORT` environment variable:
```bash
PORT=9000 uv run server.py
```

### API Endpoints

#### POST /agent
Full-featured endpoint with all options:

```bash
curl -X POST http://localhost:8001/agent \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Get the current Bitcoin price",
    "max_iterations": 25,
    "model": "anthropic/claude-haiku-4.5",
    "save_conversation": true
  }'
```

#### POST /agent/simple
Simplified endpoint with query parameters:

```bash
curl -X POST "http://localhost:8001/agent/simple?prompt=Get%20Bitcoin%20price&max_iterations=10"
```

#### GET /health
Health check:

```bash
curl http://localhost:8001/health
```

### Example Response

```json
{
  "success": true,
  "output": "The current Bitcoin price is $45,234.56 USD...",
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "error": null
}
```

## MCP Server (Local Access)

Access the Universal Adapter via MCP (Model Context Protocol) for use with Claude Desktop and other local MCP clients.

### Architecture

**Two access methods:**
- **Local (MCP):** `mcp_server.py` - stdio transport for Claude Desktop
- **Remote (HTTP):** `server.py` - FastAPI REST API for web access

Both use the same Agent with all marketplace tools.

### Claude Desktop Setup

The MCP server exposes your agent as a single `chat` tool. The agent internally has access to all marketplace tools.

**1. Configure Claude Desktop:**

Add to your config file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "universal-adapter": {
      "command": "uv",
      "args": ["run", "mcp_server.py"],
      "cwd": "/path/to/yc-hack2",
      "env": {
        "OPENROUTER_API_KEY": "your-key",
        "FIRECRAWL_API_KEY": "your-key",
        "MONGODB_URI": "your-mongodb-uri",
        "VOYAGE_API_KEY": "your-key"
      }
    }
  }
}
```

**2. Restart Claude Desktop** - The Universal Adapter tools will appear!

### Available MCP Tools

- **`chat`** - Chat with the agent (primary tool)
- **`list_marketplace_tools`** - Browse marketplace
- **`health`** - Health check

### How It Works

```
Claude Desktop → MCP (stdio) → chat(message) → Agent → Tools → Response
```

The agent handles all tool orchestration internally (search, generate, execute).

**For remote access via HTTP,** use `server.py` instead (see API Server section).

### Documentation

See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for detailed setup instructions and troubleshooting.

## Tool Marketplace API

The server includes endpoints for browsing and searching the tool marketplace.

### List All Tools

```bash
# Get all tools with pagination
curl http://localhost:8001/tools?limit=50&skip=0

# Response
[
  {
    "name": "get_bitcoin_price",
    "description": "Fetches the current Bitcoin price from CoinGecko API...",
    "parameters": {...},
    "code": "async def get_bitcoin_price()...",
    "created_at": "2026-01-31T08:52:52.949000"
  }
]
```

### Search Tools (Vector Similarity)

```bash
# Search for tools using semantic search
curl "http://localhost:8001/tools/search?q=cryptocurrency%20price&limit=10"

# Response
{
  "query": "cryptocurrency price",
  "count": 1,
  "tools": [
    {
      "name": "get_bitcoin_price",
      "description": "Fetches the current Bitcoin price...",
      "similarity_score": 0.485,
      "parameters": {...}
    }
  ]
}
```

### Get Specific Tool

```bash
# Get complete tool definition
curl http://localhost:8001/tools/get_bitcoin_price

# Response includes full code
{
  "name": "get_bitcoin_price",
  "description": "...",
  "parameters": {...},
  "code": "import httpx\n\nasync def get_bitcoin_price()..."
}
```

### Delete Tool

```bash
# Remove a tool from marketplace
curl -X DELETE http://localhost:8001/tools/get_bitcoin_price

# Response
{
  "success": true,
  "message": "Tool 'get_bitcoin_price' deleted successfully"
}
```

### List Conversations

```bash
# Get recent conversations
curl http://localhost:8001/conversations?limit=20&skip=0

# Response
[
  {
    "id": "697dc2e45e09a3756310636a",
    "conversation_id": "20260131_085206",
    "start_time": "2026-01-31T08:52:06.862134",
    "model": "anthropic/claude-haiku-4.5",
    "final_output": "I successfully fetched the current Bitcoin price..."
  }
]
```

### Get Specific Conversation

```bash
# Get full conversation with all messages
curl http://localhost:8001/conversations/697dc2e45e09a3756310636a

# Returns complete conversation including all tool calls and results
```

## Advanced Usage

### Using the API Module

Import and use the agent programmatically in your Python scripts:

```python
from services.api import ask, run_agent_sync, run_agent

# Quick usage - returns string output
output = ask("Get the current Bitcoin price")
print(output)

# Full control - returns structured response
result = run_agent_sync(
    prompt="Download a cat image",
    max_iterations=20,
    model="anthropic/claude-haiku-4.5"
)

if result.success:
    print(result.output)
    print(f"Tokens used: {result.usage}")
else:
    print(f"Error: {result.error}")

# Async usage
import asyncio
result = await run_agent("Your question here")
```

### Running Different Tasks via CLI

Simply pass different questions as command-line arguments:

```bash
# API data retrieval
uv run main.py "Fetch Bitcoin price from CoinGecko API"

# Web scraping and analysis
uv run main.py "Find and download the top 5 Python repos on GitHub"

# Data processing
uv run main.py "Get COVID-19 statistics and create a summary report"

# Image/file operations
uv run main.py "Download the Eiffel Tower image and save it"
```

The agent will automatically:
1. Search for relevant APIs
2. Generate tools to interact with them
3. Execute the tools
4. Save results to `artifacts/`

### Modify Behavior

To change max iterations or other settings, edit `main.py`:

```python
# Increase iterations for complex tasks
res = await agent.run(question, max_iterations=50)

# Change the model
agent = Agent(model="anthropic/claude-opus-4.5")

# Disable conversation logging
agent = Agent(save_conversations=False)
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key from [OpenRouter.ai](https://openrouter.ai/keys) |
| `FIRECRAWL_API_KEY` | Yes | API key from [Firecrawl.dev](https://firecrawl.dev) |
| `VOYAGE_API_KEY` | Optional | API key from [Voyage AI](https://dash.voyageai.com/) for semantic tool search embeddings. App works without it (falls back to zero-vector). |
| `MONGODB_URI` | Optional | MongoDB connection string. Defaults to `mongodb://localhost:27017/agent_db` if not set. Required for production deployments. |
| `MCP_PORT` | Optional | Port for MCP server (default: 8002) |
| `MCP_HOST` | Optional | Host for MCP server (default: 0.0.0.0) |

### Agent Configuration

Customize the agent in `services/llm.py`:

```python
agent = Agent(
    model="anthropic/claude-haiku-4.5",  # Model to use
    save_conversations=True,              # Save conversation logs
    firecrawl_api_key="...",             # Optional: override env var
)
```

## Project Structure

```
yc-hack2/
├── main.py                      # CLI entry point
├── server.py                    # FastAPI server
├── docker-compose.yml           # Docker setup with MongoDB
├── Dockerfile                   # Container image
├── dev.env                      # Your API keys (gitignored)
├── .env.example                 # Template for dev.env
├── services/
│   ├── api.py                  # API interface module
│   ├── llm.py                  # Agent with autonomous workflow
│   ├── tools.py                # Base tools + tool execution
│   ├── db.py                   # MongoDB client and operations
│   ├── env.py                  # Environment variables
│   └── logging.py              # Logging configuration
├── artifacts/                   # Agent's file workspace (gitignored)
├── logs/                       # Application logs (gitignored)
└── understand/                 # Reference documentation (temporary)
```

## Troubleshooting

### "No module named 'services'"
Run `uv sync` to install dependencies.

### "OPENROUTER_API_KEY not found"
Create `dev.env` file with your API keys (see `.env.example`).

### Firecrawl Timeout Errors
Normal for large documentation pages. The agent handles these gracefully and continues with alternate URLs.

### Agent Hits Max Iterations
Increase `max_iterations` in `main.py` for complex tasks.

## Documentation

The `understand/` directory contains reference documentation:
- **architecture.md** - System architecture and components
- **data-models.md** - Data structures and schemas
- **flows.md** - Workflow diagrams and sequences
- **implementation-plan.md** - Development roadmap
- **prompts-reference.md** - System prompts and examples

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t autonomous-agent .

# Run with environment variables
docker run -d \
  -p 8001:8001 \
  -e OPENROUTER_API_KEY=your_key \
  -e FIRECRAWL_API_KEY=your_key \
  --name agent \
  autonomous-agent

# Check logs
docker logs -f agent

# Stop
docker stop agent
```

### Using Docker Compose

1. Create a `dev.env` file with your API keys (see `.env.example`)

2. Start the service:
```bash
docker-compose up -d
```

3. View logs:
```bash
docker-compose logs -f
```

4. Stop the service:
```bash
docker-compose down
```

The agent will be available at `http://localhost:8001`

### Docker Features

- ✅ MongoDB database included
- ✅ Health checks configured
- ✅ Auto-restart on failure
- ✅ Persistent volumes for artifacts and MongoDB data
- ✅ Optimized layer caching
- ✅ Non-blocking unbuffered output

### Deploy to Railway

1. **Connect** your repo to Railway and use the Dockerfile (or Nixpacks; a `Procfile` is included).
2. **Set environment variables** in the Railway dashboard (no `dev.env` in deploy):
   - `OPENROUTER_API_KEY` (required for chat)
   - `FIRECRAWL_API_KEY` (required for web search/scrape)
   - `MONGODB_URI` (required; use [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) or Railway MongoDB)
   - `VOYAGE_API_KEY` (optional; for semantic tool search; app runs without it)
3. **PORT** is set by Railway; the app binds to `0.0.0.0:$PORT`.
4. **Health:** Railway can probe `/` or `/health`; both return 200 when the app is up.

If you see "Application failed to respond", check deploy logs for startup errors (e.g. missing `MONGODB_URI` or invalid keys) and ensure all required env vars are set.

## Contributing

This is a hackathon project. Feel free to extend it with:
- MongoDB integration for tool storage
- Enhanced sandboxing for code execution
- Tool versioning and marketplace features
- Multi-agent collaboration
- Web UI for agent interaction

## License

MIT
