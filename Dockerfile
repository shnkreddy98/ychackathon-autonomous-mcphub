# Use Python 3.13 slim image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-install-project

# Copy rest of the application
COPY . .

# Create directories that will be used at runtime
RUN mkdir -p artifacts generated_tools conversations logs

# Expose port
EXPOSE 8001

# Environment variables (can be overridden)
ENV PORT=8001
ENV PYTHONUNBUFFERED=1

# Health check (use PORT so Railway's dynamic port works)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8001}/health || exit 1'

# Run the server (reads PORT from env; Railway sets PORT)
CMD ["uv", "run", "server.py"]
