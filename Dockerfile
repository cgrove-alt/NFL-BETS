FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml README.md ./

# Create package structure for editable install
COPY nfl_bets/ ./nfl_bets/
COPY api/ ./api/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi uvicorn[standard]

# Create directory for model storage
RUN mkdir -p /app/nfl_bets/models/saved

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose port (Railway will set $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start API server
# Railway sets $PORT, so we use shell form to expand it
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
