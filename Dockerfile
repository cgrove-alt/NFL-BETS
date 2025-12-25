FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
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

# Copy bundled models for initial seeding
COPY models/trained/ ./models/bundled/

# Create directory for model storage (will be Railway Volume mount point)
RUN mkdir -p /app/models/trained

# Copy startup script
COPY scripts/startup.sh ./scripts/
RUN chmod +x scripts/startup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose port (Railway will set $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start via startup script (seeds models to volume if needed)
CMD ["./scripts/startup.sh"]
