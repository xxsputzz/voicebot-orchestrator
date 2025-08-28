# Dockerfile for Voicebot Orchestrator Core
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    ORCHESTRATOR_HOST=0.0.0.0 \
    ORCHESTRATOR_PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directories
RUN mkdir -p /app /app/logs /app/cache /app/adapters /app/exports \
    && chown -R appuser:appuser /app

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser voicebot_orchestrator/ ./voicebot_orchestrator/
COPY --chown=appuser:appuser pyproject.toml ./

# Create configuration template
RUN echo '{\
  "orchestrator": {\
    "host": "0.0.0.0",\
    "port": 8000,\
    "workers": 4,\
    "timeout": 300\
  },\
  "microservices": {\
    "stt_service": "http://stt-service:8001",\
    "llm_service": "http://llm-service:8002",\
    "tts_service": "http://tts-service:8003",\
    "cache_service": "http://cache-service:8004",\
    "analytics_service": "http://analytics-service:8005"\
  },\
  "cache": {\
    "redis_url": "redis://redis:6379",\
    "similarity_threshold": 0.20,\
    "max_cache_size": 10000\
  }\
}' > /app/config.json && chown appuser:appuser /app/config.json

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${ORCHESTRATOR_PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Labels for metadata
LABEL maintainer="Orkestra Team <team@orkestra.ai>" \
      version="1.0.0" \
      description="Voicebot Orchestrator Core Microservice" \
      org.opencontainers.image.title="voicebot-orchestrator-core" \
      org.opencontainers.image.description="Central coordination service for voicebot operations" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Orkestra" \
      org.opencontainers.image.licenses="MIT"

# Start command
CMD ["python", "-m", "voicebot_orchestrator.microservices.orchestrator_core"]
