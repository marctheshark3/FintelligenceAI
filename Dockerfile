# FintelligenceAI Dockerfile
# Multi-stage build for development and production

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==$POETRY_VERSION
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml ./

# Development stage
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry install --with dev

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/notebooks /app/config \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories with proper permissions
RUN mkdir -p /app/data/{chroma,uploads,dspy_cache,experiments} \
    && chown -R appuser:appuser /app/data

USER appuser

# Expose ports
EXPOSE 8000 8888

# Default command for development
CMD ["uvicorn", "fintelligence_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies
RUN poetry install --only main,production

# Create necessary directories
RUN mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser README.md ./

# Create data directories with proper permissions
RUN mkdir -p /app/data/{chroma,uploads,dspy_cache,experiments} \
    && chown -R appuser:appuser /app/data

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "fintelligence_ai.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
