# FintelligenceAI Setup Guide

Complete guide to setting up and running the FintelligenceAI system locally and with Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Local)](#quick-start-local)
- [Docker Setup](#docker-setup)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Production Deployment](#production-deployment)

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or WSL2 on Windows
- **Python**: 3.11 or 3.12
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space

### Required Software

1. **Python & Poetry**
```bash
# Install Python 3.11/3.12 if not installed
# Ubuntu/Debian:
sudo apt update && sudo apt install python3.11 python3.11-pip

# macOS (with Homebrew):
brew install python@3.11

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Docker & Docker Compose** (for Docker setup)
```bash
# Ubuntu/Debian:
sudo apt install docker.io docker-compose-plugin

# macOS:
brew install docker docker-compose

# Start Docker daemon
sudo systemctl start docker  # Linux
# OR open Docker Desktop on macOS
```

3. **Optional Tools**
```bash
# For API testing
sudo apt install curl jq  # Linux
brew install curl jq      # macOS

# For load testing
go install github.com/rakyll/hey@latest
```

## Quick Start (Local)

This is the **recommended approach** for development and testing.

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd FintelligenceAI

# Install dependencies with Poetry
poetry install

# Verify installation
poetry run python --version
```

### 2. Environment Configuration
```bash
# Copy environment template
cp env.template .env.local

# Edit configuration (set your API keys)
nano .env.local
```

**Required Environment Variables:**
```bash
# OpenAI API Key (required for AI features)
OPENAI_API_KEY=sk-your-key-here

# Application Settings
APP_ENVIRONMENT=development
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG
API_PORT=8000

# Database URLs (will use existing services)
DATABASE_URL=postgresql+asyncpg://fintelligence_user:fintelligence_pass@localhost:5432/fintelligence_ai
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8100
```

### 3. Start Supporting Services
```bash
# Start only the required databases
docker-compose up -d chromadb

# Check services are running
docker ps
```

### 4. Run the Application
```bash
# Set Python path and run
PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py
```

**Alternative startup methods:**
```bash
# Using uvicorn directly
poetry run uvicorn src.fintelligence_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

# Background process
nohup poetry run python src/fintelligence_ai/api/main.py > app.log 2>&1 &
```

### 5. Verify Installation
```bash
# Health check
curl http://localhost:8000/health

# Agent status
curl http://localhost:8000/agents/status
```

## Docker Setup

Use Docker for production-like environments or if you prefer containerized development.

### 1. Full Docker Stack
```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f fintelligence-api

# Stop services
docker-compose down
```

### 2. Optimized Docker Build (Faster)
```bash
# Use the slim build for faster builds
docker-compose -f docker-compose.slim.yml up -d

# Or use minimal build
docker build -f Dockerfile.minimal -t fintelligence-minimal .
```

### 3. Individual Services
```bash
# Start only databases
docker-compose up -d postgres redis chromadb

# Start just the API
docker-compose up -d fintelligence-api
```

### 4. Production Setup
```bash
# With nginx, monitoring, etc.
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Configuration

### Environment Files

The system supports multiple environment files:

- `.env` - Main environment file
- `.env.local` - Local development overrides
- `.env.production` - Production settings
- `.env.test` - Testing configuration

### Key Configuration Sections

#### 1. AI Model Configuration
```bash
# OpenAI Settings
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

# DSPy Configuration
DSPY_CACHE_DIR=./data/dspy_cache
DSPY_TRAINING_EXAMPLES=100
```

#### 2. Database Configuration
```bash
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/fintelligence_ai

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# ChromaDB Vector Store
CHROMA_HOST=localhost
CHROMA_PORT=8100
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

#### 3. Application Settings
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
API_WORKERS=1

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

#### 4. Agent Configuration
```bash
# Agent Timeouts
RESEARCH_TIMEOUT=30
GENERATION_TIMEOUT=60
VALIDATION_TIMEOUT=30

# Agent Limits
MAX_CONCURRENT_TASKS=5
MAX_TOKENS=4000
```

### Database Initialization

#### PostgreSQL Setup
```bash
# Using Docker
docker run -d \
  --name fintelligence-postgres \
  -e POSTGRES_USER=fintelligence_user \
  -e POSTGRES_PASSWORD=fintelligence_pass \
  -e POSTGRES_DB=fintelligence_ai \
  -p 5432:5432 \
  postgres:15-alpine

# Or use existing PostgreSQL
createdb fintelligence_ai
```

#### ChromaDB Setup
```bash
# Using Docker (recommended)
docker run -d \
  --name fintelligence-chromadb \
  -p 8100:8000 \
  -v chroma_data:/chroma/chroma \
  chromadb/chroma:latest

# Local installation
pip install chromadb
chroma run --host localhost --port 8100
```

## Startup Scripts

### 1. Local Development Script
Create `start_local.sh`:
```bash
#!/bin/bash
set -e

echo "🚀 Starting FintelligenceAI locally..."

# Check prerequisites
command -v poetry >/dev/null 2>&1 || { echo "Poetry not installed"; exit 1; }

# Start supporting services
echo "📦 Starting databases..."
docker-compose up -d chromadb

# Wait for services
echo "⏳ Waiting for services..."
sleep 10

# Start application
echo "🔥 Starting application..."
PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py
```

### 2. Production Script
Create `start_production.sh`:
```bash
#!/bin/bash
set -e

echo "🏭 Starting FintelligenceAI in production mode..."

# Environment checks
if [ ! -f .env ]; then
    echo "❌ .env file not found"
    exit 1
fi

# Start all services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "✅ FintelligenceAI started in production mode"
echo "🌐 Access at: https://your-domain.com"
```

## Troubleshooting

### Common Issues

#### 1. Poetry Installation Issues
```bash
# Clear Poetry cache
poetry cache clear --all pypi

# Reinstall dependencies
rm poetry.lock
poetry install

# Use system Python if needed
poetry env use system
```

#### 2. Port Conflicts
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill process using port
sudo kill -9 $(lsof -t -i:8000)

# Use different port
API_PORT=8001 poetry run python src/fintelligence_ai/api/main.py
```

#### 3. Database Connection Issues
```bash
# Check if databases are running
docker ps | grep -E "(postgres|redis|chroma)"

# Test connections
pg_isready -h localhost -p 5432
redis-cli ping
curl http://localhost:8100/api/v1/heartbeat
```

#### 4. Python Path Issues
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=src:$PYTHONPATH

# Or use absolute paths
python -m src.fintelligence_ai.api.main
```

#### 5. Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Reduce Docker memory if needed
docker-compose down
docker system prune -f
```

### Debug Mode

#### Enable Debug Logging
```bash
# In .env
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG

# Run with verbose output
PYTHONPATH=src poetry run python -v src/fintelligence_ai/api/main.py
```

#### Application Logs
```bash
# Monitor logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f fintelligence-api

# Specific service logs
docker logs fintelligence-chromadb
```

### Performance Optimization

#### 1. Local Development
```bash
# Use fewer workers
API_WORKERS=1

# Reduce cache size
DSPY_CACHE_SIZE=100

# Disable unnecessary features
ENABLE_MONITORING=false
```

#### 2. Production Optimization
```bash
# More workers
API_WORKERS=4

# Enable caching
REDIS_CACHE_TTL=3600

# Production logging
LOG_LEVEL=WARNING
```

## Development

### Development Workflow

1. **Start Development Environment**
```bash
./start_local.sh
```

2. **Code Changes** - The application supports hot reload with `--reload` flag

3. **Run Tests**
```bash
# Unit tests
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# API tests
bash test_api.sh
```

4. **Code Quality**
```bash
# Format code
poetry run black src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/
```

### Adding New Features

1. **Create Feature Branch**
```bash
git checkout -b feature/new-feature
```

2. **Implement Feature** in appropriate module

3. **Add Tests**
```bash
# Create test file
touch tests/unit/test_new_feature.py
```

4. **Update Documentation**
```bash
# Update API_TESTS.md with new endpoints
# Update this SETUP_GUIDE.md if needed
```

## Production Deployment

### 1. Server Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **Network**: Stable internet connection

### 2. Security Setup
```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set up SSL certificates
# Configure firewall rules
# Set up backup procedures
```

### 3. Monitoring
```bash
# Start with monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access monitoring
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### 4. Backup Strategy
```bash
# Database backups
pg_dump fintelligence_ai > backup.sql

# Vector store backups
docker run --rm -v chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma_backup.tar.gz /data

# Code deployment
git clone <repo> /opt/fintelligence-ai
```

---

## Quick Reference

### Essential Commands
```bash
# Start local development
PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py

# Start with Docker
docker-compose up -d

# Health check
curl http://localhost:8000/health

# View logs
docker-compose logs -f fintelligence-api

# Stop everything
docker-compose down
```

### Important Files
- `src/fintelligence_ai/api/main.py` - Main application entry point
- `docker-compose.yml` - Docker services configuration
- `.env` - Environment configuration
- `pyproject.toml` - Python dependencies
- `API_TESTS.md` - API testing guide

### Support
- Check logs first: `docker-compose logs -f`
- Review configuration: `cat .env`
- Test connectivity: `curl http://localhost:8000/health`
- Check ports: `netstat -tlnp | grep 8000`

---

**🎉 You're ready to use FintelligenceAI!**

Start with the [Quick Start (Local)](#quick-start-local) section for the fastest setup, then explore the [API Testing Guide](API_TESTS.md) to try out the features.
