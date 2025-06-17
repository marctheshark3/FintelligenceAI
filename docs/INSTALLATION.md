# Installation Guide

> Complete installation and setup guide for FintelligenceAI

## 📖 Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Environment Setup](#environment-setup)
- [Database Configuration](#database-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9 or higher (3.11+ recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space
- **Docker**: Latest version with Docker Compose

### Recommended Setup
- **CPU**: 4+ cores
- **Memory**: 16GB RAM
- **Storage**: SSD with 50GB+ free space
- **GPU**: NVIDIA GPU for local model acceleration (optional)

## Installation Methods

### Method 1: Poetry (Recommended)

```bash
# 1. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone repository
git clone https://github.com/fintelligence-ai/fintelligence-ai.git
cd FintelligenceAI

# 3. Install dependencies
poetry install --with dev

# 4. Activate environment
poetry shell
```

### Method 2: pip Installation

```bash
# 1. Clone repository
git clone https://github.com/fintelligence-ai/fintelligence-ai.git
cd FintelligenceAI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install package
pip install -e ".[dev]"
```

### Method 3: Docker Only

```bash
# 1. Clone repository
git clone https://github.com/fintelligence-ai/fintelligence-ai.git
cd FintelligenceAI

# 2. Start with Docker
docker-compose up -d

# Application will be available at http://localhost:8000
```

## Environment Setup

### 1. Environment Files

```bash
# Copy template
cp env.template .env

# Edit configuration
nano .env
```

### 2. Required Environment Variables

```bash
# API Keys (at least one required)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key  # Optional
PERPLEXITY_API_KEY=your-perplexity-key  # Optional

# Application Settings
APP_ENVIRONMENT=development
APP_DEBUG=true
APP_LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql+asyncpg://fintelligence_user:fintelligence_pass@localhost:5432/fintelligence_ai
REDIS_URL=redis://localhost:6379/0

# Vector Database
CHROMA_HOST=localhost
CHROMA_PORT=8100
CHROMA_COLLECTION_NAME=fintelligence_knowledge

# DSPy Configuration
DSPY_CACHE_DIR=./data/dspy_cache
DSPY_LOCAL_MODE=false  # Set to true for local-only mode
```

### 3. Local Mode Configuration (Optional)

For completely offline operation:

```bash
# Enable local mode
DSPY_LOCAL_MODE=true

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=4000
```

## Database Configuration

### Option 1: Docker Services (Recommended)

```bash
# Start all required services
docker-compose up -d chromadb redis postgres

# Verify services are running
docker ps
```

### Option 2: Local PostgreSQL

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
brew install postgresql  # macOS

# Create database and user
sudo -u postgres psql
CREATE DATABASE fintelligence_ai;
CREATE USER fintelligence_user WITH PASSWORD 'fintelligence_pass';
GRANT ALL PRIVILEGES ON DATABASE fintelligence_ai TO fintelligence_user;
\q
```

### Option 3: ChromaDB Standalone

```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server
chroma run --host localhost --port 8100 --path ./data/chroma
```

## Installation Verification

### 1. Health Check

```bash
# Start the application
poetry run python -m fintelligence_ai.api.main

# In another terminal, test health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": "connected",
    "vector_db": "connected",
    "cache": "connected"
  }
}
```

### 2. API Documentation

Open http://localhost:8000/docs in your browser to access the interactive API documentation.

### 3. CLI Tools

```bash
# Test CLI
poetry run python -m fintelligence_ai.cli --help

# Run health check
poetry run python -m fintelligence_ai.cli health
```

### 4. Agent Test

```bash
# Test agent functionality
curl -X POST http://localhost:8000/agents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a simple ErgoScript token contract",
    "agent_type": "generation"
  }'
```

## Development Setup

### 1. Development Dependencies

```bash
# Install with development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install
```

### 2. Code Quality Tools

```bash
# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

### 3. Jupyter Environment

```bash
# Start Jupyter Lab
docker-compose up -d jupyter

# Access at http://localhost:8888
# Token: fintelligence-dev-token
```

## Production Deployment

### 1. Production Environment

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or use environment-specific file
cp .env.template .env.production
# Edit production settings
docker-compose --env-file .env.production up -d
```

### 2. SSL Configuration

```bash
# Generate SSL certificates
mkdir -p config/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/ssl/fintelligence.key \
  -out config/ssl/fintelligence.crt

# Update nginx configuration
# See config/nginx/nginx.conf
```

### 3. Monitoring Stack

```bash
# Start with monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access monitoring
# Grafana: http://localhost:3000 (admin/fintelligence-admin)
# Prometheus: http://localhost:9090
```

## Installation Options

### Minimal Installation

For basic functionality without development tools:

```bash
# Use minimal Dockerfile
docker build -f Dockerfile.minimal -t fintelligence-minimal .
docker run -p 8000:8000 fintelligence-minimal
```

### Slim Installation

For faster builds and smaller images:

```bash
# Use slim compose
docker-compose -f docker-compose.slim.yml up -d
```

### GPU-Enabled Installation

For local model acceleration:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Use GPU-enabled compose
docker-compose -f docker-compose.local.yml up -d
```

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check which process is using port 8000
lsof -i :8000

# Use different port
export API_PORT=8080
poetry run python -m fintelligence_ai.api.main
```

#### 2. Docker Issues

```bash
# Reset Docker environment
docker-compose down
docker system prune -f
docker-compose up -d --force-recreate
```

#### 3. Python Path Issues

```bash
# Set Python path explicitly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m fintelligence_ai.api.main
```

#### 4. Permission Issues

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data/
chmod -R 755 ./data/
```

#### 5. Memory Issues

```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory: 8GB+

# Or use swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debug Mode

```bash
# Enable debug logging
export APP_DEBUG=true
export APP_LOG_LEVEL=DEBUG

# Run with verbose output
poetry run python -m fintelligence_ai.api.main --log-level debug
```

### Dependency Issues

```bash
# Clear Poetry cache
poetry cache clear --all .

# Reinstall dependencies
poetry install --with dev --no-cache

# Update dependencies
poetry update
```

### Environment Validation

```bash
# Validate installation
poetry run python -c "
import fintelligence_ai
print('✅ FintelligenceAI imported successfully')
print(f'Version: {fintelligence_ai.__version__}')

try:
    from fintelligence_ai.config import get_settings
    settings = get_settings()
    print('✅ Configuration loaded successfully')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

## Next Steps

After successful installation:

1. **[Quick Start Tutorial](./QUICK_START.md)** - Test basic functionality
2. **[Configuration Guide](./CONFIGURATION.md)** - Customize settings
3. **[Knowledge Base Setup](./KNOWLEDGE_BASE.md)** - Add your content
4. **[Local Mode Setup](./LOCAL_MODE_SETUP.md)** - Enable offline operation
5. **[Development Guide](./DEVELOPMENT.md)** - Start developing

## Support

Need help with installation?

- **Documentation**: [docs.fintelligence.ai](https://docs.fintelligence.ai)
- **GitHub Issues**: [Report installation problems](https://github.com/fintelligence-ai/fintelligence-ai/issues)
- **Discord**: [Join our community](https://discord.gg/fintelligence)
- **Email**: support@fintelligence.ai

---

**Installation complete!** 🎉 You're ready to use FintelligenceAI.
