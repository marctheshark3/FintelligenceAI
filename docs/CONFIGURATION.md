# Configuration Guide

> Complete guide to configuring FintelligenceAI for your specific needs

## 📖 Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Model Configuration](#model-configuration)
- [Database Settings](#database-settings)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Local Mode Setup](#local-mode-setup)
- [Environment-Specific Configurations](#environment-specific-configurations)

## Overview

FintelligenceAI uses a hierarchical configuration system:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env, config.yaml)
3. **Default Values** (built-in defaults)

## Environment Variables

### Core Application Settings

```bash
# Application Environment
APP_ENVIRONMENT=development  # development, staging, production
APP_DEBUG=true              # Enable debug mode
APP_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
APP_NAME=FintelligenceAI    # Application name
APP_VERSION=0.1.0           # Application version

# API Configuration
API_HOST=0.0.0.0           # API host binding
API_PORT=8000              # API port
API_PREFIX=/api/v1         # API prefix path
API_DOCS_ENABLED=true      # Enable API documentation
API_CORS_ENABLED=true      # Enable CORS
API_CORS_ORIGINS=*         # CORS allowed origins

# Security
SECRET_KEY=your-secret-key-here  # JWT secret key
JWT_ALGORITHM=HS256             # JWT algorithm
JWT_EXPIRE_MINUTES=1440         # JWT expiration (24 hours)
```

### AI Model Configuration

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_TOKENS=4000

# Perplexity Configuration (optional)
PERPLEXITY_API_KEY=your-perplexity-key
PERPLEXITY_MODEL=llama-3.1-sonar-large-128k-online

# DSPy Configuration
DSPY_CACHE_DIR=./data/dspy_cache
DSPY_CACHE_ENABLED=true
DSPY_LOCAL_MODE=false
DSPY_MAX_RETRIES=3
DSPY_TIMEOUT=30
```

### Database Configuration

```bash
# PostgreSQL Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/fintelligence_ai
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Cache
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=10
REDIS_SOCKET_TIMEOUT=5

# Vector Database (ChromaDB)
CHROMA_HOST=localhost
CHROMA_PORT=8100
CHROMA_COLLECTION_NAME=fintelligence_knowledge
CHROMA_DISTANCE_FUNCTION=cosine
CHROMA_BATCH_SIZE=100
```

### Local Mode Configuration

```bash
# Enable Local Mode
DSPY_LOCAL_MODE=true

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=4000
OLLAMA_KEEP_ALIVE=5m
OLLAMA_TIMEOUT=30
OLLAMA_VERIFY_SSL=false
```

## Configuration Files

### 1. Environment File (.env)

```bash
# Create from template
cp env.template .env

# Environment-specific files
.env.development    # Development settings
.env.staging       # Staging settings
.env.production    # Production settings
.env.local         # Local overrides
```

### 2. YAML Configuration (config/settings.yaml)

```yaml
# config/settings.yaml
app:
  name: "FintelligenceAI"
  version: "0.1.0"
  environment: "development"
  debug: true

api:
  host: "0.0.0.0"
  port: 8000
  prefix: "/api/v1"
  docs_enabled: true
  cors_enabled: true

models:
  openai:
    model: "gpt-4"
    max_tokens: 4000
    temperature: 0.1
    embedding_model: "text-embedding-3-small"

  dspy:
    cache_enabled: true
    local_mode: false
    max_retries: 3

databases:
  postgres:
    pool_size: 10
    max_overflow: 20

  redis:
    db: 0
    max_connections: 10

  chroma:
    collection_name: "fintelligence_knowledge"
    distance_function: "cosine"

logging:
  level: "INFO"
  format: "detailed"
  file_enabled: true
  file_path: "./logs/fintelligence.log"
```

## Model Configuration

### OpenAI Models

```python
# Model configurations
OPENAI_MODELS = {
    'gpt-4': {
        'max_tokens': 8192,
        'context_window': 128000,
        'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06}
    },
    'gpt-3.5-turbo': {
        'max_tokens': 4096,
        'context_window': 16385,
        'cost_per_1k_tokens': {'input': 0.0015, 'output': 0.002}
    }
}
```

### Local Models (Ollama)

```python
# Recommended local models
LOCAL_MODELS = {
    'development': {
        'model': 'llama3.2:1b',
        'embedding': 'nomic-embed-text',
        'memory_req': '2GB'
    },
    'balanced': {
        'model': 'llama3:8b',
        'embedding': 'nomic-embed-text',
        'memory_req': '8GB'
    },
    'quality': {
        'model': 'llama3:70b',
        'embedding': 'nomic-embed-text',
        'memory_req': '40GB'
    }
}
```

## Database Settings

### PostgreSQL Configuration

```python
# Database connection settings
DATABASE_CONFIG = {
    'pool_size': 10,           # Connection pool size
    'max_overflow': 20,        # Max overflow connections
    'pool_timeout': 30,        # Pool timeout in seconds
    'pool_recycle': 3600,      # Connection recycle time
    'echo': False,             # Log SQL queries
    'echo_pool': False,        # Log pool events
    'pool_pre_ping': True,     # Validate connections
}
```

### Redis Configuration

```python
# Redis connection settings
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None,
    'max_connections': 10,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True,
    'health_check_interval': 30
}

# Cache settings
CACHE_CONFIG = {
    'default_ttl': 3600,       # Default TTL in seconds
    'key_prefix': 'fintel:',   # Key prefix
    'serializer': 'json',      # json, pickle, msgpack
    'compression': True,       # Enable compression
}
```

### ChromaDB Configuration

```python
# Vector database settings
CHROMA_CONFIG = {
    'host': 'localhost',
    'port': 8100,
    'collection_name': 'fintelligence_knowledge',
    'distance_function': 'cosine',  # cosine, l2, ip
    'batch_size': 100,
    'max_batch_size': 1000,
    'embedding_function': 'openai',  # openai, ollama
    'persistence_enabled': True,
    'data_path': './data/chroma'
}
```

## Security Configuration

### Authentication & Authorization

```python
# JWT Configuration
JWT_CONFIG = {
    'secret_key': 'your-secret-key',
    'algorithm': 'HS256',
    'expire_minutes': 1440,  # 24 hours
    'refresh_expire_days': 7,
    'issuer': 'fintelligence-ai',
    'audience': 'fintelligence-users'
}

# API Key Configuration
API_KEY_CONFIG = {
    'header_name': 'X-API-Key',
    'query_param': 'api_key',
    'auto_error': True,
    'rate_limit_per_minute': 100
}
```

### CORS Configuration

```python
# CORS settings
CORS_CONFIG = {
    'allow_origins': ['*'],  # Specific origins in production
    'allow_credentials': True,
    'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
    'allow_headers': ['*'],
    'expose_headers': ['X-Total-Count'],
    'max_age': 600
}
```

## Performance Tuning

### Application Performance

```python
# Performance settings
PERFORMANCE_CONFIG = {
    'async_workers': 4,        # Number of async workers
    'thread_pool_size': 10,    # Thread pool size
    'max_request_size': 100,   # Max request size in MB
    'request_timeout': 30,     # Request timeout in seconds
    'keep_alive_timeout': 2,   # Keep-alive timeout
    'graceful_timeout': 30,    # Graceful shutdown timeout
}
```

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
APP_ENVIRONMENT=development
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG
API_DOCS_ENABLED=true
OPENAI_MODEL=gpt-3.5-turbo
DSPY_CACHE_ENABLED=true
DATABASE_ECHO=true
```

### Production Environment

```bash
# .env.production
APP_ENVIRONMENT=production
APP_DEBUG=false
APP_LOG_LEVEL=WARNING
API_DOCS_ENABLED=false
SECRET_KEY=production-secret-key
DATABASE_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=20
RATE_LIMITING_ENABLED=true
CORS_ALLOW_ORIGINS=https://yourdomain.com
```

## Configuration Validation

### Health Checks

```python
# Configuration health monitoring
HEALTH_CHECKS = {
    'database': {
        'enabled': True,
        'timeout': 5,
        'interval': 30
    },
    'redis': {
        'enabled': True,
        'timeout': 2,
        'interval': 15
    },
    'vector_db': {
        'enabled': True,
        'timeout': 5,
        'interval': 60
    },
    'model_apis': {
        'enabled': True,
        'timeout': 10,
        'interval': 300
    }
}
```

## Configuration Best Practices

### 1. Security Best Practices

- Store sensitive data in environment variables
- Use strong secret keys (32+ characters)
- Enable HTTPS in production
- Restrict CORS origins
- Implement rate limiting
- Regular secret rotation

### 2. Performance Best Practices

- Tune database connection pools
- Enable caching where appropriate
- Use async operations
- Monitor memory usage
- Optimize model parameters

### 3. Monitoring Best Practices

- Enable structured logging
- Set up health checks
- Monitor key metrics
- Configure alerts
- Regular backup procedures

## Troubleshooting Configuration

### Common Issues

```bash
# Check current configuration
python -c "
from fintelligence_ai.config import get_settings
settings = get_settings()
print(f'Environment: {settings.app.environment}')
print(f'Debug: {settings.app.debug}')
print(f'API Port: {settings.api.port}')
print(f'Database URL: {settings.database.url}')
"

# Validate configuration
python -m fintelligence_ai.cli config validate

# Test database connection
python -m fintelligence_ai.cli config test-db

# Test model access
python -m fintelligence_ai.cli config test-models
```

## Next Steps

After configuring FintelligenceAI:

1. **[Installation Guide](./INSTALLATION.md)** - Complete the installation
2. **[Local Mode Setup](./LOCAL_MODE_SETUP.md)** - Configure offline operation
3. **[Knowledge Base Guide](./KNOWLEDGE_BASE.md)** - Set up your knowledge base
4. **[Development Guide](./DEVELOPMENT.md)** - Start development
5. **[Troubleshooting Guide](./TROUBLESHOOTING.md)** - Common issues

---

**Configuration complete!** 🎛️ Your FintelligenceAI instance is now properly configured.
