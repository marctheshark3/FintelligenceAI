# Troubleshooting Guide

> Solutions to common issues and debugging tips for FintelligenceAI

## 📖 Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Local Mode Issues](#local-mode-issues)
- [Database Problems](#database-problems)
- [API Issues](#api-issues)
- [Debug Tools](#debug-tools)

## Quick Diagnostics

### Health Check Command
```bash
# Quick system check
python -m fintelligence_ai.cli health --verbose

# Expected output:
✅ Application: Healthy
✅ Database: Connected
✅ Vector DB: Connected
✅ Cache: Connected
✅ OpenAI API: Accessible
✅ Local Mode: Available (Ollama)
```

### Log Analysis
```bash
# Check recent logs
tail -f ./logs/fintelligence.log

# Search for errors
grep "ERROR\|CRITICAL" ./logs/fintelligence.log | tail -10

# Check Docker logs
docker-compose logs -f fintelligence-api
```

## Installation Issues

### 1. Poetry Installation Fails

**Symptoms:**
```
ERROR: Poetry not found
Command 'poetry' not found
```

**Solutions:**
```bash
# Method 1: Official installer
curl -sSL https://install.python-poetry.org | python3 -

# Method 2: pip installation
pip install poetry

# Method 3: Add to PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### 2. Python Version Issues

**Symptoms:**
```
ERROR: Python 3.9+ required
SyntaxError: invalid syntax
```

**Solutions:**
```bash
# Check Python version
python --version

# Install Python 3.11 (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Use pyenv for version management
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7
```

### 3. Dependency Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver
ModuleNotFoundError: No module named 'xyz'
```

**Solutions:**
```bash
# Clear pip cache
pip cache purge

# Clear Poetry cache
poetry cache clear --all .

# Reinstall dependencies
poetry install --no-cache

# Create fresh environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Configuration Problems

### 1. Missing API Keys

**Symptoms:**
```
ERROR: OpenAI API key not found
AuthenticationError: Invalid API key
```

**Solutions:**
```bash
# Check environment variables
echo $OPENAI_API_KEY

# Set API key temporarily
export OPENAI_API_KEY="sk-your-key-here"

# Add to .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Verify API key works
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 2. Database Connection Issues

**Symptoms:**
```
ERROR: Connection failed
sqlalchemy.exc.OperationalError
psycopg2.OperationalError: connection refused
```

**Solutions:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres
sudo systemctl status postgresql

# Test connection manually
psql -h localhost -p 5432 -U fintelligence_user -d fintelligence_ai

# Reset Docker containers
docker-compose down
docker-compose up -d postgres
docker-compose logs postgres

# Check DATABASE_URL format
echo $DATABASE_URL
# Should be: postgresql+asyncpg://user:pass@host:port/db
```

### 3. Redis Connection Problems

**Symptoms:**
```
redis.exceptions.ConnectionError
ERROR: Redis unavailable
```

**Solutions:**
```bash
# Check Redis status
docker ps | grep redis
redis-cli ping

# Restart Redis
docker-compose restart redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Clear Redis data if corrupted
redis-cli FLUSHALL
```

## Runtime Errors

### 1. Model API Errors

**Symptoms:**
```
openai.RateLimitError: Rate limit exceeded
openai.InvalidRequestError: Invalid model
```

**Solutions:**
```bash
# Check API usage
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Use different model
export OPENAI_MODEL=gpt-3.5-turbo

# Implement exponential backoff
# (built into the system, check logs for retries)

# Switch to local mode temporarily
export DSPY_LOCAL_MODE=true
```

### 2. Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate memory
OOMKilled: Out of memory
```

**Solutions:**
```bash
# Check memory usage
free -h
docker stats

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Reduce batch sizes
export CHROMA_BATCH_SIZE=50
export MODEL_BATCH_SIZE=5

# Enable swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Vector Database Issues

**Symptoms:**
```
ERROR: ChromaDB connection failed
Collection not found
Embedding dimension mismatch
```

**Solutions:**
```bash
# Restart ChromaDB
docker-compose restart chromadb

# Check ChromaDB health
curl http://localhost:8100/api/v1/heartbeat

# Reset ChromaDB data
rm -rf ./data/chroma/*
docker-compose restart chromadb

# Rebuild embeddings
python -m fintelligence_ai.cli knowledge reindex
```

## Performance Issues

### 1. Slow API Responses

**Symptoms:**
```
Timeout after 30 seconds
Slow response times > 10s
```

**Diagnostics:**
```bash
# Check system resources
top
htop
docker stats

# Profile API endpoints
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Check database performance
docker-compose exec postgres psql -U fintelligence_user -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC LIMIT 10;"
```

**Solutions:**
```bash
# Enable caching
export CACHE_ENABLED=true
export CACHE_TTL=3600

# Optimize database connections
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=50

# Use faster embedding model
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Enable async processing
export ASYNC_WORKERS=4
```

### 2. High Memory Usage

**Symptoms:**
```
Memory usage > 90%
Frequent garbage collection
Swap usage increasing
```

**Solutions:**
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# Reduce cache sizes
export VECTOR_CACHE_SIZE=1GB
export MODEL_CACHE_SIZE=2GB

# Enable memory cleanup
export GC_THRESHOLD=0.7
export CACHE_CLEANUP_INTERVAL=300

# Use memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

## Local Mode Issues

### 1. Ollama Connection Problems

**Symptoms:**
```
ERROR: Ollama server not reachable
Connection refused on port 11434
```

**Solutions:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
systemctl start ollama  # Linux
brew services start ollama  # macOS

# Check Ollama status
ollama ps
ollama list

# Restart Ollama
ollama serve &
```

### 2. Model Not Found

**Symptoms:**
```
ERROR: Model 'llama3:latest' not found
ollama: model not available locally
```

**Solutions:**
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3:latest
ollama pull nomic-embed-text

# Check model status
ollama show llama3:latest

# Use different model
export OLLAMA_MODEL=llama3.2:1b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 3. Local Mode Performance Issues

**Symptoms:**
```
Very slow generation times
High CPU usage
Model loading delays
```

**Solutions:**
```bash
# Check system requirements
ollama ps
nvidia-smi  # For GPU users

# Use smaller model
ollama pull llama3.2:1b
export OLLAMA_MODEL=llama3.2:1b

# Optimize Ollama settings
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_NUM_PARALLEL=2

# Enable GPU acceleration (if available)
export OLLAMA_GPU_ENABLED=true
```

## Database Problems

### 1. Migration Issues

**Symptoms:**
```
alembic.util.exc.CommandError
Migration failed
Database schema mismatch
```

**Solutions:**
```bash
# Check migration status
alembic current
alembic history

# Run pending migrations
alembic upgrade head

# Reset database (CAUTION: destroys data)
alembic downgrade base
alembic upgrade head

# Manual migration
docker-compose exec postgres psql -U fintelligence_user -f migrations/reset.sql
```

### 2. Data Corruption

**Symptoms:**
```
IntegrityError: Foreign key violation
Data inconsistency errors
Corrupted index
```

**Solutions:**
```bash
# Backup current data
pg_dump -h localhost -U fintelligence_user fintelligence_ai > backup.sql

# Check database integrity
docker-compose exec postgres psql -U fintelligence_user -c "
VACUUM ANALYZE;
REINDEX DATABASE fintelligence_ai;
"

# Restore from backup
docker-compose exec postgres psql -U fintelligence_user -c "DROP DATABASE fintelligence_ai;"
docker-compose exec postgres psql -U fintelligence_user -c "CREATE DATABASE fintelligence_ai;"
docker-compose exec postgres psql -U fintelligence_user fintelligence_ai < backup.sql
```

## API Issues

### 1. CORS Errors

**Symptoms:**
```
Access-Control-Allow-Origin error
CORS policy blocked
```

**Solutions:**
```bash
# Enable CORS in development
export API_CORS_ENABLED=true
export API_CORS_ORIGINS=*

# Specific origins for production
export API_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Check CORS headers
curl -I -X OPTIONS http://localhost:8000/api/v1/health \
  -H "Origin: http://localhost:3000"
```

### 2. Authentication Issues

**Symptoms:**
```
401 Unauthorized
Invalid token
Token expired
```

**Solutions:**
```bash
# Check token generation
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Verify token
export TOKEN="your-jwt-token"
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/protected

# Reset secret key
export SECRET_KEY=$(openssl rand -hex 32)
```

### 3. Rate Limiting Issues

**Symptoms:**
```
429 Too Many Requests
Rate limit exceeded
```

**Solutions:**
```bash
# Check rate limit headers
curl -I http://localhost:8000/api/v1/health

# Increase rate limits
export RATE_LIMIT_PER_MINUTE=1000
export RATE_LIMIT_PER_HOUR=10000

# Disable rate limiting (development only)
export RATE_LIMITING_ENABLED=false
```

## Debug Tools

### 1. Enable Debug Mode

```bash
# Enable comprehensive debugging
export APP_DEBUG=true
export APP_LOG_LEVEL=DEBUG

# Start with debug flags
python -m fintelligence_ai.api.main --debug --reload
```

### 2. Log Analysis Tools

```bash
# Real-time log monitoring
tail -f ./logs/fintelligence.log | grep ERROR

# Structured log analysis
jq '.level == "ERROR"' ./logs/fintelligence.jsonl

# Performance analysis
grep "duration" ./logs/fintelligence.log | awk '{print $NF}' | sort -n
```

### 3. System Monitoring

```bash
# Resource monitoring
watch -n 1 'docker stats --no-stream'

# Network monitoring
netstat -tulpn | grep :8000

# Process monitoring
ps aux | grep fintelligence
```

### 4. Database Debugging

```bash
# Query performance
docker-compose exec postgres psql -U fintelligence_user -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
WHERE total_time > 1000
ORDER BY total_time DESC;"

# Connection monitoring
docker-compose exec postgres psql -U fintelligence_user -c "
SELECT pid, usename, application_name, client_addr, state
FROM pg_stat_activity
WHERE datname = 'fintelligence_ai';"
```

## Emergency Recovery

### 1. Complete Reset

```bash
# Stop all services
docker-compose down

# Remove all data (CAUTION)
rm -rf ./data/
mkdir -p ./data/{chroma,dspy_cache,uploads,backups}

# Reset containers
docker-compose down -v
docker system prune -f

# Restart fresh
docker-compose up -d
```

### 2. Backup and Restore

```bash
# Create backup
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh backup-2025-01-15.tar.gz
```

### 3. Rollback to Previous Version

```bash
# Check git history
git log --oneline -10

# Rollback to previous commit
git checkout HEAD~1

# Rebuild and restart
docker-compose build
docker-compose up -d
```

## Getting Help

### 1. Collect Debug Information

```bash
# Generate system report
python -m fintelligence_ai.cli debug-report > debug-report.txt
```

### 2. Community Support

- **GitHub Issues**: [Report bugs](https://github.com/fintelligence-ai/fintelligence-ai/issues)
- **Discord**: [Community chat](https://discord.gg/fintelligence)
- **Discussions**: [GitHub Discussions](https://github.com/fintelligence-ai/fintelligence-ai/discussions)

### 3. Professional Support

- **Email**: support@fintelligence.ai
- **Documentation**: [docs.fintelligence.ai](https://docs.fintelligence.ai)
- **Enterprise Support**: Available for production deployments

---

**Still having issues?** Please include the debug report and specific error messages when asking for help.
