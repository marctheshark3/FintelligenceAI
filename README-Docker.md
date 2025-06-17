# FintelligenceAI Docker Setup Guide

This guide provides comprehensive instructions for running FintelligenceAI using Docker Compose.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available for Docker
- 10GB+ free disk space

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd FintelligenceAI

# Option A: Use the interactive setup script (recommended)
./scripts/docker-setup.sh

# Option B: Manual setup
cp env.template .env
nano .env  # Edit environment variables
```

### 2. Configure Environment Variables

**Critical configurations to update in `.env`:**

```bash
# API Keys (Required for AI functionality)
OPENAI_API_KEY=your_actual_openai_api_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here  # Optional fallback

# Security (Change these!)
POSTGRES_PASSWORD=your_secure_database_password
JWT_SECRET_KEY=your_very_secure_jwt_secret_key_here
GRAFANA_PASSWORD=your_secure_grafana_password

# Optional: GitHub token for knowledge base ingestion
GITHUB_TOKEN=your_github_token_here
```

**Note**: The `.env` file works for both local development and Docker deployment. Docker-specific settings like host mappings are automatically handled by the Docker Compose configuration.

### 3. Start the Stack

```bash
# Development mode (with hot reloading)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Basic mode (minimal services)
docker-compose up -d
```

### 4. Verify Installation

```bash
# Check all services are running
docker-compose ps

# View logs
docker-compose logs -f fintelligence-api
docker-compose logs -f fintelligence-ui

# Health checks
curl http://localhost:8000/health  # API health
curl http://localhost:3000/health  # UI health
```

## 🏗️ Architecture Overview

### Core Services

| Service | Port | Description |
|---------|------|-------------|
| `fintelligence-api` | 8000 | FastAPI backend with DSPy and RAG pipeline |
| `fintelligence-ui` | 3000 | Nginx-served web interface |
| `postgres` | 5432 | PostgreSQL database for metadata |
| `redis` | 6379 | Redis cache for sessions and temporary data |
| `chromadb` | 8100 | Vector database for embeddings |

### Monitoring Services

| Service | Port | Description |
|---------|------|-------------|
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3001 | Dashboards and visualization |
| `nginx` | 80/443 | Reverse proxy (production) |

### Optional Services

| Service | Port | Description | Profile |
|---------|------|-------------|---------|
| `ollama` | 11434 | Local LLM server | `local-ai` |
| `adminer` | 8080 | Database admin interface | `dev` |
| `redis-commander` | 8081 | Redis admin interface | `dev` |

## 🛠️ Development Setup

### Enable Development Mode

```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access development tools
open http://localhost:8080  # Database Admin (Adminer)
open http://localhost:8081  # Redis Commander
```

### Development Features

- **Hot Reloading**: Code changes automatically restart the API server
- **Direct Database Access**: PostgreSQL exposed on port 5432
- **Admin Interfaces**: Easy access to database and Redis
- **Debug Logging**: Verbose logging enabled

### Code Development Workflow

```bash
# Edit code in src/ directory - changes auto-reload
# Edit UI in ui/ directory - changes visible immediately
# View logs in real-time
docker-compose logs -f fintelligence-api

# Run tests
docker-compose exec fintelligence-api pytest

# Access database directly
docker-compose exec postgres psql -U fintelligence_user -d fintelligence_ai_dev

# Access Redis CLI
docker-compose exec redis redis-cli
```

## 🚀 Production Deployment

### Production Checklist

1. **Security Configuration**:
   ```bash
   # Update these in .env for production
   APP_ENVIRONMENT=production
   APP_DEBUG=false
   SECURE_COOKIES=true
   HTTPS_ONLY=true

   # Use strong passwords
   POSTGRES_PASSWORD=complex_secure_password
   JWT_SECRET_KEY=long_random_secret_key
   GRAFANA_PASSWORD=secure_grafana_password
   ```

2. **SSL Certificates** (if using HTTPS):
   ```bash
   # Place SSL certificates in config/ssl/
   cp your-cert.pem config/ssl/cert.pem
   cp your-key.pem config/ssl/key.pem
   ```

3. **Resource Limits**: Review and adjust resource limits in `docker-compose.prod.yml`

4. **Backup Strategy**: Configure automated backups (see Backup section)

### Production Deployment

```bash
# Deploy production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs

# Scale services if needed
docker-compose up --scale fintelligence-api=3 -d
```

## 🌐 Access Points

### Main Applications

- **Web UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

### Monitoring & Admin

- **Grafana Dashboard**: http://localhost:3001 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **Database Admin**: http://localhost:8080 (dev mode)
- **Redis Admin**: http://localhost:8081 (dev mode)

### Development Tools

- **Jupyter Lab**: http://localhost:8888 (dev mode)
- **API Direct Access**: http://localhost:8000

## ⚙️ Configuration

### Service Profiles

Control which services run using Docker Compose profiles:

```bash
# Run with local AI (includes Ollama)
docker-compose --profile local-ai up -d

# Run with production proxy
docker-compose --profile production up -d

# Run minimal stack
docker-compose up -d
```

### Environment Variable Reference

See `docker.env` for complete environment variable documentation.

**Key Categories**:
- **App Settings**: Debug, logging, ports
- **AI Models**: OpenAI, Anthropic, Ollama configuration
- **Databases**: PostgreSQL, Redis, ChromaDB settings
- **Security**: JWT, CORS, rate limiting
- **Monitoring**: Prometheus, Grafana, logging

### Volume Mapping

**Persistent Data**:
- `postgres_data` → Database storage
- `redis_data` → Cache storage
- `chroma_data` → Vector embeddings
- `prometheus_data` → Metrics history
- `grafana_data` → Dashboard configs

**Development Mounts**:
- `./src` → Live code reloading
- `./ui` → UI development
- `./data` → Local data persistence
- `./logs` → Application logs

## 🔧 Common Operations

### Service Management

```bash
# Start specific service
docker-compose up -d fintelligence-api

# Restart service
docker-compose restart fintelligence-api

# View service logs
docker-compose logs -f fintelligence-api

# Scale service
docker-compose up --scale fintelligence-api=3 -d

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ destroys data)
docker-compose down -v
```

### Database Operations

```bash
# Database backup
docker-compose exec postgres pg_dump -U fintelligence_user fintelligence_ai > backup.sql

# Database restore
docker-compose exec -T postgres psql -U fintelligence_user fintelligence_ai < backup.sql

# Connect to database
docker-compose exec postgres psql -U fintelligence_user -d fintelligence_ai

# Reset database (⚠️ destroys data)
docker-compose down postgres
docker volume rm fintelligenceai_postgres_data
docker-compose up -d postgres
```

### Monitoring and Debugging

```bash
# View all service status
docker-compose ps

# Check resource usage
docker stats

# Debug specific service
docker-compose exec fintelligence-api bash

# View API logs with filtering
docker-compose logs fintelligence-api | grep ERROR

# Monitor health checks
watch docker-compose ps
```

## 🗄️ Backup and Recovery

### Automated Backups (Production)

The production configuration includes automated database backups:

```bash
# Check backup service logs
docker-compose logs postgres-backup

# List backups
ls -la data/backups/

# Manual backup
docker-compose exec postgres-backup /backup.sh
```

### Manual Backup Procedures

```bash
# Full data backup
mkdir -p backups/$(date +%Y%m%d)

# Database backup
docker-compose exec postgres pg_dump -U fintelligence_user fintelligence_ai > \
  backups/$(date +%Y%m%d)/database.sql

# Vector database backup
docker-compose exec chromadb tar czf - /chroma/chroma > \
  backups/$(date +%Y%m%d)/chromadb.tar.gz

# Application data backup
tar czf backups/$(date +%Y%m%d)/app_data.tar.gz data/ logs/
```

### Recovery Procedures

```bash
# Restore database
docker-compose exec -T postgres psql -U fintelligence_user fintelligence_ai < \
  backups/20241201/database.sql

# Restore vector database (requires service restart)
docker-compose down chromadb
docker run --rm -v fintelligenceai_chroma_data:/chroma ubuntu \
  tar xzf - -C /chroma < backups/20241201/chromadb.tar.gz
docker-compose up -d chromadb
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Conflicts**:
   ```bash
   # Change ports in .env
   API_PORT=8001
   UI_PORT=3001
   ```

2. **Permission Issues**:
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER data/ logs/
   ```

3. **Memory Issues**:
   ```bash
   # Check Docker memory allocation
   docker system df
   docker system prune
   ```

4. **API Key Issues**:
   ```bash
   # Verify API keys are set
   docker-compose exec fintelligence-api env | grep API_KEY
   ```

### Debug Mode

```bash
# Enable debug mode
echo "APP_DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Restart with debug logging
docker-compose restart fintelligence-api

# View debug logs
docker-compose logs -f fintelligence-api | grep DEBUG
```

### Health Check Status

```bash
# Check all health statuses
docker-compose ps --format "table {{.Name}}\t{{.Status}}"

# Detailed health check
for service in fintelligence-api fintelligence-ui postgres redis chromadb; do
  echo "=== $service ==="
  docker-compose exec $service curl -f http://localhost:8000/health 2>/dev/null || echo "Health check failed"
done
```

## 📊 Performance Tuning

### Resource Allocation

Adjust resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # Increase for better performance
      memory: 4G       # Adjust based on your system
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Database Optimization

```bash
# PostgreSQL tuning (add to postgres environment)
- POSTGRES_SHARED_BUFFERS=256MB
- POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
- POSTGRES_WORK_MEM=4MB
```

### API Scaling

```bash
# Scale API instances
docker-compose up --scale fintelligence-api=3 -d

# Use nginx load balancer
docker-compose --profile production up -d
```

## 🆘 Getting Help

1. **Check Logs**: Always start with `docker-compose logs`
2. **Verify Environment**: Ensure all required variables are set in `.env`
3. **Resource Check**: Verify Docker has sufficient resources
4. **Network Issues**: Check if ports are available and not blocked
5. **API Keys**: Verify all API keys are valid and have sufficient quota

For additional support, check the main project documentation or create an issue in the repository.

---

## 📝 Examples

### Example 1: Development Workflow

```bash
# Setup development environment
cp docker.env .env
# Edit .env with your API keys

# Start development stack
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access Jupyter for experimentation
open http://localhost:8888

# Make code changes in src/ - they auto-reload
# Test changes in browser at http://localhost:3000

# View logs
docker-compose logs -f fintelligence-api
```

### Example 2: Production Deployment

```bash
# Prepare production environment
cp docker.env .env
# Configure production settings in .env

# Deploy with nginx reverse proxy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d

# Verify deployment
curl http://localhost/health

# Monitor with Grafana
open http://localhost:3001
```

### Example 3: Local AI Setup

```bash
# Add local AI support
docker-compose --profile local-ai up -d

# Wait for Ollama to start, then pull models
docker-compose exec ollama ollama pull llama3.2
docker-compose exec ollama ollama pull nomic-embed-text

# Configure API to use local models
echo "DSPY_LOCAL_MODE=true" >> .env
echo "DSPY_MODEL_PROVIDER=ollama" >> .env
docker-compose restart fintelligence-api
```
