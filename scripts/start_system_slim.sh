#!/bin/bash

# FintelligenceAI Slim System Startup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

# Print banner
echo "🚀 FintelligenceAI Slim System Startup"
echo "======================================="
echo "📦 This uses optimized Docker builds for faster startup and smaller images"
echo ""

# Check if running from project root
if [ ! -f "docker-compose.slim.yml" ]; then
    error "Must be run from project root directory (where docker-compose.slim.yml is located)"
    exit 1
fi

# Check for required tools
log "Checking required tools..."
for tool in docker docker-compose python3; do
    if ! command -v $tool &> /dev/null; then
        error "$tool is not installed or not in PATH"
        exit 1
    fi
done
success "All required tools are available"

# Check Docker daemon
log "Checking Docker daemon..."
if ! docker info &> /dev/null; then
    error "Docker daemon is not running. Please start Docker first."
    exit 1
fi
success "Docker daemon is running"

# Environment setup
log "Setting up environment..."

# Check if .env exists
if [ ! -f ".env" ]; then
    warn "No .env file found, creating from template..."
    if [ -f "env.template" ]; then
        cp env.template .env
        warn "Please edit .env file with your API keys and configuration"
        echo ""
        echo "Do you want to continue with default values? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log "Please configure .env file and run the script again"
            exit 1
        fi
    else
        error "No env.template found. Please create .env file manually."
        exit 1
    fi
fi

# Create necessary directories (minimal for slim build)
log "Creating minimal data directories..."
mkdir -p data/chroma data/uploads logs
success "Data directories created"

# Cleanup any existing containers (optional)
if [[ "$1" == "--clean" ]]; then
    log "Cleaning up existing containers..."
    docker-compose -f docker-compose.slim.yml down -v 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
fi

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build and start the system
log "Building optimized containers (this may take a few minutes on first run)..."

# Use build cache if available
if docker images | grep -q "fintelligence-ai"; then
    log "Using cached layers for faster build..."
fi

docker-compose -f docker-compose.slim.yml build --parallel

log "Starting slim containers..."
docker-compose -f docker-compose.slim.yml up -d

# Wait for services with shorter timeouts
log "Waiting for services to start (optimized checks)..."

# Function to check service health with shorter timeout
check_service_quick() {
    local service_name=$1
    local url=$2
    local max_attempts=15  # Reduced from 30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            success "$service_name is ready!"
            return 0
        fi
        echo -n "."
        sleep 1  # Reduced from 2
        ((attempt++))
    done
    
    error "$service_name failed to start after ${max_attempts} attempts"
    return 1
}

# Quick health checks
echo -n "API: "
if check_service_quick "API" "http://localhost:8000/health"; then
    API_READY=true
else
    API_READY=false
fi

echo -n "ChromaDB: "
if check_service_quick "ChromaDB" "http://localhost:8100/api/v1/heartbeat"; then
    CHROMA_READY=true
else
    CHROMA_READY=false
fi

# Database quick checks
echo -n "PostgreSQL: "
if docker-compose -f docker-compose.slim.yml exec -T postgres pg_isready -U fintelligence_user > /dev/null 2>&1; then
    success "PostgreSQL is ready!"
    POSTGRES_READY=true
else
    warn "PostgreSQL may still be starting..."
    POSTGRES_READY=false
fi

echo -n "Redis: "
if docker-compose -f docker-compose.slim.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    success "Redis is ready!"
    REDIS_READY=true
else
    warn "Redis may still be starting..."
    REDIS_READY=false
fi

# Show system info
echo ""
log "Slim System Status:"
echo "  API:        $($API_READY && echo "✅ Ready" || echo "❌ Failed")"
echo "  ChromaDB:   $($CHROMA_READY && echo "✅ Ready" || echo "❌ Failed")"
echo "  PostgreSQL: $($POSTGRES_READY && echo "✅ Ready" || echo "⏳ Starting")"
echo "  Redis:      $($REDIS_READY && echo "✅ Ready" || echo "⏳ Starting")"

# Show container sizes
echo ""
log "Container Information:"
docker-compose -f docker-compose.slim.yml images

# Quick validation if API is ready
if $API_READY; then
    echo ""
    log "Running quick API validation..."
    if curl -s "http://localhost:8000/health" | grep -q "status"; then
        success "API validation passed!"
    else
        warn "API responding but validation needs attention"
    fi
else
    warn "Skipping validation as API is not ready"
fi

# Show service URLs
echo ""
success "🎉 FintelligenceAI Slim System Started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Service URLs:"
echo "  🔗 API Documentation:     http://localhost:8000/docs"
echo "  🔗 API Health Check:      http://localhost:8000/health"
echo "  💾 ChromaDB:              http://localhost:8100"
echo "  🗄️ PostgreSQL:            localhost:5432"
echo "  🏃 Redis:                 localhost:6379"
echo ""
echo "🛠️  Management Commands:"
echo "  • View logs:               docker-compose -f docker-compose.slim.yml logs -f"
echo "  • Stop system:             docker-compose -f docker-compose.slim.yml down"
echo "  • Quick test:              bash scripts/quick_test.sh"
echo "  • Full validation:         python3 scripts/validate_system.py"
echo ""
echo "📊 Performance Benefits:"
echo "  • Smaller image sizes"
echo "  • Faster startup times"
echo "  • Reduced memory usage"
echo "  • Optimized layer caching"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Final status
if $API_READY && $CHROMA_READY; then
    success "Slim system is ready for use! 🚀"
    exit 0
else
    warn "System started but some services may need a moment. Check logs if needed."
    exit 1
fi 