#!/bin/bash

# FintelligenceAI System Startup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
echo "🚀 FintelligenceAI System Startup"
echo "=================================="

# Check if running from project root
if [ ! -f "docker-compose.yml" ]; then
    error "Must be run from project root directory (where docker-compose.yml is located)"
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
        warn "Required variables: OPENAI_API_KEY, DATABASE_URL, etc."
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

# Create necessary directories
log "Creating data directories..."
mkdir -p data/chroma data/chroma_dev data/uploads data/experiments logs
mkdir -p data/backups data/cache
success "Data directories created"

# Cleanup any existing containers (optional)
if [[ "$1" == "--clean" ]]; then
    log "Cleaning up existing containers..."
    docker-compose down -v 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
fi

# Pull latest images
log "Pulling latest Docker images..."
docker-compose pull

# Build application image
log "Building application image..."
docker-compose build

# Start the system
log "Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
log "Waiting for services to start..."

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            success "$service_name is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    error "$service_name failed to start after ${max_attempts} attempts"
    return 1
}

# Check services
log "Checking service health..."

# API Health Check
echo -n "API (port 8000): "
if check_service "API" "http://localhost:8000/health"; then
    API_READY=true
else
    API_READY=false
fi

# ChromaDB Health Check  
echo -n "ChromaDB (port 8100): "
if check_service "ChromaDB" "http://localhost:8100/api/v1/heartbeat"; then
    CHROMA_READY=true
else
    CHROMA_READY=false
fi

# PostgreSQL Health Check
echo -n "PostgreSQL (port 5432): "
if docker-compose exec -T postgres pg_isready -U fintelligence_user > /dev/null 2>&1; then
    success "PostgreSQL is ready!"
    POSTGRES_READY=true
else
    error "PostgreSQL is not ready"
    POSTGRES_READY=false
fi

# Redis Health Check
echo -n "Redis (port 6379): "
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    success "Redis is ready!"
    REDIS_READY=true
else
    error "Redis is not ready"
    REDIS_READY=false
fi

# Overall health assessment
echo ""
log "Service Status Summary:"
echo "  API:        $($API_READY && echo "✅ Ready" || echo "❌ Failed")"
echo "  ChromaDB:   $($CHROMA_READY && echo "✅ Ready" || echo "❌ Failed")"
echo "  PostgreSQL: $($POSTGRES_READY && echo "✅ Ready" || echo "❌ Failed")"
echo "  Redis:      $($REDIS_READY && echo "✅ Ready" || echo "❌ Failed")"

# Check if we can run validation
if $API_READY; then
    echo ""
    log "Running quick system validation..."
    if python3 scripts/validate_system.py; then
        success "System validation passed!"
    else
        warn "System validation had some issues, but core services are running"
    fi
else
    warn "Skipping validation as API is not ready"
fi

# Show service URLs and information
echo ""
success "🎉 FintelligenceAI System Started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Service URLs:"
echo "  🔗 API Documentation:     http://localhost:8000/docs"
echo "  🔗 API Health Check:      http://localhost:8000/health"
echo "  🔗 Swagger UI:            http://localhost:8000/docs"
echo "  🔗 ReDoc:                 http://localhost:8000/redoc"
echo ""
echo "  📊 Grafana Dashboard:     http://localhost:3000"
echo "      Username: admin"
echo "      Password: fintelligence-admin"
echo ""
echo "  📈 Prometheus:            http://localhost:9090"
echo "  🪐 Jupyter Lab:           http://localhost:8888"
echo "      Token: fintelligence-dev-token"
echo "  💾 ChromaDB:              http://localhost:8100"
echo ""
echo "  🗄️ PostgreSQL:            localhost:5432"
echo "      Database: fintelligence_ai"
echo "      Username: fintelligence_user"
echo ""
echo "  🏃 Redis:                 localhost:6379"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🛠️  Management Commands:"
echo "  • View logs:               docker-compose logs -f"
echo "  • Stop system:             docker-compose down"
echo "  • Restart specific service: docker-compose restart <service>"
echo "  • Run validation:          python3 scripts/validate_system.py"
echo "  • Quick test:              bash scripts/quick_test.sh"
echo ""
echo "🔧 Troubleshooting:"
echo "  • Check service status:    docker-compose ps"
echo "  • View service logs:       docker-compose logs <service>"
echo "  • Rebuild containers:      docker-compose up --build -d"
echo "  • Full reset:              bash scripts/start_system.sh --clean"
echo ""

# Final status
if $API_READY && $CHROMA_READY; then
    success "System is ready for use! 🚀"
    exit 0
else
    warn "System started but some services may need attention. Check the logs above."
    exit 1
fi 