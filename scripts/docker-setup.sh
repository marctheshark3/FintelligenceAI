#!/bin/bash

# FintelligenceAI Docker Setup Script
# This script helps you get started with FintelligenceAI using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_TEMPLATE="$PROJECT_ROOT/env.template"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker installation
check_docker() {
    log_info "Checking Docker installation..."

    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker and try again."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command_exists docker-compose; then
        log_error "Docker Compose is not installed. Please install Docker Compose and try again."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi

    log_success "Docker is properly installed and running"
}

# Function to check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check available memory
    if command_exists free; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$AVAILABLE_MEM" -lt 4096 ]; then
            log_warning "Available memory is ${AVAILABLE_MEM}MB. Recommended minimum is 4GB."
        else
            log_success "Memory check passed (${AVAILABLE_MEM}MB available)"
        fi
    fi

    # Check available disk space
    AVAILABLE_DISK=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    AVAILABLE_DISK_GB=$((AVAILABLE_DISK / 1024 / 1024))
    if [ "$AVAILABLE_DISK_GB" -lt 10 ]; then
        log_warning "Available disk space is ${AVAILABLE_DISK_GB}GB. Recommended minimum is 10GB."
    else
        log_success "Disk space check passed (${AVAILABLE_DISK_GB}GB available)"
    fi
}

# Function to setup environment file
setup_env_file() {
    log_info "Setting up environment file..."

    if [ -f "$ENV_FILE" ]; then
        log_warning "Environment file already exists at $ENV_FILE"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing environment file"
            return
        fi
    fi

    if [ ! -f "$ENV_TEMPLATE" ]; then
        log_error "Environment template not found at $ENV_TEMPLATE"
        exit 1
    fi

    cp "$ENV_TEMPLATE" "$ENV_FILE"
    log_success "Environment file created at $ENV_FILE (copied from env.template)"
    log_info "Please edit $ENV_FILE to configure your API keys and other settings"
}

# Function to configure API keys
configure_api_keys() {
    log_info "Configuring API keys..."

    echo
    echo "FintelligenceAI requires API keys for AI functionality."
    echo "You can configure them now or edit the .env file later."
    echo

    read -p "Do you want to configure API keys now? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping API key configuration"
        return
    fi

    # OpenAI API Key
    read -p "Enter your OpenAI API key (or press Enter to skip): " OPENAI_KEY
    if [ -n "$OPENAI_KEY" ]; then
        sed -i.bak "s/OPENAI_API_KEY=your_openai_api_key_here/OPENAI_API_KEY=$OPENAI_KEY/" "$ENV_FILE"
        log_success "OpenAI API key configured"
    fi

    # Anthropic API Key
    read -p "Enter your Anthropic API key (optional, press Enter to skip): " ANTHROPIC_KEY
    if [ -n "$ANTHROPIC_KEY" ]; then
        sed -i.bak "s/ANTHROPIC_API_KEY=your_anthropic_api_key_here/ANTHROPIC_API_KEY=$ANTHROPIC_KEY/" "$ENV_FILE"
        log_success "Anthropic API key configured"
    fi

    # GitHub Token
    read -p "Enter your GitHub token (optional, press Enter to skip): " GITHUB_TOKEN
    if [ -n "$GITHUB_TOKEN" ]; then
        sed -i.bak "s/GITHUB_TOKEN=your_github_personal_access_token_here/GITHUB_TOKEN=$GITHUB_TOKEN/" "$ENV_FILE"
        log_success "GitHub token configured"
    fi

    # Clean up backup files
    rm -f "$ENV_FILE.bak"
}

# Function to configure passwords
configure_passwords() {
    log_info "Configuring secure passwords..."

    # Generate random passwords
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    JWT_SECRET=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
    GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)

    # Update environment file
    sed -i.bak "s/POSTGRES_PASSWORD=your_secure_password_here/POSTGRES_PASSWORD=$DB_PASSWORD/" "$ENV_FILE"
    sed -i.bak "s/JWT_SECRET_KEY=your_very_secret_jwt_key_here_use_strong_random_string/JWT_SECRET_KEY=$JWT_SECRET/" "$ENV_FILE"
    sed -i.bak "s/GRAFANA_PASSWORD=admin/GRAFANA_PASSWORD=$GRAFANA_PASSWORD/" "$ENV_FILE"

    # Clean up backup files
    rm -f "$ENV_FILE.bak"

    log_success "Secure passwords generated and configured"
    log_info "Grafana admin password: $GRAFANA_PASSWORD (save this!)"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."

    mkdir -p "$PROJECT_ROOT/data/"{chroma,uploads,dspy_cache,experiments,backups}
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/config/"{prometheus,grafana/provisioning,nginx,ssl}
    mkdir -p "$PROJECT_ROOT/notebooks"

    log_success "Directories created"
}

# Function to start services
start_services() {
    log_info "Starting FintelligenceAI services..."

    cd "$PROJECT_ROOT"

    echo
    echo "Choose deployment mode:"
    echo "1) Development (with hot reloading and dev tools)"
    echo "2) Production (optimized for performance)"
    echo "3) Basic (minimal services)"
    echo "4) Local AI (includes Ollama for local models)"
    echo
    read -p "Select mode (1-4): " -n 1 -r MODE
    echo

    case $MODE in
        1)
            log_info "Starting in development mode..."
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
            DEVELOPMENT_MODE=true
            ;;
        2)
            log_info "Starting in production mode..."
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
            DEVELOPMENT_MODE=false
            ;;
        3)
            log_info "Starting in basic mode..."
            docker-compose up -d
            DEVELOPMENT_MODE=false
            ;;
        4)
            log_info "Starting with local AI support..."
            docker-compose --profile local-ai up -d
            DEVELOPMENT_MODE=false
            LOCAL_AI=true
            ;;
        *)
            log_warning "Invalid selection. Starting in basic mode..."
            docker-compose up -d
            DEVELOPMENT_MODE=false
            ;;
    esac

    log_success "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    # Wait for API to be ready
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_success "API is ready"
            break
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warning "API took longer than expected to start. Check logs with: docker-compose logs fintelligence-api"
    fi

    echo
}

# Function to display access information
display_access_info() {
    log_success "FintelligenceAI is now running!"
    echo
    echo "🌐 Access URLs:"
    echo "   • Web Interface: http://localhost:3000"
    echo "   • API Documentation: http://localhost:8000/docs"
    echo "   • API Health Check: http://localhost:8000/health"
    echo
    echo "📊 Monitoring:"
    echo "   • Grafana Dashboard: http://localhost:3001"
    echo "   • Prometheus Metrics: http://localhost:9090"
    echo

    if [ "$DEVELOPMENT_MODE" = true ]; then
        echo "🛠️ Development Tools:"
        echo "   • Database Admin: http://localhost:8080"
        echo "   • Redis Admin: http://localhost:8081"
        echo
    fi

    if [ "$LOCAL_AI" = true ]; then
        echo "🤖 Local AI:"
        echo "   • Ollama Server: http://localhost:11434"
        echo "   • Pull models with: docker-compose exec ollama ollama pull llama3.2"
        echo
    fi

    echo "📋 Useful Commands:"
    echo "   • View logs: docker-compose logs -f fintelligence-api"
    echo "   • Stop services: docker-compose down"
    echo "   • Restart: docker-compose restart"
    echo
    echo "📖 For more information, see README-Docker.md"
}

# Function to display help
show_help() {
    echo "FintelligenceAI Docker Setup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --check-only   Only check requirements, don't set up"
    echo "  --env-only     Only set up environment file"
    echo "  --start-only   Only start services (requires existing .env)"
    echo "  --stop         Stop all services"
    echo "  --reset        Stop services and remove all data (⚠️  destructive)"
    echo
    echo "Examples:"
    echo "  $0              # Full setup and start"
    echo "  $0 --check-only # Only check requirements"
    echo "  $0 --stop       # Stop all services"
}

# Function to stop services
stop_services() {
    log_info "Stopping FintelligenceAI services..."
    cd "$PROJECT_ROOT"
    docker-compose down
    log_success "Services stopped"
}

# Function to reset (stop and remove data)
reset_services() {
    log_warning "This will stop all services and remove all data!"
    read -p "Are you sure? Type 'yes' to confirm: " -r
    echo
    if [ "$REPLY" != "yes" ]; then
        log_info "Reset cancelled"
        return
    fi

    log_info "Stopping services and removing data..."
    cd "$PROJECT_ROOT"
    docker-compose down -v
    docker system prune -f
    rm -rf "$PROJECT_ROOT/data" "$PROJECT_ROOT/logs"
    log_success "Services stopped and data removed"
}

# Main function
main() {
    echo "🚀 FintelligenceAI Docker Setup"
    echo "==============================="
    echo

    # Parse command line arguments
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        --check-only)
            check_docker
            check_requirements
            exit 0
            ;;
        --env-only)
            setup_env_file
            configure_api_keys
            configure_passwords
            exit 0
            ;;
        --start-only)
            check_docker
            if [ ! -f "$ENV_FILE" ]; then
                log_error "Environment file not found. Run setup first or use --env-only"
                exit 1
            fi
            start_services
            wait_for_services
            display_access_info
            exit 0
            ;;
        --stop)
            stop_services
            exit 0
            ;;
        --reset)
            reset_services
            exit 0
            ;;
        "")
            # Full setup
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac

    # Full setup process
    check_docker
    check_requirements
    setup_env_file
    configure_api_keys
    configure_passwords
    create_directories
    start_services
    wait_for_services
    display_access_info

    echo
    log_success "Setup complete! 🎉"
    echo
    echo "Next steps:"
    echo "1. Visit http://localhost:3000 to access the web interface"
    echo "2. Check the logs if you encounter any issues: docker-compose logs"
    echo "3. Read README-Docker.md for detailed documentation"
    echo
}

# Run main function
main "$@"
