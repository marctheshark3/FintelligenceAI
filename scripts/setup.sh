#!/bin/bash

# FintelligenceAI Development Environment Setup Script
# This script verifies and sets up the development environment

set -e  # Exit on any error

echo "🚀 FintelligenceAI Development Environment Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_status "Python $python_version is installed and meets requirements (>= $required_version)"
else
    print_error "Python $required_version or higher is required. Current version: $python_version"
    exit 1
fi

# Check if Docker is installed
echo "Checking Docker installation..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | cut -d' ' -f3 | sed 's/,//')
    print_status "Docker $docker_version is installed"
else
    print_warning "Docker is not installed. Some features may not work."
fi

# Check if Docker Compose is installed
echo "Checking Docker Compose installation..."
if command -v docker-compose &> /dev/null; then
    compose_version=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
    print_status "Docker Compose $compose_version is installed"
else
    print_warning "Docker Compose is not installed. Container orchestration may not work."
fi

# Check if Poetry is installed
echo "Checking Poetry installation..."
if command -v poetry &> /dev/null; then
    poetry_version=$(poetry --version | cut -d' ' -f3)
    print_status "Poetry $poetry_version is installed"
else
    print_warning "Poetry is not installed. Installing via pip..."
    pip install poetry
    print_status "Poetry installed successfully"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/{chroma,uploads,dspy_cache,experiments}
mkdir -p logs
mkdir -p config/{redis,nginx,prometheus}
print_status "Directories created successfully"

# Check if .env file exists
echo "Checking environment configuration..."
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Copying from template..."
    if [ -f "env.template" ]; then
        cp env.template .env
        print_status ".env file created from template"
        print_warning "Please edit .env file with your API keys and configuration"
    else
        print_error "env.template not found. Please create a .env file manually."
    fi
else
    print_status ".env file exists"
fi

# Install dependencies with Poetry
echo "Installing Python dependencies..."
if poetry install --with dev; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
if poetry run pre-commit install; then
    print_status "Pre-commit hooks installed successfully"
else
    print_warning "Failed to install pre-commit hooks. You can install them manually later with 'pre-commit install'"
fi

# Test imports
echo "Testing core imports..."
if poetry run python -c "import dspy, chromadb, fastapi, pydantic; print('All core imports successful')"; then
    print_status "Core dependencies are working correctly"
else
    print_error "Some core dependencies are not working. Please check the installation."
    exit 1
fi

# Test configuration loading
echo "Testing configuration loading..."
if poetry run python -c "from src.fintelligence_ai.config import get_settings; settings = get_settings(); print(f'Configuration loaded: {settings.app_name}')"; then
    print_status "Configuration system is working correctly"
else
    print_warning "Configuration system needs attention. Please check your .env file."
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Start services: docker-compose up -d"
echo "  3. Run the application: poetry run uvicorn fintelligence_ai.api.main:app --reload"
echo "  4. Access API docs at: http://localhost:8000/docs"
echo ""
print_status "Development environment is ready!"
