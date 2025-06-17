# FintelligenceAI

> Intelligent RAG Pipeline & AI Agent System for ErgoScript Smart Contract Generation

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

FintelligenceAI is a comprehensive framework for building modular RAG (Retrieval-Augmented Generation) pipelines and AI agents using DSPy. The system enables end-to-end development of domain-specific language generation models with intelligent retrieval capabilities, starting with Ergo smart contract script generation.

## 🚀 Key Features

- **Modular RAG Pipeline**: Vector-based semantic search with domain-specific embeddings
- **AI Agent Framework**: Multi-agent orchestration with tool integration
- **DSPy Integration**: Advanced optimization using MIPROv2 and BootstrapFinetune
- **ErgoScript Specialization**: Curated knowledge base and validation tools
- **Production Ready**: Scalable deployment with monitoring and observability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │  AI Agents      │    │  RAG Pipeline   │
│                 │───▶│                 │───▶│                 │
│ Natural Language│    │ Research/Gen/Val│    │ Retrieval + Gen │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Tool Calling  │    │ Vector Database │
                       │                 │    │                 │
                       │ Ergo Validation │    │ ChromaDB/Pinecone│
                       └─────────────────┘    └─────────────────┘
```

## 🔧 Quick Start

### Prerequisites

- **Docker & Docker Compose**: Required for containerized deployment
- **Git**: For cloning the repository
- **4GB+ RAM**: Recommended for optimal performance
- **10GB+ Disk Space**: For Docker images and data storage

### Installation Options

#### Option 1: Automated Setup (Recommended) 🚀

The fastest way to get FintelligenceAI running is using our automated setup script:

1. **Clone the repository**
   ```bash
   git clone https://github.com/marctheshark3/FintelligenceAI.git
   cd FintelligenceAI
   ```

2. **Run the automated setup script**
   ```bash
   chmod +x scripts/docker-setup.sh
   ./scripts/docker-setup.sh
   ```

   The script will:
   - ✅ Check Docker installation and system requirements
   - ✅ Set up environment configuration
   - ✅ Configure API keys (optional)
   - ✅ Generate secure passwords automatically
   - ✅ Create necessary directories
   - ✅ Start all services with your chosen deployment mode

3. **Choose your deployment mode** when prompted:
   - **Development** (1): Hot reloading + dev tools
   - **Production** (2): Optimized for performance
   - **Basic** (3): Minimal services
   - **Local AI** (4): Includes Ollama for local models

#### Option 2: Manual Setup

1. **Clone and setup environment**
   ```bash
   git clone https://github.com/marctheshark3/FintelligenceAI.git
   cd FintelligenceAI
   cp env.template .env
   ```

2. **Edit environment file**
   ```bash
   nano .env  # Add your API keys and configuration
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

### ⚡ Setup Script Options

The `docker-setup.sh` script supports various options for different use cases:

```bash
# Full interactive setup (recommended for first-time users)
./scripts/docker-setup.sh

# Quick start if you already have .env configured
./scripts/docker-setup.sh --start-only

# Setup environment file only
./scripts/docker-setup.sh --env-only

# Check system requirements without installing
./scripts/docker-setup.sh --check-only

# Stop all services
./scripts/docker-setup.sh --stop

# Reset everything (⚠️ destructive - removes all data)
./scripts/docker-setup.sh --reset

# Show help and all options
./scripts/docker-setup.sh --help
```

### 🌐 Access Points

Once setup is complete, you can access:

- **🌐 Web Interface**: http://localhost:3000
- **📊 API Documentation**: http://localhost:8000/docs
- **🔍 API Health Check**: http://localhost:8000/health
- **📈 Grafana Dashboard**: http://localhost:3001
- **🔧 Prometheus Metrics**: http://localhost:9090

### 🔑 API Key Configuration

FintelligenceAI supports multiple AI providers. During setup, you can configure:

- **OpenAI**: Required for GPT models (recommended)
- **Anthropic**: For Claude models (optional)
- **GitHub Token**: For enhanced repository access (optional)
- **Local Models**: Ollama (automatically configured in Local AI mode)

**Post-Setup Configuration**: You can always edit your API keys in the `.env` file and restart services:

```bash
nano .env  # Edit your configuration
docker-compose restart
```

### 🐳 Docker Setup Details

The automated setup script (`scripts/docker-setup.sh`) provides a comprehensive Docker deployment solution:

#### What the Setup Script Does

1. **System Checks**: Verifies Docker installation and system requirements
2. **Environment Setup**: Creates `.env` file from template with secure defaults
3. **Password Generation**: Automatically generates secure passwords for databases
4. **Directory Creation**: Sets up required data and configuration directories
5. **Service Orchestration**: Starts services in your chosen deployment mode
6. **Health Verification**: Waits for services to be ready and accessible

#### Deployment Modes Explained

| Mode | Use Case | Includes | Resource Usage |
|------|----------|----------|----------------|
| **Development** | Local development with hot reloading | Dev tools, volume mounts, debug info | High |
| **Production** | Optimized for performance and stability | Optimized builds, health checks | Medium |
| **Basic** | Minimal setup for testing | Core services only | Low |
| **Local AI** | Complete offline AI capability | Ollama, local models | High |

#### Troubleshooting

```bash
# Check service status
docker-compose ps

# View logs for specific service
docker-compose logs fintelligence-api
docker-compose logs fintelligence-ui

# Restart specific service
docker-compose restart fintelligence-api

# Check system requirements
./scripts/docker-setup.sh --check-only

# Complete reset if issues persist
./scripts/docker-setup.sh --reset
./scripts/docker-setup.sh  # Fresh setup
```

#### Port Configuration

If you need to change default ports (e.g., if 3000 or 8000 are in use), edit your `.env` file:

```bash
# Edit ports in .env file
UI_PORT=3001          # Web interface port
API_PORT=8001         # API port
GRAFANA_PORT=3002     # Grafana dashboard port
OLLAMA_PORT=11435     # Ollama port (if using Local AI mode)
```

## 📁 Project Structure

```
fintelligence-ai/
├── src/fintelligence_ai/
│   ├── api/                 # FastAPI application
│   ├── agents/              # AI agent implementations
│   ├── core/                # Core DSPy modules
│   ├── rag/                 # RAG pipeline components
│   ├── models/              # Data models and schemas
│   ├── utils/               # Utility functions
│   └── config/              # Configuration management
├── tests/                   # Test suite
├── docs/                    # Documentation
│   └── ErgoScript_Security_Guide.md  # ErgoScript security best practices
├── scripts/                 # Utility scripts
├── data/                    # Data storage
├── notebooks/               # Jupyter notebooks
└── config/                  # Configuration files
└── mining_emission_contract.md      # Mining contract case study
```

## 🔬 Development Workflow

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   poetry install --with dev
   ```

2. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Code formatting and linting**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   ```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fintelligence_ai

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## 📖 Usage Examples

### Basic RAG Query

```python
from fintelligence_ai import FintelligenceAI

# Initialize the system
ai = FintelligenceAI()

# Generate ErgoScript
result = ai.generate_script(
    "Create a simple token contract with minting capability"
)

print(result.code)
print(result.explanation)
```

### Using Agents

```python
from fintelligence_ai.agents import ResearchAgent, GenerationAgent

# Research phase
research_agent = ResearchAgent()
research_result = await research_agent.research(
    "ErgoScript token standards and best practices"
)

# Generation phase
gen_agent = GenerationAgent()
code = await gen_agent.generate(
    context=research_result,
    requirements="ERC-20 compatible token"
)
```

## 🔧 Configuration

The application uses environment variables for configuration. Key settings include:

```bash
# LLM Provider
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/chroma

# Ergo Integration
ERGO_NODE_URL=http://localhost:9052

# DSPy Settings
DSPY_OPTIMIZER=MIPROv2
DSPY_TRAINING_SIZE=100
```

See `env.template` for complete configuration options.

## 🚀 Deployment

### Docker Deployment

#### Using the Setup Script (Recommended)

```bash
# Production deployment with automated setup
./scripts/docker-setup.sh
# Choose option (2) Production mode when prompted

# Or force production mode directly
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Manual Deployment Options

```bash
# Development mode with hot reloading
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production mode (optimized)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Local AI mode (includes Ollama)
docker-compose --profile local-ai up -d

# Basic mode (minimal services)
docker-compose up -d
```

#### Deployment Management

```bash
# View logs
docker-compose logs -f fintelligence-api

# Stop services
./scripts/docker-setup.sh --stop
# or manually
docker-compose down

# Restart services
docker-compose restart

# Reset everything (⚠️ removes all data)
./scripts/docker-setup.sh --reset
```

### Cloud Deployment

The application is designed for cloud deployment with:

- **Auto-scaling**: Horizontal scaling support
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Observability**: LangSmith tracing and Sentry error tracking
- **Security**: JWT authentication and rate limiting

## 🧪 Optimization with DSPy

FintelligenceAI leverages DSPy's advanced optimization capabilities:

```python
from fintelligence_ai.core.optimizer import optimize_pipeline

# Optimize RAG pipeline
optimized_pipeline = optimize_pipeline(
    training_data=training_examples,
    optimizer="MIPROv2",
    max_iterations=50
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install

# Run the full test suite
pytest

# Check code quality
make lint
make test
make docs
```

## 🏆 Recent Accomplishments

### ErgoScript Security Leadership
- **Expert-Validated Security Guide**: Created comprehensive security best practices based on real expert feedback
- **Mining Contract Case Study**: Developed and fixed a production-ready token emission contract with 5 critical security improvements
- **Community Impact**: Established security patterns that prevent common ErgoScript vulnerabilities

### AI Agent System
- **Multi-Agent Architecture**: Research, Generation, and Validation agents working in coordination
- **RAG Integration**: Advanced retrieval-augmented generation with domain-specific knowledge
- **Production Ready**: Full API, monitoring, and deployment infrastructure

### Technical Excellence
- **Docker Infrastructure**: Multi-stage builds, production configurations, monitoring stack
- **Code Quality**: Pre-commit hooks, comprehensive testing, type checking
- **Expert Collaboration**: Integrated real expert feedback into development workflow

## 📊 Performance Metrics

Current system performance targets:

- **Code Generation**: <3 seconds response time
- **Retrieval Accuracy**: >0.8 relevance score
- **Code Correctness**: >90% syntactically correct
- **System Uptime**: >99.5% availability
- **Security Coverage**: 100% of critical ErgoScript vulnerabilities documented

## 🗺️ Roadmap

### Phase 1: Foundation ✅ (Completed)
- [x] Project setup and configuration
- [x] Core DSPy modules
- [x] Basic RAG pipeline
- [x] Vector database integration
- [x] Poetry dependency management
- [x] Docker containerization
- [x] Pre-commit hooks and code quality

### Phase 2: AI Agent Framework ✅ (Completed)
- [x] AI agent framework (Research, Generation, Validation agents)
- [x] Agent orchestration system
- [x] Advanced retrieval strategies
- [x] DSPy optimization integration
- [x] Comprehensive evaluation suite
- [x] Knowledge ingestion and processing
- [x] Multi-agent workflow coordination

### Phase 3: ErgoScript Specialization ✅ (Completed)
- [x] ErgoScript-specific knowledge base
- [x] Smart contract generation capabilities
- [x] Expert-validated security patterns
- [x] Mining emission contract case study
- [x] Comprehensive security guide
- [x] Contract validation and testing

### Phase 4: Production Infrastructure ✅ (Completed)
- [x] FastAPI application development
- [x] RESTful API endpoints
- [x] Docker deployment infrastructure
- [x] Monitoring and observability (Prometheus, Grafana)
- [x] Multi-environment configurations
- [x] Health checks and system validation
- [x] Production-ready Docker images

### Phase 5: Documentation & Security 🎯 (Recently Completed)
- [x] Comprehensive README and documentation
- [x] ErgoScript Security Best Practices Guide
- [x] Real-world security case studies
- [x] Expert validation workflows
- [x] Security checklists and templates
- [x] Community contribution guidelines

### Phase 6: Optimization & Enhancement (In Progress)
- [ ] Continuous model improvement based on usage
- [ ] Multi-domain smart contract support (beyond Ergo)
- [ ] Advanced optimization techniques
- [ ] Community features and feedback integration
- [ ] Performance monitoring and auto-scaling
- [ ] Advanced security scanning and validation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) for the fantastic optimization framework
- [Ergo Platform](https://ergoplatform.org/) for the innovative blockchain platform
- [LangChain](https://langchain.com/) for additional AI tooling
- The open-source community for invaluable tools and libraries

## 📚 Additional Documentation

- **[ErgoScript Security Guide](docs/ErgoScript_Security_Guide.md)**: Comprehensive security best practices for ErgoScript development
- **[Mining Emission Contract Case Study](mining_emission_contract.md)**: Real-world contract development with expert validation

## 📞 Support

- **Documentation**: [https://fintelligence-ai.readthedocs.io](https://fintelligence-ai.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/fintelligence-ai/fintelligence-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fintelligence-ai/fintelligence-ai/discussions)
- **Email**: contact@fintelligence.ai

---

**Built with ❤️ by the FintelligenceAI Team**
