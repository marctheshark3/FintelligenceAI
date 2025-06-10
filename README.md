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

- Python 3.9 or higher
- Docker and Docker Compose
- OpenAI API key (or other LLM provider)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/fintelligence-ai/fintelligence-ai.git
   cd fintelligence-ai
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp env.template .env

   # Edit .env with your API keys and configuration
   nano .env
   ```

3. **Install dependencies**
   ```bash
   # Using Poetry (recommended)
   pip install poetry
   poetry install --with dev

   # Or using pip
   pip install -e ".[dev]"
   ```

4. **Start services with Docker**
   ```bash
   docker-compose up -d
   ```

5. **Run the application**
   ```bash
   # Development mode
   uvicorn fintelligence_ai.api.main:app --reload

   # Or using the Docker container
   docker-compose up fintelligence-api
   ```

### Access Points

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Jupyter Lab**: http://localhost:8888 (token: `fintelligence-dev-token`)
- **Grafana**: http://localhost:3000 (admin/fintelligence-admin)
- **ChromaDB**: http://localhost:8100

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

```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose --profile monitoring up -d
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

## 📊 Performance Metrics

Current system performance targets:

- **Code Generation**: <3 seconds response time
- **Retrieval Accuracy**: >0.8 relevance score
- **Code Correctness**: >90% syntactically correct
- **System Uptime**: >99.5% availability

## 🗺️ Roadmap

### Phase 1: Foundation (Completed)
- [x] Project setup and configuration
- [x] Core DSPy modules
- [x] Basic RAG pipeline
- [x] Vector database integration

### Phase 2: Enhancement (In Progress)
- [ ] AI agent framework
- [ ] Advanced retrieval strategies
- [ ] DSPy optimization integration
- [ ] Comprehensive evaluation suite

### Phase 3: Production
- [ ] API development
- [ ] Deployment infrastructure
- [ ] Monitoring and observability
- [ ] User interface

### Phase 4: Optimization
- [ ] Continuous model improvement
- [ ] Multi-domain support
- [ ] Advanced optimization techniques
- [ ] Community features

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
