# Development Guide

> Complete guide for developing and contributing to FintelligenceAI

## 📖 Table of Contents

- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [API Development](#api-development)
- [Performance Optimization](#performance-optimization)

## Development Environment

### Prerequisites

```bash
# Required tools
Python 3.11+
Poetry
Docker & Docker Compose
Git
```

### Setup

```bash
# 1. Clone and setup
git clone https://github.com/fintelligence-ai/fintelligence-ai.git
cd FintelligenceAI

# 2. Install dependencies
poetry install --with dev

# 3. Setup pre-commit hooks
poetry run pre-commit install

# 4. Start development services
docker-compose up -d chromadb redis

# 5. Start development server
poetry run python -m fintelligence_ai.api.main --reload
```

## Project Structure

```
FintelligenceAI/
├── src/fintelligence_ai/         # Main application code
│   ├── api/                      # FastAPI application
│   ├── agents/                   # AI agent implementations
│   ├── core/                     # Core DSPy modules
│   ├── rag/                      # RAG pipeline components
│   ├── knowledge/                # Knowledge base management
│   ├── models/                   # Data models and schemas
│   ├── utils/                    # Utility functions
│   ├── config/                   # Configuration management
│   └── cli.py                   # Command-line interface
├── tests/                        # Test suite
├── docs/                         # Documentation
└── config/                       # Configuration files
```

## Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes and test
poetry run pytest tests/ -v

# 3. Format code
poetry run black src/ tests/
poetry run ruff check src/ tests/ --fix

# 4. Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

## Code Standards

### Python Style Guide

```python
# Follow PEP 8 with type hints
from typing import Optional, List
import asyncio

class DocumentProcessor:
    """Processes documents for knowledge base ingestion."""

    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size

    async def process(
        self,
        content: str,
        metadata: Optional[dict] = None
    ) -> List[str]:
        """Process document content into chunks."""
        chunks = self._chunk_content(content)
        return [await self._clean_chunk(chunk) for chunk in chunks]
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=fintelligence_ai --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/ -v
```

### Writing Tests

```python
# tests/unit/test_agents.py
import pytest
from fintelligence_ai.agents.generation import GenerationAgent

class TestGenerationAgent:
    async def test_generate_simple_prompt(self):
        agent = GenerationAgent()
        result = await agent.generate("Create a token contract")
        assert result is not None
```

## API Development

### Creating New Endpoints

```python
# 1. Define models
from pydantic import BaseModel

class NewFeatureRequest(BaseModel):
    input_data: str
    options: Optional[dict] = None

# 2. Add route
from fastapi import APIRouter

router = APIRouter(prefix="/new-feature")

@router.post("/process")
async def process_data(request: NewFeatureRequest):
    # Implementation
    return {"result": "processed"}
```

## Performance Optimization

### Async Best Practices

```python
# Use async/await properly
async def process_documents(documents: List[str]) -> List[str]:
    tasks = [process_single_document(doc) for doc in documents]
    return await asyncio.gather(*tasks)
```

### Caching Strategies

```python
# Use Redis for caching
@cache(ttl=3600)  # Cache for 1 hour
async def expensive_operation(param: str) -> str:
    await asyncio.sleep(5)
    return f"Result for {param}"
```

## Contributing

### Pull Request Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with Black
- [ ] Type hints added
- [ ] All checks pass

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/fintelligence-ai/fintelligence-ai/issues)
- **Discord**: [Join community](https://discord.gg/fintelligence)

---

**Happy coding!** 🚀 Thank you for contributing to FintelligenceAI.
