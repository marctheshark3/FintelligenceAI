# Quick Start Guide

> Get FintelligenceAI running in 5 minutes

## 🚀 5-Minute Setup

### Prerequisites

- Python 3.9+ installed
- Docker installed and running
- 4GB+ free RAM

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone https://github.com/fintelligence-ai/fintelligence-ai.git
cd FintelligenceAI

# Install dependencies
pip install poetry
poetry install
```

### Step 2: Quick Configuration

```bash
# Copy environment template
cp env.template .env

# Edit .env with your API key (required)
echo "OPENAI_API_KEY=your-key-here" >> .env
echo "APP_ENVIRONMENT=development" >> .env
```

### Step 3: Start Services

```bash
# Start required services
docker-compose up -d chromadb redis

# Verify services
docker ps
```

### Step 4: Run FintelligenceAI

```bash
# Start the application
poetry run python -m fintelligence_ai.api.main
```

### Step 5: Set Up Knowledge Base (Optional)

```bash
# Add some example content
echo "https://github.com/ergoplatform/eips" >> knowledge-base/github-repos.txt

# Ingest with real-time progress tracking
python scripts/ingest_knowledge.py

# Verify content was processed
python scripts/ingest_knowledge.py --show-tree
```

### Step 6: Test & Verify

Open your browser to http://localhost:8000/docs and test the API!

## 🎯 First API Call

```bash
# Test health endpoint
curl http://localhost:8000/health

# Generate your first ErgoScript
curl -X POST http://localhost:8000/agents/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a simple token contract"}'
```

## 🔄 Alternative: Local-Only Mode

Want to run completely offline? Use Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3:latest

# Enable local mode
echo "DSPY_LOCAL_MODE=true" >> .env
echo "OLLAMA_MODEL=llama3:latest" >> .env

# Start without external APIs
poetry run python -m fintelligence_ai.api.main
```

## 📚 Next Steps

- **[Full Installation Guide](./INSTALLATION.md)** - Complete setup options
- **[Configuration Guide](./CONFIGURATION.md)** - Customize your setup
- **[Knowledge Base Guide](./KNOWLEDGE_BASE.md)** - Add your own content
- **[Local Mode Setup](./LOCAL_MODE_SETUP.md)** - Complete offline operation

## 🆘 Troubleshooting

**Port 8000 already in use?**
```bash
export API_PORT=8080
poetry run python -m fintelligence_ai.api.main
```

**Docker issues?**
```bash
# Reset Docker
docker-compose down
docker-compose up -d --force-recreate
```

**API key errors?**
```bash
# Verify your .env file
cat .env | grep OPENAI_API_KEY
```

**Need help?** Check [Troubleshooting Guide](./TROUBLESHOOTING.md) or open an issue on GitHub.
