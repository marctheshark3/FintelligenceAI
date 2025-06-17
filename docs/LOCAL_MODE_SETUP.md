# FintelligenceAI Local-Only Mode Setup Guide

This guide explains how to set up and use FintelligenceAI in local-only mode using Ollama instead of OpenAI, enabling complete offline operation without external API dependencies.

## Overview

Local-only mode allows you to run FintelligenceAI entirely on your local machine using open-source language models via Ollama. This provides:

- **Complete Privacy**: All data and processing stays on your machine
- **No API Costs**: No charges for API usage
- **Offline Operation**: Works without internet connection (after initial setup)
- **Full Control**: Choose from various open-source models
- **Compliance**: Meet strict data privacy requirements

## Prerequisites

### System Requirements
- **RAM**: Minimum 8GB, recommended 16GB+ for better performance
- **Storage**: At least 10GB free space for models
- **OS**: Linux, macOS, or Windows
- **GPU**: Optional but recommended for faster inference (NVIDIA with CUDA support)

### Software Requirements
- Docker and Docker Compose
- Ollama server
- Python 3.9+

## Installation Steps

### 1. Install Ollama

#### On Linux/macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### On Windows:
1. Download the installer from https://ollama.ai/download
2. Run the installer and follow the setup wizard

#### Verify Installation:
```bash
ollama --version
```

### 2. Pull Required Models

FintelligenceAI requires both a language model and an embedding model for local operation.

#### Language Models (choose one):

**Recommended for development:**
```bash
# Llama 3.2 (3B parameters) - good balance of speed and capability
ollama pull llama3.2

# Llama 3.2 (1B parameters) - fastest, lower resource usage
ollama pull llama3.2:1b
```

**For better performance:**
```bash
# Llama 3.1 (8B parameters) - better quality, requires more resources
ollama pull llama3.1:8b

# Code Llama (7B parameters) - specialized for code generation
ollama pull codellama:7b

# Mistral (7B parameters) - alternative high-quality model
ollama pull mistral:7b
```

#### Embedding Models:
```bash
# Nomic Embed Text (recommended) - optimized for text embeddings
ollama pull nomic-embed-text

# Alternative embedding models
ollama pull mxbai-embed-large
ollama pull all-minilm
```

### 3. Start Ollama Server

```bash
# Start Ollama server (runs on port 11434 by default)
ollama serve
```

Verify the server is running:
```bash
curl http://localhost:11434/api/tags
```

### 4. Configure FintelligenceAI

Create or update your environment configuration:

#### Using .env file:
```bash
# Create .env file in project root
cat > .env << 'EOF'
# Enable local-only mode
DSPY_LOCAL_MODE=true
DSPY_MODEL_PROVIDER=ollama

# Ollama configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=4096
OLLAMA_TIMEOUT=300
OLLAMA_KEEP_ALIVE=5m

# Optional: Disable OpenAI completely
# OPENAI_API_KEY=
EOF
```

#### Using environment variables:
```bash
export DSPY_LOCAL_MODE=true
export DSPY_MODEL_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Configuration Options

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DSPY_LOCAL_MODE` | `false` | Enable/disable local-only mode |
| `DSPY_MODEL_PROVIDER` | `openai` | Model provider (ollama, openai, claude) |

### Ollama Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_HOST` | `localhost` | Ollama server hostname |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `OLLAMA_MODEL` | `llama3.2` | Default language model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature (0.0-2.0) |
| `OLLAMA_MAX_TOKENS` | `4096` | Maximum tokens per response |
| `OLLAMA_TIMEOUT` | `300` | Request timeout in seconds |
| `OLLAMA_KEEP_ALIVE` | `5m` | Keep model in memory duration |
| `OLLAMA_BASE_URL` | - | Full URL (overrides host:port) |
| `OLLAMA_VERIFY_SSL` | `true` | Verify SSL certificates |

### Performance Tuning

#### Memory Management:
```bash
# Keep models loaded longer for faster responses
export OLLAMA_KEEP_ALIVE=30m

# Or never unload models (uses more memory)
export OLLAMA_KEEP_ALIVE=-1
```

#### GPU Acceleration:
```bash
# Enable GPU if available (automatic detection)
# Ensure NVIDIA drivers and CUDA are installed

# Check GPU usage
nvidia-smi

# Monitor Ollama logs for GPU usage
journalctl -u ollama -f
```

## Running FintelligenceAI in Local Mode

### 1. Start the Application

#### Development Mode:
```bash
# Start with local configuration
python -m fintelligence_ai.api.main
```

#### Production Mode:
```bash
# Using Docker Compose
docker-compose up -d

# Or using the slim configuration for local-only
docker-compose -f docker-compose.slim.yml up -d
```

### 2. Verify Local Mode

Check the startup logs for confirmation:
```
✅ DSPy configured with Ollama (Local) - Model: llama3.2
   Server: http://localhost:11434
```

### 3. Test the Configuration

#### CLI Health Check:
```bash
# Run health check
fintelligence health

# Expected output should show:
# ✅ Ollama Server: Available
# ✅ Language Model: llama3.2
# ✅ Embedding Model: nomic-embed-text
```

#### API Health Check:
```bash
curl http://localhost:8000/health
```

## Model Recommendations

### By Use Case

**Development & Testing:**
- Language Model: `llama3.2:1b` or `llama3.2`
- Embedding Model: `nomic-embed-text`
- RAM Usage: ~4-6GB

**Production & Quality:**
- Language Model: `llama3.1:8b` or `codellama:7b`
- Embedding Model: `nomic-embed-text` or `mxbai-embed-large`
- RAM Usage: ~8-12GB

**High Performance:**
- Language Model: `llama3.1:70b` (requires 40GB+ RAM)
- Embedding Model: `nomic-embed-text`
- RAM Usage: 40GB+

### Model Performance Comparison

| Model | Size | RAM Usage | Speed | Quality | Best For |
|-------|------|-----------|-------|---------|----------|
| llama3.2:1b | 1.3GB | ~2GB | Fast | Good | Development, Quick Testing |
| llama3.2 | 2.0GB | ~4GB | Medium | Very Good | General Use, Balanced |
| llama3.1:8b | 4.7GB | ~8GB | Slower | Excellent | Production, High Quality |
| codellama:7b | 3.8GB | ~6GB | Medium | Good* | Code Generation |
| mistral:7b | 4.1GB | ~7GB | Medium | Excellent | Alternative to Llama |

*CodeLlama is specialized for code but may not perform as well for general text.

## Troubleshooting

### Common Issues

#### 1. Ollama Server Not Available
```bash
Error: Ollama server not available at http://localhost:11434

Solutions:
1. Check if Ollama is running: ps aux | grep ollama
2. Start Ollama: ollama serve
3. Check port availability: netstat -tulpn | grep 11434
4. Try different port: export OLLAMA_PORT=11435
```

#### 2. Model Not Found
```bash
Error: Model llama3.2 not available and could not be pulled

Solutions:
1. Pull model manually: ollama pull llama3.2
2. Check available models: ollama list
3. Use different model: export OLLAMA_MODEL=llama3.1
```

#### 3. Out of Memory
```bash
Error: Model failed to load - insufficient memory

Solutions:
1. Use smaller model: export OLLAMA_MODEL=llama3.2:1b
2. Close other applications
3. Add swap space
4. Use model with lower memory requirements
```

#### 4. Slow Performance
```bash
Solutions:
1. Use GPU acceleration (install CUDA)
2. Use smaller model for faster responses
3. Increase keep_alive: export OLLAMA_KEEP_ALIVE=30m
4. Add more RAM
5. Use SSD storage
```

#### 5. Embedding Dimension Mismatch
```bash
Error: Embedding dimension mismatch

Solutions:
1. Clear vector database: rm -rf ./data/chroma
2. Restart application to rebuild index
3. Ensure consistent embedding model
```

### Debug Mode

Enable detailed logging:
```bash
export APP_LOG_LEVEL=DEBUG
export DSPY_LOG_LEVEL=DEBUG

# Run application
python -m fintelligence_ai.api.main
```

### Health Monitoring

#### Check Ollama Status:
```bash
# List loaded models
ollama ps

# Check model details
ollama show llama3.2

# Monitor resource usage
htop
nvidia-smi  # For GPU
```

## Advanced Configuration

### Custom Model Configuration

#### Using Custom Models:
```bash
# Pull custom model
ollama pull custom-model:latest

# Configure FintelligenceAI
export OLLAMA_MODEL=custom-model:latest
```

#### Model Parameters:
```bash
# Fine-tune model behavior
export OLLAMA_TEMPERATURE=0.2    # More deterministic
export OLLAMA_TOP_P=0.9          # Nucleus sampling
export OLLAMA_TOP_K=50           # Top-k sampling
```

### Multi-Model Setup

Run different models for different tasks:
```bash
# Different models for different agents
export OLLAMA_RESEARCH_MODEL=llama3.1:8b
export OLLAMA_GENERATION_MODEL=codellama:7b
export OLLAMA_VALIDATION_MODEL=llama3.2
```

### Docker Configuration

#### Custom Docker Compose for Local Mode:
```yaml
# docker-compose.local.yml
version: '3.8'
services:
  fintelligence-ai:
    build: .
    environment:
      - DSPY_LOCAL_MODE=true
      - OLLAMA_HOST=ollama
      - OLLAMA_MODEL=llama3.2
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0

volumes:
  ollama_data:
```

Start with:
```bash
docker-compose -f docker-compose.local.yml up -d
```

## Performance Optimization

### Hardware Optimization

#### CPU Optimization:
```bash
# Use all available CPU cores
export OMP_NUM_THREADS=$(nproc)
export OLLAMA_NUM_THREADS=$(nproc)
```

#### Memory Optimization:
```bash
# Optimize memory usage
export OLLAMA_MMAP=true
export OLLAMA_LOW_VRAM=true  # For limited GPU memory
```

#### GPU Optimization:
```bash
# Force GPU usage
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1
```

### Application Optimization

#### Caching:
```bash
# Enable aggressive caching
export DSPY_CACHE_ENABLED=true
export DSPY_CACHE_SIZE=1000
```

#### Batch Processing:
```bash
# Optimize batch sizes
export EMBEDDING_BATCH_SIZE=50
export RAG_BATCH_SIZE=10
```

## Migration from OpenAI

### Gradual Migration

1. **Start with Development**: Use local mode for development
2. **Test Performance**: Compare quality and speed
3. **Update Configuration**: Switch environment variables
4. **Monitor Results**: Check logs and metrics
5. **Full Migration**: Deploy to production

### Configuration Comparison

| Feature | OpenAI Mode | Local Mode |
|---------|-------------|------------|
| API Key | Required | Not needed |
| Internet | Required | Optional* |
| Cost | Per token | Hardware only |
| Privacy | External | Complete local |
| Models | GPT-4, GPT-3.5 | Llama, Mistral, etc. |
| Performance | Very fast | Depends on hardware |

*Internet needed only for initial model download

### Quality Expectations

- **Code Generation**: CodeLlama performs well for ErgoScript
- **RAG Performance**: May require tuning for specific domains
- **Response Quality**: Generally good, varies by model size
- **Speed**: Slower than OpenAI but acceptable for most use cases

## Support and Community

### Getting Help

1. **Documentation**: Check this guide and project docs
2. **Issues**: Report bugs on GitHub
3. **Community**: Join Discord/Slack for support
4. **Ollama Support**: Official Ollama documentation

### Contributing

- Test different model combinations
- Report performance benchmarks
- Submit configuration improvements
- Share optimization tips

## Future Enhancements

Planned improvements for local mode:

- **Model Switching**: Dynamic model selection per task
- **Quantization**: Support for quantized models
- **Distributed Setup**: Multi-node Ollama clusters
- **Model Fine-tuning**: Domain-specific model training
- **Performance Monitoring**: Detailed metrics and alerting

---

For questions or issues with local mode setup, please refer to the troubleshooting section or open an issue on the project repository.
