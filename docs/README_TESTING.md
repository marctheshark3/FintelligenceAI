# FintelligenceAI Quick Testing & Setup

## 🚀 Your System is Running!

✅ **FintelligenceAI is currently running on `http://localhost:8000`**

## Immediate Tests You Can Run

### 1. **Basic Health Check**
```bash
curl http://localhost:8000/health
```
**Expected**: `{"status":"healthy","app":"FintelligenceAI",...}`

### 2. **Agent Status Check**
```bash
curl http://localhost:8000/agents/status
```
**Expected**: JSON with orchestrator and 3 agents (research, generation, validation)

### 3. **Agent Health Check**
```bash
curl http://localhost:8000/agents/health
```

### 4. **Simple Research Query** (requires OpenAI API key)
```bash
curl -X POST http://localhost:8000/agents/research \
  -H "Content-Type: application/json" \
  -d '{"query": "ErgoScript basics", "scope": "quick"}'
```

### 5. **Simple Code Generation** (requires OpenAI API key)
```bash
curl -X POST http://localhost:8000/agents/generate-code/simple \
  -H "Content-Type: application/json" \
  -d '{"description": "Simple hello world contract", "complexity_level": "beginner"}'
```

### 6. **Run Test Script**
```bash
./test_api.sh
```

## 📋 Current System Status

Based on our testing:

- ✅ **API Server**: Running on port 8000
- ✅ **Multi-Agent System**: All 4 agents initialized (orchestrator, research, generation, validation)
- ✅ **Core Health**: System reports healthy
- ✅ **FastAPI Framework**: Working correctly
- ⚠️ **AI Features**: May need OpenAI API key configuration
- ⚠️ **ChromaDB**: Connection issues (system still functional)

## 🔧 How to Start the System

### Quick Start (What's Currently Running)
```bash
# This is what you have running now:
PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py
```

### To Restart the System
```bash
# Stop current process (Ctrl+C if running in terminal)
# Start ChromaDB
docker-compose up -d chromadb

# Start the application
PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py
```

### Alternative Startup Methods
```bash
# Using uvicorn directly (with auto-reload)
poetry run uvicorn src.fintelligence_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

# Background process
nohup PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py > app.log 2>&1 &
```

## 🛠️ Configuration

### Environment Variables (in `.env` file)
```bash
# Required for AI features
OPENAI_API_KEY=sk-your-key-here

# Application settings
APP_ENVIRONMENT=development
API_PORT=8000

# Database connections
DATABASE_URL=postgresql+asyncpg://fintelligence_user:fintelligence_pass@localhost:5432/fintelligence_ai
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8100
```

## 📚 Documentation

- **[Complete Setup Guide](SETUP_GUIDE.md)** - Full installation and configuration
- **[API Testing Guide](API_TESTS.md)** - Comprehensive API tests and examples
- **[FintelligenceAI.md](FintelligenceAI.md)** - Complete project documentation

## ⚡ Performance Comparison

**Local vs Docker:**
- **Local Poetry**: ~30 seconds install, ~15 seconds startup ✅
- **Docker Build**: 2000+ seconds (33+ minutes) ❌

**Recommendation**: Use local development for now, Docker for production.

## 🐛 Troubleshooting

### If the system stops working:

1. **Check if running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Restart the application**:
   ```bash
   PYTHONPATH=src poetry run python src/fintelligence_ai/api/main.py
   ```

3. **Check logs** (if running in background):
   ```bash
   tail -f app.log
   ```

4. **Port conflicts**:
   ```bash
   sudo lsof -i :8000  # Check what's using port 8000
   ```

### Common Issues:

- **"Module not found"**: Ensure `PYTHONPATH=src` is set
- **"Connection refused"**: The application stopped, restart it
- **"500 errors"**: Usually missing OpenAI API key for AI features
- **"422 errors"**: Invalid JSON in your API requests

## 🎯 Next Steps

1. **Set up OpenAI API key** in `.env` file for full AI functionality
2. **Try the API tests** in `API_TESTS.md`
3. **Explore the complete setup guide** in `SETUP_GUIDE.md`
4. **Run the validation script**: `poetry run python scripts/validate_system.py`

---

**🎉 You have a working FintelligenceAI system!**

The core framework is running perfectly. Configure your OpenAI API key to unlock the full AI-powered ErgoScript generation capabilities.
