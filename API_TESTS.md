# FintelligenceAI API Testing Guide

This guide provides comprehensive API tests you can run against your FintelligenceAI instance.

## Prerequisites

- FintelligenceAI running on `http://localhost:8000`
- `curl` installed
- `jq` installed (optional, for pretty JSON output)

## Quick Health Checks

### 1. Basic Health Check
```bash
curl -X GET "http://localhost:8000/health" | jq
```

**Expected Response:**
```json
{
  "status": "healthy",
  "app": "FintelligenceAI",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2024-01-01T00:00:00Z",
  "services": {
    "api": "healthy",
    "database": "not_checked",
    "redis": "not_checked",
    "chromadb": "not_checked"
  }
}
```

### 2. Agent Health Check
```bash
curl -X GET "http://localhost:8000/agents/health" | jq
```

### 3. Agent Status
```bash
curl -X GET "http://localhost:8000/agents/status" | jq
```

## Core Agent Testing

### 4. Research Agent - Documentation Lookup
```bash
curl -X POST "http://localhost:8000/agents/research" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to create a simple ErgoScript contract for token issuance?",
    "scope": "comprehensive",
    "include_examples": true
  }' | jq
```

### 5. Simple Code Generation
```bash
curl -X POST "http://localhost:8000/agents/generate-code/simple" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a simple token issuance contract",
    "use_case": "token",
    "complexity_level": "beginner",
    "requirements": ["Fixed token supply", "Owner-only minting"]
  }' | jq
```

### 6. Code Validation
```bash
curl -X POST "http://localhost:8000/agents/validate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "{ val tokenId = INPUTS(0).tokens(0)._1; tokenId }",
    "use_case": "token",
    "validation_criteria": {
      "syntax_check": true,
      "semantic_check": true,
      "security_check": true
    }
  }' | jq
```

### 7. Full Orchestrated Code Generation
```bash
curl -X POST "http://localhost:8000/agents/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a decentralized auction contract for NFTs",
    "use_case": "auction",
    "complexity_level": "intermediate",
    "requirements": [
      "Minimum bid increment",
      "Auction duration",
      "Automatic winner selection"
    ],
    "constraints": [
      "Gas efficient",
      "Secure bid handling"
    ]
  }' | jq
```

## Advanced Features Testing

### 8. Optimization Metrics
```bash
curl -X GET "http://localhost:8000/optimization/metrics/summary" | jq
```

### 9. Quick Evaluation
```bash
curl -X POST "http://localhost:8000/optimization/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "{ val tokenId = INPUTS(0).tokens(0)._1; tokenId }",
    "criteria": {
      "functionality": true,
      "security": true,
      "efficiency": true
    }
  }' | jq
```

### 10. Knowledge Search
```bash
curl -X POST "http://localhost:8000/knowledge/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ErgoScript box protection",
    "filters": {
      "category": "documentation",
      "difficulty": "intermediate"
    },
    "max_results": 5
  }' | jq
```

## Test Scripts

### Quick Test Script
Create a file `test_api.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"
echo "🧪 Testing FintelligenceAI API..."

# Test 1: Health Check
echo "1. Health Check..."
curl -s -X GET "$BASE_URL/health" | jq '.status' || echo "❌ Failed"

# Test 2: Agent Status  
echo "2. Agent Status..."
curl -s -X GET "$BASE_URL/agents/status" | jq '.orchestrator.status' || echo "❌ Failed"

# Test 3: Simple Research
echo "3. Research Query..."
curl -s -X POST "$BASE_URL/agents/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "ErgoScript basics", "scope": "quick"}' \
  | jq '.findings' || echo "❌ Failed"

# Test 4: Simple Generation
echo "4. Code Generation..."
curl -s -X POST "$BASE_URL/agents/generate-code/simple" \
  -H "Content-Type: application/json" \
  -d '{"description": "Hello world contract", "complexity_level": "beginner"}' \
  | jq '.generated_code' || echo "❌ Failed"

echo "✅ Tests completed!"
```

Make it executable and run:
```bash
chmod +x test_api.sh
./test_api.sh
```

### Load Testing (Optional)
```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Test with 10 concurrent requests
hey -n 10 -c 2 -H "Content-Type: application/json" \
  -d '{"query": "ErgoScript help"}' \
  -m POST http://localhost:8000/agents/research
```

## Expected Error Scenarios

### Testing Error Handling
```bash
# Invalid JSON
curl -X POST "http://localhost:8000/agents/research" \
  -H "Content-Type: application/json" \
  -d '{invalid json}' 

# Missing required fields
curl -X POST "http://localhost:8000/agents/generate-code" \
  -H "Content-Type: application/json" \
  -d '{}' 

# Non-existent endpoint
curl -X GET "http://localhost:8000/non-existent" 
```

## Performance Testing

### Response Time Testing
```bash
# Time API responses
time curl -s -X GET "http://localhost:8000/health" > /dev/null

# Detailed timing
curl -w "@curl-format.txt" -s -X GET "http://localhost:8000/agents/status" -o /dev/null
```

Create `curl-format.txt`:
```
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the API is running on port 8000
2. **500 Errors**: Check logs for missing dependencies or configuration
3. **422 Errors**: Validate your JSON payload structure
4. **Timeout**: Some operations (like code generation) may take longer

### Debug Mode
Run with verbose output:
```bash
curl -v -X GET "http://localhost:8000/health"
```

### Check Logs
Monitor the application logs while testing:
```bash
# In another terminal
tail -f logs/app.log  # If logging to file
# OR monitor the running process output
```

## Interactive Testing with HTTPie (Alternative)

Install HTTPie: `pip install httpie`

```bash
# Health check
http GET localhost:8000/health

# Research query
http POST localhost:8000/agents/research query="ErgoScript tutorial" scope="quick"

# Code generation  
http POST localhost:8000/agents/generate-code/simple \
  description="Simple token contract" \
  complexity_level="beginner"
```

## WebSocket Testing (If Available)

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket (if implemented)
wscat -c ws://localhost:8000/ws
```

---

**Note**: Some endpoints may return errors initially due to missing configurations (database connections, API keys, etc.). This is normal for a development setup. Focus on testing the core health and status endpoints first. 