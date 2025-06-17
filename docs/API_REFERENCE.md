# API Reference

> Complete API reference for FintelligenceAI

## 📖 Table of Contents

- [Authentication](#authentication)
- [Base URL](#base-url)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)

## Authentication

FintelligenceAI supports multiple authentication methods:

### API Key Authentication
```bash
# Header-based authentication
curl -H "X-API-Key: your-api-key" \
     https://api.fintelligence.ai/v1/agents/generate

# Query parameter authentication
curl "https://api.fintelligence.ai/v1/agents/generate?api_key=your-api-key"
```

### JWT Authentication
```bash
# Get token
curl -X POST https://api.fintelligence.ai/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -H "Authorization: Bearer your-jwt-token" \
     https://api.fintelligence.ai/v1/agents/generate
```

## Base URL

- **Production**: `https://api.fintelligence.ai/v1`
- **Staging**: `https://staging-api.fintelligence.ai/v1`
- **Development**: `http://localhost:8000/api/v1`

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "message": "Optional message",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "prompt",
      "reason": "Prompt cannot be empty"
    }
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Invalid request parameters |
| `AUTHENTICATION_ERROR` | Invalid or missing authentication |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `MODEL_ERROR` | LLM model error |
| `KNOWLEDGE_BASE_ERROR` | Knowledge base retrieval error |
| `INTERNAL_ERROR` | Unexpected server error |

## Rate Limiting

| Endpoint | Rate Limit |
|----------|------------|
| `/agents/generate` | 100 requests/hour |
| `/knowledge/search` | 500 requests/hour |
| `/knowledge/upload` | 10 requests/hour |
| Other endpoints | 1000 requests/hour |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642176000
```

## Endpoints

### Health Check

#### GET `/health`

Check system health and status.

**Request:**
```bash
curl https://api.fintelligence.ai/v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600,
    "services": {
      "database": "connected",
      "vector_db": "connected",
      "cache": "connected",
      "llm_provider": "available"
    }
  }
}
```

---

### Agent Operations

#### POST `/agents/generate`

Generate ErgoScript code using AI agents.

**Request:**
```json
{
  "prompt": "Create a token contract with minting capability",
  "agent_type": "generation",
  "temperature": 0.1,
  "max_tokens": 2000,
  "include_explanation": true,
  "context": {
    "use_case": "token",
    "complexity": "intermediate"
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Natural language description |
| `agent_type` | string | No | Agent type: `generation`, `research`, `validation` |
| `temperature` | float | No | Generation temperature (0.0-2.0) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `include_explanation` | boolean | No | Include code explanation |
| `context` | object | No | Additional context parameters |

**Response:**
```json
{
  "success": true,
  "data": {
    "code": "{\n  val tokenContract = ...\n}",
    "explanation": "This contract implements...",
    "metadata": {
      "agent_type": "generation",
      "model": "gpt-4",
      "tokens_used": 1500,
      "processing_time": 2.3,
      "confidence": 0.92
    },
    "validation": {
      "syntax_valid": true,
      "complexity_score": 7,
      "security_notes": []
    }
  }
}
```

#### POST `/agents/research`

Research information about ErgoScript topics.

**Request:**
```json
{
  "query": "How to implement oracle contracts in ErgoScript?",
  "depth": "comprehensive",
  "include_examples": true,
  "sources": ["docs", "github", "community"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": "Oracle contracts in ErgoScript...",
    "key_points": [
      "Data posting mechanisms",
      "Trust models",
      "Price feed implementation"
    ],
    "examples": [
      {
        "title": "Basic Oracle Contract",
        "code": "...",
        "description": "..."
      }
    ],
    "references": [
      {
        "title": "Oracle Documentation",
        "url": "https://docs.ergoplatform.com/oracle",
        "relevance": 0.95
      }
    ]
  }
}
```

#### POST `/agents/validate`

Validate ErgoScript code.

**Request:**
```json
{
  "code": "{\n  val contract = ...\n}",
  "validation_type": "comprehensive",
  "check_security": true,
  "check_performance": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "syntax_check": {
      "valid": true,
      "errors": []
    },
    "security_analysis": {
      "score": 8.5,
      "vulnerabilities": [],
      "recommendations": [
        "Consider adding input validation"
      ]
    },
    "performance_analysis": {
      "complexity": "medium",
      "gas_estimate": 1200,
      "optimizations": []
    }
  }
}
```

---

### Knowledge Base

#### POST `/knowledge/upload`

Upload documents to the knowledge base.

**Request:**
```bash
curl -X POST https://api.fintelligence.ai/v1/knowledge/upload \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "category=documentation" \
  -F "metadata={\"author\":\"John Doe\"}"
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Document file (PDF, MD, TXT) |
| `category` | string | No | Document category |
| `metadata` | json | No | Additional metadata |

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456",
    "filename": "document.pdf",
    "size": 102400,
    "pages": 10,
    "chunks_created": 25,
    "processing_status": "completed",
    "metadata": {
      "category": "documentation",
      "author": "John Doe"
    }
  }
}
```

#### GET `/knowledge/search`

Search the knowledge base.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | integer | No | Number of results (default: 10) |
| `category` | string | No | Filter by category |
| `threshold` | float | No | Similarity threshold |

**Request:**
```bash
curl "https://api.fintelligence.ai/v1/knowledge/search?query=token%20contracts&limit=5"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "chunk_789",
        "content": "Token contracts in ErgoScript...",
        "similarity": 0.92,
        "metadata": {
          "source": "official_docs",
          "category": "examples",
          "document_id": "doc_123456"
        }
      }
    ],
    "total_results": 25,
    "query_time": 0.15
  }
}
```

#### GET `/knowledge/documents`

List documents in the knowledge base.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `limit` | integer | No | Items per page (default: 20) |
| `category` | string | No | Filter by category |

**Response:**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "id": "doc_123456",
        "filename": "ergoscript_guide.pdf",
        "category": "documentation",
        "upload_date": "2025-01-15T10:00:00Z",
        "size": 102400,
        "chunks": 25
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 100,
      "total_pages": 5
    }
  }
}
```

#### DELETE `/knowledge/documents/{document_id}`

Delete a document from the knowledge base.

**Request:**
```bash
curl -X DELETE https://api.fintelligence.ai/v1/knowledge/documents/doc_123456
```

**Response:**
```json
{
  "success": true,
  "message": "Document deleted successfully"
}
```

---

### Configuration

#### GET `/config/models`

Get available models and current configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "current": {
      "primary_model": "gpt-4",
      "embedding_model": "text-embedding-3-large",
      "local_mode": false
    },
    "available_models": {
      "generation": [
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-sonnet",
        "llama3:latest"
      ],
      "embedding": [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "nomic-embed-text"
      ]
    }
  }
}
```

#### POST `/config/models`

Update model configuration.

**Request:**
```json
{
  "primary_model": "gpt-4",
  "embedding_model": "text-embedding-3-large",
  "temperature": 0.1,
  "local_mode": false
}
```

---

### Statistics

#### GET `/stats/usage`

Get API usage statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "last_30_days",
    "requests": {
      "total": 1500,
      "successful": 1450,
      "failed": 50
    },
    "tokens": {
      "input": 500000,
      "output": 300000,
      "total": 800000
    },
    "by_endpoint": {
      "/agents/generate": 800,
      "/knowledge/search": 600,
      "/agents/validate": 100
    }
  }
}
```

#### GET `/stats/performance`

Get system performance metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "response_times": {
      "avg": 2.3,
      "p50": 1.8,
      "p95": 4.2,
      "p99": 7.1
    },
    "success_rates": {
      "overall": 0.967,
      "by_endpoint": {
        "/agents/generate": 0.95,
        "/knowledge/search": 0.99
      }
    },
    "resource_usage": {
      "cpu": 45.2,
      "memory": 67.8,
      "storage": 23.1
    }
  }
}
```

## SDK Examples

### Python SDK

```python
from fintelligence_ai import FintelligenceClient

# Initialize client
client = FintelligenceClient(api_key="your-api-key")

# Generate code
result = await client.agents.generate(
    prompt="Create a token contract",
    agent_type="generation",
    temperature=0.1
)

print(result.code)
print(result.explanation)

# Search knowledge base
results = await client.knowledge.search(
    query="oracle contracts",
    limit=5
)

for result in results:
    print(f"Score: {result.similarity}")
    print(f"Content: {result.content}")
```

### JavaScript SDK

```javascript
import { FintelligenceClient } from '@fintelligence/sdk';

// Initialize client
const client = new FintelligenceClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.fintelligence.ai/v1'
});

// Generate code
const result = await client.agents.generate({
  prompt: 'Create a token contract',
  agentType: 'generation',
  temperature: 0.1
});

console.log(result.code);
console.log(result.explanation);
```

### cURL Examples

```bash
# Generate ErgoScript
curl -X POST https://api.fintelligence.ai/v1/agents/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Create a simple token contract",
    "agent_type": "generation",
    "temperature": 0.1
  }'

# Search knowledge base
curl "https://api.fintelligence.ai/v1/knowledge/search?query=contracts&limit=5" \
  -H "X-API-Key: your-api-key"

# Upload document
curl -X POST https://api.fintelligence.ai/v1/knowledge/upload \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "category=documentation"
```

## Webhooks

### Event Types

| Event | Description |
|-------|-------------|
| `document.uploaded` | New document added to knowledge base |
| `document.processed` | Document processing completed |
| `generation.completed` | Code generation finished |
| `validation.completed` | Code validation finished |

### Webhook Payload

```json
{
  "event": "document.uploaded",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "document_id": "doc_123456",
    "filename": "guide.pdf",
    "category": "documentation"
  }
}
```

---

## Next Steps

- **[Quick Start Guide](./QUICK_START.md)** - Get started quickly
- **[Authentication Guide](./AUTHENTICATION.md)** - Detailed auth setup
- **[SDK Documentation](./SDK_REFERENCE.md)** - Language-specific SDKs
- **[Webhook Guide](./WEBHOOKS.md)** - Event notifications

---

**Need help?** Contact our support team at support@fintelligence.ai
