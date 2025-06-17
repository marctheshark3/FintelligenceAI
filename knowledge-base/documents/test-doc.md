# Test Document for FintelligenceAI Knowledge Base

This is a test document to verify that the knowledge base ingestion system works correctly with Ollama embeddings.

## Purpose

This document serves as a simple test case for:

- **Local embeddings**: Using Ollama instead of OpenAI
- **Document processing**: Testing markdown parsing
- **Vector storage**: Verifying chunks are stored correctly

## ErgoScript Example

Here's a simple ErgoScript example:

```ergoscript
{
    val threshold = 1000000L
    sigmaProp(OUTPUTS(0).value >= threshold)
}
```

This contract ensures that the first output has a value of at least 1 ERG.

## Testing Notes

- **Embedding Model**: nomic-embed-text (via Ollama)
- **Category**: general
- **Source**: local_files
- **Complexity**: beginner

If you can see this document in the knowledge base after ingestion, the system is working correctly!
