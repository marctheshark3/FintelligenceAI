#!/usr/bin/env python3
"""
Test script for the ErgoScript knowledge base setup.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fintelligence_ai.knowledge.ingestion import setup_ergoscript_knowledge_base, get_knowledge_base_stats
from fintelligence_ai.rag.factory import create_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_knowledge_base_setup():
    """Test the complete knowledge base setup and RAG pipeline."""
    
    print("🚀 Starting ErgoScript Knowledge Base Test")
    print("=" * 50)
    
    # Step 1: Set up knowledge base
    print("\n📚 Setting up knowledge base...")
    result = await setup_ergoscript_knowledge_base()
    
    if result.success:
        print(f"✅ Success! Processed {result.documents_processed} documents")
        print(f"   Created {result.chunks_created} chunks")
        print(f"   Stored {result.chunks_stored} chunks")
        print(f"   Processing time: {result.processing_time_seconds:.2f}s")
    else:
        print("❌ Knowledge base setup failed!")
        for error in result.errors:
            print(f"   Error: {error}")
        return False
    
    # Step 2: Get statistics
    print("\n📊 Getting knowledge base statistics...")
    stats = await get_knowledge_base_stats()
    
    if stats:
        print(f"   Documents: {stats.get('document_count', 0)}")
        print(f"   Collection: {stats.get('collection_name', 'Unknown')}")
        print(f"   Size: {stats.get('collection_size_mb', 0)} MB")
    else:
        print("   No statistics available")
    
    # Step 3: Test RAG pipeline
    print("\n🔍 Testing RAG pipeline...")
    try:
        pipeline = create_rag_pipeline("development")
        await pipeline.initialize()
        
        test_queries = [
            "How do I create a simple token contract in ErgoScript?",
            "Show me an example of a pin lock contract",
            "What is a swap contract and how does it work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            result = await pipeline.process_query(query, retrieval_strategy="hybrid")
            
            print(f"   Response length: {len(result.response)} characters")
            print(f"   Retrieved docs: {len(result.retrieved_documents)}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Processing time: {result.processing_time_ms}ms")
            
            # Show first retrieved document
            if result.retrieved_documents:
                first_doc = result.retrieved_documents[0]
                title = first_doc.metadata.get('title', 'Unknown')
                print(f"   Top result: {title} (score: {first_doc.score:.3f})")
    
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_knowledge_base_setup())
    sys.exit(0 if success else 1)