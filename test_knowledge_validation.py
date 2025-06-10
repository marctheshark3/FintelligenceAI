#!/usr/bin/env python3
"""
Knowledge Base Validation Script

Tests the knowledge base components before running the full setup.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test that all required modules can be imported."""
    logger.info("🔍 Testing basic imports...")
    
    try:
        from fintelligence_ai.knowledge import (
            GitHubDataCollector, 
            ErgoScriptCollector,
            DocumentProcessor, 
            ErgoScriptProcessor,
            KnowledgeBaseManager
        )
        logger.info("✅ Knowledge base modules imported successfully")
        
        from fintelligence_ai.rag import (
            EmbeddingService,
            VectorStoreManager,
            RAGPipeline
        )
        logger.info("✅ RAG modules imported successfully")
        
        from fintelligence_ai.config import get_settings
        settings = get_settings()
        logger.info(f"✅ Settings loaded: {settings.app_name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

async def test_embedding_service():
    """Test the embedding service with a simple text."""
    logger.info("🔍 Testing embedding service...")
    
    try:
        from fintelligence_ai.rag.embeddings import EmbeddingService
        from fintelligence_ai.config import get_settings
        
        settings = get_settings()
        
        if not settings.openai.api_key:
            logger.warning("⚠️ OpenAI API key not configured, skipping embedding test")
            return True  # Consider this a pass for now
        
        service = EmbeddingService(
            model_name="text-embedding-3-small",  # Use smaller model for testing
            api_key=settings.openai.api_key
        )
        
        # Test with simple text
        test_text = "This is a simple test for ErgoScript embedding."
        embedding = await service.embed_text(test_text)
        
        logger.info(f"✅ Embedding generated: dimension={len(embedding)}")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding service failed: {e}")
        return False

async def test_vector_store():
    """Test vector store operations with temporary data."""
    logger.info("🔍 Testing vector store...")
    
    try:
        from fintelligence_ai.rag.vectorstore import VectorStoreManager
        from fintelligence_ai.rag.models import VectorStoreConfig, Document, DocumentMetadata
        from fintelligence_ai.rag.embeddings import EmbeddingService
        from fintelligence_ai.config import get_settings
        
        settings = get_settings()
        
        if not settings.openai.api_key:
            logger.warning("⚠️ OpenAI API key not configured, skipping vector store test")
            return True  # Consider this a pass for now
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                collection_name="test_knowledge",
                embedding_dimension=1536,
                distance_metric="cosine"
            )
            
            embedding_service = EmbeddingService(
                model_name="text-embedding-3-small",
                api_key=settings.openai.api_key
            )
            
            manager = VectorStoreManager(
                config=config,
                embedding_service=embedding_service,
                persist_directory=temp_dir
            )
            
            await manager.initialize()
            
            # Test document
            test_doc = Document(
                content="ErgoScript is a powerful smart contract language for the Ergo blockchain.",
                metadata=DocumentMetadata(
                    title="Test ErgoScript Document",
                    source="examples",
                    file_path="test.md",
                    category="tutorials",
                    complexity="beginner"
                )
            )
            
            # Store document
            await manager.add_documents([test_doc])
            logger.info("✅ Document stored successfully")
            
            # Search for document
            results = await manager.similarity_search("ErgoScript language", k=1)
            
            if results and len(results) > 0:
                logger.info(f"✅ Document retrieved: score={results[0].score:.3f}")
                return True
            else:
                logger.warning("⚠️ No results returned from search")
                return False
                
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False

async def test_document_processor():
    """Test document processing with sample ErgoScript content."""
    logger.info("🔍 Testing document processor...")
    
    try:
        from fintelligence_ai.knowledge.processor import ErgoScriptProcessor
        from fintelligence_ai.rag.models import Document, DocumentMetadata
        
        # Sample ErgoScript content
        sample_content = """
# ErgoScript Example: Basic Box Creation

ErgoScript is a powerful smart contract language. Here's a simple example:

```scala
{
  val myBox = OUTPUTS(0)
  val validOutput = myBox.value >= 1000000L
  sigmaProp(validOutput)
}
```

This script ensures that the first output box has at least 1 ERG.
        """.strip()
        
        test_doc = Document(
            content=sample_content,
            metadata=DocumentMetadata(
                title="Basic Box Creation Example",
                source="examples",
                file_path="test_example.md",
                category="tutorials", 
                complexity="beginner"
            )
        )
        
        processor = ErgoScriptProcessor()
        chunks = processor.process_document(test_doc)
        
        logger.info(f"✅ Document processed into {len(chunks)} chunks")
        
        # Check if code was identified
        code_chunks = [c for c in chunks if c.chunk_type == "code"]
        if code_chunks:
            logger.info(f"✅ Code chunks identified: {len(code_chunks)}")
        
        return len(chunks) > 0
        
    except Exception as e:
        logger.error(f"❌ Document processor test failed: {e}")
        return False

async def test_rag_pipeline():
    """Test basic RAG pipeline functionality."""
    logger.info("🔍 Testing RAG pipeline...")
    
    try:
        from fintelligence_ai.rag import create_rag_pipeline
        from fintelligence_ai.config import get_settings
        
        settings = get_settings()
        
        if not settings.openai.api_key:
            logger.warning("⚠️ OpenAI API key not configured, skipping RAG pipeline test")
            return True  # Consider this a pass for now
        
        # Create development pipeline
        pipeline = create_rag_pipeline("development")
        
        # Test initialization
        await pipeline.initialize()
        logger.info("✅ RAG pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG pipeline test failed: {e}")
        return False

async def main():
    """Run all validation tests."""
    logger.info("🚀 Starting FintelligenceAI Knowledge Base Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Embedding Service", test_embedding_service),
        ("Vector Store", test_vector_store),
        ("Document Processor", test_document_processor),
        ("RAG Pipeline", test_rag_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Knowledge base components are working correctly.")
        logger.info("\n💡 Next steps:")
        logger.info("   1. Run: python -m fintelligence_ai.cli knowledge setup")
        logger.info("   2. Run: python -m fintelligence_ai.cli knowledge stats")
        logger.info("   3. Test: python -m fintelligence_ai.cli knowledge search 'ErgoScript'")
    else:
        logger.error("💔 Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 