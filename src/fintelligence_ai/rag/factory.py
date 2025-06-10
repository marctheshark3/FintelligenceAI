"""
Factory functions for creating RAG pipeline instances.

This module provides convenient factory functions for creating RAG pipeline
instances with different configurations for various use cases.
"""

import logging
import os
from typing import Optional

from ..config.settings import get_settings
from .embeddings import EmbeddingService
from .models import VectorStoreConfig, RetrievalConfig, GenerationConfig
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def create_default_rag_pipeline(
    collection_name: str = "ergoscript_knowledge",
    persist_directory: Optional[str] = None,
    embedding_model: str = "text-embedding-3-large",
    generation_model: str = "gpt-4o-mini",
    retrieval: Optional[dict] = None,
    generation: Optional[dict] = None,
    vector_store: Optional[dict] = None,
) -> RAGPipeline:
    """
    Create a RAG pipeline with default configurations.
    
    Args:
        collection_name: Name for the vector store collection
        persist_directory: Directory for vector store persistence
        embedding_model: Embedding model to use
        generation_model: Generation model to use
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured RAG pipeline instance
    """
    settings = get_settings()
    
    # Use settings default if not provided
    persist_directory = persist_directory or settings.chromadb.persist_directory
    
    # Vector store configuration
    vector_store_defaults = {
        "embedding_dimension": 1536 if "small" in embedding_model else 3072,
        "distance_metric": "cosine",
    }
    if vector_store:
        vector_store_defaults.update(vector_store)
    
    vector_store_config = VectorStoreConfig(
        collection_name=collection_name,
        **vector_store_defaults,
    )
    
    # Retrieval configuration
    retrieval_defaults = {
        "top_k": 10,
        "similarity_threshold": 0.7,
        "enable_reranking": True,
        "rerank_top_k": 5,
    }
    if retrieval:
        retrieval_defaults.update(retrieval)
    
    retrieval_config = RetrievalConfig(**retrieval_defaults)
    
    # Generation configuration
    generation_defaults = {
        "model_name": generation_model,
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    if generation:
        generation_defaults.update(generation)
    
    generation_config = GenerationConfig(**generation_defaults)
    
    # Embedding service
    embedding_service = EmbeddingService(
        model_name=embedding_model,
        api_key=settings.openai.api_key,
    )
    
    # Create pipeline
    pipeline = RAGPipeline(
        vector_store_config=vector_store_config,
        retrieval_config=retrieval_config,
        generation_config=generation_config,
        embedding_service=embedding_service,
        persist_directory=persist_directory,
    )
    
    logger.info(f"Created default RAG pipeline with collection: {collection_name}")
    return pipeline


def create_ergoscript_pipeline(
    persist_directory: Optional[str] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline optimized for ErgoScript code generation.
    
    Args:
        persist_directory: Directory for vector store persistence
        **kwargs: Additional configuration overrides
        
    Returns:
        RAG pipeline optimized for ErgoScript
    """
    # Override defaults for ErgoScript optimization
    ergoscript_overrides = {
        "vector_store": {
            "collection_name": "ergoscript_knowledge",
            **kwargs.get("vector_store", {}),
        },
        "retrieval": {
            "top_k": 8,
            "similarity_threshold": 0.75,
            "enable_reranking": True,
            "rerank_top_k": 5,
            "hybrid_search_alpha": 0.8,  # Favor semantic search for code
            **kwargs.get("retrieval", {}),
        },
        "generation": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,  # Lower temperature for code generation
            "max_tokens": 2048,
            **kwargs.get("generation", {}),
        },
    }
    
    return create_default_rag_pipeline(
        collection_name=ergoscript_overrides["vector_store"]["collection_name"],
        persist_directory=persist_directory,
        embedding_model="text-embedding-3-large",
        retrieval=ergoscript_overrides["retrieval"],
        generation=ergoscript_overrides["generation"],
    )


def create_documentation_pipeline(
    persist_directory: Optional[str] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline optimized for documentation queries.
    
    Args:
        persist_directory: Directory for vector store persistence
        **kwargs: Additional configuration overrides
        
    Returns:
        RAG pipeline optimized for documentation
    """
    documentation_overrides = {
        "vector_store": {
            "collection_name": "ergo_documentation",
            **kwargs.get("vector_store", {}),
        },
        "retrieval": {
            "top_k": 12,
            "similarity_threshold": 0.65,
            "enable_reranking": True,
            "rerank_top_k": 8,
            "hybrid_search_alpha": 0.6,  # Balance semantic and keyword search
            **kwargs.get("retrieval", {}),
        },
        "generation": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.3,  # Slightly higher for explanations
            "max_tokens": 3072,
            **kwargs.get("generation", {}),
        },
    }
    
    return create_default_rag_pipeline(
        collection_name=documentation_overrides["vector_store"]["collection_name"],
        persist_directory=persist_directory,
        embedding_model="text-embedding-3-large",
        retrieval=documentation_overrides["retrieval"],
        generation=documentation_overrides["generation"],
    )


def create_development_pipeline(
    collection_name: str = "dev_ergoscript",
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline for development and testing.
    
    Args:
        collection_name: Name for the collection
        **kwargs: Additional configuration overrides
        
    Returns:
        RAG pipeline configured for development
    """
    dev_overrides = {
        "vector_store": {
            "collection_name": collection_name,
            **kwargs.get("vector_store", {}),
        },
        "retrieval": {
            "top_k": 5,  # Smaller for faster testing
            "similarity_threshold": 0.6,
            "enable_reranking": False,  # Disable for speed
            **kwargs.get("retrieval", {}),
        },
        "generation": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1024,  # Shorter responses for testing
            **kwargs.get("generation", {}),
        },
    }
    
    return create_default_rag_pipeline(
        collection_name=dev_overrides["vector_store"]["collection_name"],
        persist_directory="./data/chroma_dev",
        embedding_model="text-embedding-3-small",  # Faster for development
        retrieval=dev_overrides["retrieval"],
        generation=dev_overrides["generation"],
    )


def create_production_pipeline(
    persist_directory: Optional[str] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline optimized for production use.
    
    Args:
        persist_directory: Directory for vector store persistence
        **kwargs: Additional configuration overrides
        
    Returns:
        RAG pipeline optimized for production
    """
    settings = get_settings()
    
    # Use production settings
    production_overrides = {
        "vector_store": {
            "collection_name": "ergoscript_production",
            **kwargs.get("vector_store", {}),
        },
        "retrieval": {
            "top_k": 10,
            "similarity_threshold": 0.75,
            "enable_reranking": True,
            "rerank_top_k": 6,
            "hybrid_search_alpha": 0.7,
            **kwargs.get("retrieval", {}),
        },
        "generation": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2048,
            **kwargs.get("generation", {}),
        },
    }
    
    return create_default_rag_pipeline(
        persist_directory=persist_directory or "./data/chroma_prod",
        embedding_model="text-embedding-3-large",
        **production_overrides,
    )


def get_pipeline_for_environment(
    environment: Optional[str] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Get a RAG pipeline configured for the current environment.
    
    Args:
        environment: Environment name (development, production, etc.)
        **kwargs: Additional configuration overrides
        
    Returns:
        Environment-appropriate RAG pipeline
    """
    settings = get_settings()
    env = environment or settings.app_environment
    
    if env == "development":
        return create_development_pipeline(**kwargs)
    elif env == "production":
        return create_production_pipeline(**kwargs)
    elif env == "testing":
        return create_development_pipeline(
            collection_name="test_ergoscript",
            **kwargs,
        )
    else:
        # Default to ergoscript pipeline
        return create_ergoscript_pipeline(**kwargs)


# Convenience exports
__all__ = [
    "create_default_rag_pipeline",
    "create_ergoscript_pipeline", 
    "create_documentation_pipeline",
    "create_development_pipeline",
    "create_production_pipeline",
    "get_pipeline_for_environment",
] 