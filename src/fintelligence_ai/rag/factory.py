"""
Factory functions for creating optimized RAG pipeline configurations.

This module provides factory functions to create RAG pipelines with different
optimizations for various use cases like EIP documentation, code repositories,
and ErgoScript development.
"""

import logging
from typing import Optional

from ..config.settings import get_settings
from .embeddings import EmbeddingService
from .models import GenerationConfig, RetrievalConfig, VectorStoreConfig
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
        "similarity_threshold": 0.3,  # Lowered from 0.7 to 0.3 for better recall
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


def create_ergoscript_pipeline(collection_name: str = None) -> RAGPipeline:
    """
    Create a pipeline specifically optimized for ErgoScript development.

    This pipeline is tuned for:
    - ErgoScript syntax and semantics
    - Smart contract development
    - Ergo Platform specific features
    - Box model and UTXO handling
    - Integration with Ergo ecosystem

    Args:
        collection_name: Name of the collection to use

    Returns:
        RAGPipeline configured for ErgoScript development
    """
    # Use the existing create_default_rag_pipeline with ErgoScript-specific settings
    ergoscript_retrieval = {
        "top_k": 15,
        "similarity_threshold": 0.4,  # Lower for ErgoScript specificity
        "enable_reranking": True,
        "rerank_top_k": 12,
        "hybrid_search_alpha": 0.3,  # Heavy keyword focus
    }

    ergoscript_generation = {
        "model_name": "gpt-4",
        "temperature": 0.05,  # Very low temperature for precise guidance
        "max_tokens": 2048,
    }

    return create_default_rag_pipeline(
        collection_name=collection_name or "ergoscript_development",
        retrieval=ergoscript_retrieval,
        generation=ergoscript_generation,
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


def create_eip_documentation_pipeline(
    persist_directory: Optional[str] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline optimized for EIP (Ergo Improvement Proposal) queries.

    Args:
        persist_directory: Directory for vector store persistence
        **kwargs: Additional configuration overrides

    Returns:
        RAG pipeline optimized for EIP documentation
    """
    eip_overrides = {
        "vector_store": {
            "collection_name": "eip_documentation",
            **kwargs.get("vector_store", {}),
        },
        "retrieval": {
            "top_k": 15,  # More results for better coverage
            "similarity_threshold": 0.5,  # Lower threshold for technical docs
            "enable_reranking": True,
            "rerank_top_k": 8,
            "hybrid_search_alpha": 0.5,  # Balance semantic and keyword search
            **kwargs.get("retrieval", {}),
        },
        "generation": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.2,  # Slightly higher for explanations
            "max_tokens": 3072,
            **kwargs.get("generation", {}),
        },
    }

    return create_default_rag_pipeline(
        collection_name=eip_overrides["vector_store"]["collection_name"],
        persist_directory=persist_directory,
        embedding_model="text-embedding-3-large",
        retrieval=eip_overrides["retrieval"],
        generation=eip_overrides["generation"],
    )


def create_flexible_documentation_pipeline(
    collection_name: str = None, query_type: str = "general"
) -> RAGPipeline:
    """
    Create a flexible documentation pipeline that adapts based on query type.

    Args:
        collection_name: Name of the collection to use
        query_type: Type of query - "eip", "code", "github", or "general"

    Returns:
        Configured RAGPipeline optimized for the query type
    """
    if query_type == "eip":
        return create_eip_documentation_pipeline()
    elif query_type in ["code", "github"]:
        return create_code_repository_pipeline(collection_name)
    elif query_type == "ergoscript":
        return create_ergoscript_pipeline(collection_name)
    else:
        return create_default_rag_pipeline(collection_name=collection_name)


def create_code_repository_pipeline(collection_name: str = None) -> RAGPipeline:
    """
    Create a pipeline optimized for code repository and GitHub content retrieval.

    This pipeline is specifically tuned for:
    - Code examples and implementations
    - GitHub repository content
    - Programming language specific queries
    - Function/class/method lookups
    - API documentation in code

    Args:
        collection_name: Name of the collection to use

    Returns:
        RAGPipeline configured for code repository retrieval
    """
    # Use the existing create_default_rag_pipeline with code-specific settings
    code_retrieval = {
        "top_k": 20,  # More results for comprehensive code examples
        "similarity_threshold": 0.45,  # Lower threshold for code semantic similarity
        "enable_reranking": True,
        "rerank_top_k": 15,
        "hybrid_search_alpha": 0.4,  # Keyword-favored for exact function/class names
    }

    code_generation = {
        "model_name": "gpt-4",
        "temperature": 0.1,  # Low temperature for precise code explanations
        "max_tokens": 2048,
    }

    return create_default_rag_pipeline(
        collection_name=collection_name or "code_repository",
        retrieval=code_retrieval,
        generation=code_generation,
    )


def create_default_pipeline(collection_name: str = None) -> RAGPipeline:
    """
    Create a default RAG pipeline with standard settings.

    Args:
        collection_name: Name of the collection to use

    Returns:
        RAGPipeline with standard configuration
    """
    return create_default_rag_pipeline(collection_name=collection_name or "default")


# Convenience exports
__all__ = [
    "create_default_rag_pipeline",
    "create_default_pipeline",
    "create_ergoscript_pipeline",
    "create_documentation_pipeline",
    "create_development_pipeline",
    "create_production_pipeline",
    "get_pipeline_for_environment",
    "create_eip_documentation_pipeline",
    "create_flexible_documentation_pipeline",
    "create_code_repository_pipeline",
]
