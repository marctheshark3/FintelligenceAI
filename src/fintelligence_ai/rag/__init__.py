"""
FintelligenceAI RAG Pipeline Module

This module provides the core RAG (Retrieval-Augmented Generation) pipeline
implementation using DSPy for ErgoScript generation and intelligent document retrieval.
"""

from .embeddings import EmbeddingService
from .factory import (
    create_default_rag_pipeline,
    create_development_pipeline,
    create_documentation_pipeline,
    create_ergoscript_pipeline,
    create_production_pipeline,
    get_pipeline_for_environment,
)
from .generation import ErgoScriptGenerator, GenerationEngine
from .pipeline import RAGPipeline
from .reranker import DocumentReranker
from .retrieval import DocumentRetriever, RetrievalEngine
from .vectorstore import VectorStoreManager


# Convenience function
def create_rag_pipeline(environment: str = "development", **kwargs) -> RAGPipeline:
    """
    Create a RAG pipeline for the specified environment.

    Args:
        environment: Environment name (development, production, ergoscript)
        **kwargs: Additional configuration overrides

    Returns:
        Configured RAG pipeline
    """
    return get_pipeline_for_environment(environment, **kwargs)


__all__ = [
    "RAGPipeline",
    "RetrievalEngine",
    "DocumentRetriever",
    "GenerationEngine",
    "ErgoScriptGenerator",
    "DocumentReranker",
    "EmbeddingService",
    "VectorStoreManager",
    "create_rag_pipeline",
    "create_default_rag_pipeline",
    "create_ergoscript_pipeline",
    "create_documentation_pipeline",
    "create_development_pipeline",
    "create_production_pipeline",
    "get_pipeline_for_environment",
]
