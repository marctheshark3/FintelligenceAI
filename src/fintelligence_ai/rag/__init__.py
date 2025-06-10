"""
FintelligenceAI RAG Pipeline Module

This module provides the core RAG (Retrieval-Augmented Generation) pipeline
implementation using DSPy for ErgoScript generation and intelligent document retrieval.
"""

from .pipeline import RAGPipeline
from .retrieval import RetrievalEngine, DocumentRetriever
from .generation import GenerationEngine, ErgoScriptGenerator
from .reranker import DocumentReranker
from .embeddings import EmbeddingService
from .vectorstore import VectorStoreManager
from .factory import (
    create_default_rag_pipeline,
    create_ergoscript_pipeline,
    create_documentation_pipeline,
    create_development_pipeline,
    create_production_pipeline,
    get_pipeline_for_environment,
)

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
