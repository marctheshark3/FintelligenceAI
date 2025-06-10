"""
FintelligenceAI: Intelligent RAG Pipeline & AI Agent System

A comprehensive framework for building modular RAG pipelines and AI agents
using DSPy, specialized for ErgoScript smart contract generation.
"""

from .config import get_settings
from .knowledge import KnowledgeBaseManager, setup_ergoscript_knowledge_base
from .rag import RAGPipeline, create_rag_pipeline

__version__ = "0.1.0"
__author__ = "FintelligenceAI Team"

# Main components for easy access
__all__ = [
    "get_settings",
    "RAGPipeline",
    "create_rag_pipeline",
    "setup_ergoscript_knowledge_base",
    "KnowledgeBaseManager",
]
