"""
Knowledge Base Management Module

Handles data collection, processing, and ingestion for the FintelligenceAI system.
"""

from .collector import ErgoScriptCollector, GitHubDataCollector
from .ingestion import (
    IngestionPipeline,
    KnowledgeBaseManager,
    get_knowledge_base_stats,
    setup_ergoscript_knowledge_base,
)
from .processor import DocumentProcessor, ErgoScriptProcessor

__all__ = [
    "GitHubDataCollector",
    "ErgoScriptCollector",
    "DocumentProcessor",
    "ErgoScriptProcessor",
    "KnowledgeBaseManager",
    "IngestionPipeline",
    "setup_ergoscript_knowledge_base",
    "get_knowledge_base_stats",
]
