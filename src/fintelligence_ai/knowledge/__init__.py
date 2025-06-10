"""
Knowledge Base Management Module

Handles data collection, processing, and ingestion for the FintelligenceAI system.
"""

from .collector import GitHubDataCollector, ErgoScriptCollector
from .processor import DocumentProcessor, ErgoScriptProcessor
from .ingestion import (
    KnowledgeBaseManager, 
    IngestionPipeline,
    setup_ergoscript_knowledge_base,
    get_knowledge_base_stats
)

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