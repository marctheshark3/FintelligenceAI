"""
Retrieval engine for document search and ranking.

This module implements the retrieval component of the RAG pipeline, providing
semantic search, keyword search, and hybrid retrieval strategies.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import dspy
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingService
from .models import (
    Query, 
    RetrievalResult, 
    RetrievalConfig,
    Document,
    QueryIntent,
)
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentRetriever(dspy.Retrieve):
    """
    DSPy-compatible document retriever for the RAG pipeline.
    
    This class implements the DSPy Retrieve interface to integrate seamlessly
    with DSPy modules while providing advanced retrieval capabilities.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: RetrievalConfig,
        k: int = 10,
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Vector store manager for similarity search
            config: Configuration for retrieval operations
            k: Default number of documents to retrieve
        """
        super().__init__(k=k)
        self.vector_store = vector_store
        self.config = config
        self.k = k
        
        logger.info(f"Initialized DocumentRetriever with k={k}")
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """
        Retrieve documents for the given query or queries.
        
        This method implements the DSPy Retrieve interface.
        
        Args:
            query_or_queries: Single query string or list of query strings
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            DSPy Prediction containing retrieved passages
        """
        k = k or self.k
        
        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries
        
        all_passages = []
        
        for query_text in queries:
            # Create query object
            query = Query(text=query_text, max_results=k)
            
            # Retrieve documents
            results = self.vector_store.search_similar(query, top_k=k)
            
            # Convert to passages for DSPy
            passages = [
                result.content for result in results
            ]
            
            all_passages.extend(passages)
        
        # Return DSPy prediction
        return dspy.Prediction(passages=all_passages)


class RetrievalEngine:
    """
    Advanced retrieval engine supporting multiple search strategies.
    
    This engine provides semantic search, keyword search, and hybrid retrieval
    with configurable ranking and filtering capabilities.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: RetrievalConfig,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_store: Vector store manager for similarity search
            config: Configuration for retrieval operations
            embedding_service: Service for generating query embeddings
        """
        self.vector_store = vector_store
        self.config = config
        self.embedding_service = embedding_service or vector_store.embedding_service
        
        # Initialize BM25 for keyword search (will be populated when needed)
        self._bm25_index = None
        self._bm25_documents = []
        self._bm25_doc_ids = []
        
        logger.info("Initialized RetrievalEngine")
    
    def retrieve(
        self,
        query: Query,
        strategy: str = "semantic",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using the specified strategy.
        
        Args:
            query: Query object with text and parameters
            strategy: Retrieval strategy ("semantic", "keyword", "hybrid")
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results ranked by relevance
        """
        logger.debug(f"Retrieving documents with strategy: {strategy}")
        
        if strategy == "semantic":
            return self._semantic_retrieval(query, filters)
        elif strategy == "keyword":
            return self._keyword_retrieval(query, filters)
        elif strategy == "hybrid":
            return self._hybrid_retrieval(query, filters)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def retrieve_with_intent(
        self,
        query: Query,
        intent: Optional[QueryIntent] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents based on query intent.
        
        Args:
            query: Query object
            intent: Detected or specified query intent
            
        Returns:
            List of retrieval results optimized for the intent
        """
        intent = intent or query.intent or self._detect_intent(query.text)
        
        # Adjust retrieval strategy based on intent
        if intent == QueryIntent.CODE_GENERATION:
            # Prefer code examples and patterns
            filters = {"category": "examples", "type": "complete_contract"}
            strategy = "hybrid"
        elif intent == QueryIntent.DOCUMENTATION:
            # Prefer official documentation
            filters = {"source": "official_docs", "category": "api"}
            strategy = "semantic"
        elif intent == QueryIntent.EXAMPLES:
            # Focus on examples and tutorials
            filters = {"category": "examples"}
            strategy = "semantic"
        elif intent == QueryIntent.BEST_PRACTICES:
            # Look for best practices and patterns
            filters = {"category": "best_practices"}
            strategy = "semantic"
        else:
            # Default strategy
            filters = None
            strategy = "hybrid"
        
        results = self.retrieve(query, strategy, filters)
        
        # Apply intent-specific post-processing
        return self._post_process_by_intent(results, intent)
    
    def _semantic_retrieval(
        self,
        query: Query,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform semantic search using vector similarity."""
        results = self.vector_store.search_similar(
            query,
            top_k=self.config.top_k,
            filters=filters,
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result.score >= self.config.similarity_threshold
        ]
        
        # Update retrieval method
        for result in filtered_results:
            result.retrieval_method = "semantic"
        
        logger.debug(f"Semantic retrieval returned {len(filtered_results)} results")
        return filtered_results
    
    def _keyword_retrieval(
        self,
        query: Query,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform keyword search using BM25."""
        # Ensure BM25 index is built
        self._ensure_bm25_index()
        
        if not self._bm25_index:
            logger.warning("BM25 index not available, falling back to semantic search")
            return self._semantic_retrieval(query, filters)
        
        # Tokenize query
        query_tokens = self._tokenize(query.text)
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top-k documents
        top_indices = scores.argsort()[-self.config.top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self._bm25_doc_ids) and scores[idx] > 0:
                doc_id = self._bm25_doc_ids[idx]
                document = self.vector_store.get_document(doc_id)
                
                if document:
                    # Apply filters if specified
                    if filters and not self._matches_filters(document, filters):
                        continue
                    
                    # Normalize BM25 score to 0-1 range (approximate)
                    normalized_score = min(1.0, scores[idx] / 10.0)
                    
                    result = RetrievalResult(
                        document_id=doc_id,
                        content=document.content,
                        title=document.title,
                        score=normalized_score,
                        metadata=document.metadata,
                        rank=rank + 1,
                        retrieval_method="keyword",
                    )
                    results.append(result)
        
        logger.debug(f"Keyword retrieval returned {len(results)} results")
        return results
    
    def _hybrid_retrieval(
        self,
        query: Query,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining semantic and keyword search."""
        # Get results from both methods
        semantic_results = self._semantic_retrieval(query, filters)
        keyword_results = self._keyword_retrieval(query, filters)
        
        # Combine and re-rank results
        combined_results = self._combine_results(
            semantic_results,
            keyword_results,
            alpha=self.config.hybrid_search_alpha,
        )
        
        # Update retrieval method
        for result in combined_results:
            result.retrieval_method = "hybrid"
        
        logger.debug(f"Hybrid retrieval returned {len(combined_results)} results")
        return combined_results[:self.config.top_k]
    
    def _combine_results(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        alpha: float = 0.7,
    ) -> List[RetrievalResult]:
        """
        Combine semantic and keyword search results.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            alpha: Weight for semantic scores (1-alpha for keyword scores)
            
        Returns:
            Combined and re-ranked results
        """
        # Create document score mapping
        doc_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            doc_scores[result.document_id] = {
                "semantic": result.score * alpha,
                "keyword": 0.0,
                "result": result,
            }
        
        # Add keyword scores
        for result in keyword_results:
            if result.document_id in doc_scores:
                doc_scores[result.document_id]["keyword"] = result.score * (1 - alpha)
            else:
                doc_scores[result.document_id] = {
                    "semantic": 0.0,
                    "keyword": result.score * (1 - alpha),
                    "result": result,
                }
        
        # Calculate combined scores and sort
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = scores["semantic"] + scores["keyword"]
            result = scores["result"]
            result.score = combined_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(combined_results):
            result.rank = rank + 1
        
        return combined_results
    
    def _ensure_bm25_index(self):
        """Ensure BM25 index is built from vector store documents."""
        if self._bm25_index is not None:
            return
        
        logger.info("Building BM25 index from vector store")
        
        try:
            # Get all documents from vector store
            stats = self.vector_store.get_collection_stats()
            if stats.get("document_count", 0) == 0:
                logger.warning("No documents in vector store for BM25 indexing")
                return
            
            # Note: This is a simplified approach. In production, you'd want
            # to iterate through all documents more efficiently
            documents = []
            doc_ids = []
            
            # For now, we'll build the index when first needed
            # This could be optimized by maintaining the index as documents are added
            
            if documents:
                tokenized_docs = [self._tokenize(doc) for doc in documents]
                self._bm25_index = BM25Okapi(tokenized_docs)
                self._bm25_documents = documents
                self._bm25_doc_ids = doc_ids
                
                logger.info(f"Built BM25 index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {str(e)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for keyword search."""
        # Simple tokenization - could be enhanced with proper NLP
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]
    
    def _matches_filters(self, document: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches the specified filters."""
        for key, value in filters.items():
            if key == "source" and document.metadata.source.value != value:
                return False
            elif key == "category" and document.metadata.category.value != value:
                return False
            elif key == "complexity" and document.metadata.complexity.value != value:
                return False
            elif key == "tested" and document.metadata.tested != bool(value):
                return False
            elif key == "tags" and value not in document.metadata.tags:
                return False
        return True
    
    def _detect_intent(self, query_text: str) -> QueryIntent:
        """
        Detect query intent from text.
        
        This is a simple rule-based approach that could be enhanced
        with ML-based intent classification.
        """
        query_lower = query_text.lower()
        
        # Code generation keywords
        if any(keyword in query_lower for keyword in [
            "generate", "create", "write", "implement", "build", "code"
        ]):
            return QueryIntent.CODE_GENERATION
        
        # Documentation keywords
        elif any(keyword in query_lower for keyword in [
            "how to", "documentation", "guide", "manual", "reference"
        ]):
            return QueryIntent.DOCUMENTATION
        
        # Example keywords
        elif any(keyword in query_lower for keyword in [
            "example", "sample", "demo", "show me", "tutorial"
        ]):
            return QueryIntent.EXAMPLES
        
        # Best practices keywords
        elif any(keyword in query_lower for keyword in [
            "best practice", "recommended", "should", "pattern", "convention"
        ]):
            return QueryIntent.BEST_PRACTICES
        
        # Debugging keywords
        elif any(keyword in query_lower for keyword in [
            "error", "debug", "fix", "problem", "issue", "troubleshoot"
        ]):
            return QueryIntent.DEBUGGING
        
        # Default to documentation
        return QueryIntent.DOCUMENTATION
    
    def _post_process_by_intent(
        self,
        results: List[RetrievalResult],
        intent: QueryIntent,
    ) -> List[RetrievalResult]:
        """Apply intent-specific post-processing to results."""
        if intent == QueryIntent.CODE_GENERATION:
            # Prioritize tested and complete examples
            results.sort(key=lambda x: (
                x.metadata.tested,
                x.metadata.complexity.value == "beginner",
                x.score
            ), reverse=True)
        
        elif intent == QueryIntent.EXAMPLES:
            # Prioritize examples with clear descriptions
            results.sort(key=lambda x: (
                len(x.content) > 100,  # Prefer substantial examples
                x.metadata.category.value == "examples",
                x.score
            ), reverse=True)
        
        return results 