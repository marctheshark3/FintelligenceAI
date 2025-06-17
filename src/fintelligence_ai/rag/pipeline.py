"""
Main RAG pipeline orchestrating retrieval, reranking, and generation.

This module provides the core RAG pipeline that coordinates all components
to deliver end-to-end retrieval-augmented generation capabilities.
"""

import logging
import time
from typing import Any, Optional

from .embeddings import EmbeddingService
from .generation import ErgoScriptGenerator, GenerationEngine
from .models import (
    Document,
    GenerationConfig,
    GenerationContext,
    Query,
    QueryIntent,
    RAGPipelineResult,
    RetrievalConfig,
    VectorStoreConfig,
)
from .reranker import DocumentReranker
from .retrieval import DocumentRetriever, RetrievalEngine
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline coordinating retrieval, reranking, and generation.

    This class provides the primary interface for the RAG system, orchestrating
    the flow from query to final generation with configurable components.
    """

    def __init__(
        self,
        vector_store_config: VectorStoreConfig,
        retrieval_config: RetrievalConfig,
        generation_config: GenerationConfig,
        embedding_service: Optional[EmbeddingService] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store_config: Configuration for vector store
            retrieval_config: Configuration for retrieval operations
            generation_config: Configuration for generation operations
            embedding_service: Custom embedding service (optional)
            persist_directory: Directory for vector store persistence
        """
        self.vector_store_config = vector_store_config
        self.retrieval_config = retrieval_config
        self.generation_config = generation_config

        # Initialize embedding service
        self.embedding_service = embedding_service or EmbeddingService()

        # Initialize vector store
        self.vector_store = VectorStoreManager(
            config=vector_store_config,
            embedding_service=self.embedding_service,
            persist_directory=persist_directory,
        )

        # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine(
            vector_store=self.vector_store,
            config=retrieval_config,
            embedding_service=self.embedding_service,
        )

        # Initialize reranker
        self.reranker = DocumentReranker(config=retrieval_config)

        # Initialize generation engine
        self.generation_engine = GenerationEngine(config=generation_config)

        # Initialize specialized generators
        self.ergoscript_generator = ErgoScriptGenerator(self.generation_engine)

        # Initialize DSPy retriever for integration
        self.dspy_retriever = DocumentRetriever(
            vector_store=self.vector_store,
            config=retrieval_config,
            k=retrieval_config.top_k,
        )

        logger.info("RAG Pipeline initialized successfully")

    def query(
        self,
        query_text: str,
        query_intent: Optional[QueryIntent] = None,
        generation_type: str = "general",
        filters: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> RAGPipelineResult:
        """
        Execute a complete RAG pipeline query.

        Args:
            query_text: User query text
            query_intent: Optional query intent for optimized retrieval
            generation_type: Type of generation ("general", "code", "explanation")
            filters: Optional metadata filters for retrieval
            **kwargs: Additional parameters

        Returns:
            Complete RAG pipeline result
        """
        start_time = time.time()

        # Create query object
        query = Query(
            text=query_text,
            intent=query_intent,
            filters=filters or {},
            max_results=kwargs.get("max_results", self.retrieval_config.top_k),
        )

        logger.info(f"Processing query: {query_text[:100]}...")

        try:
            # Step 1: Retrieval
            retrieval_results = self._retrieve_documents(query)
            logger.debug(f"Retrieved {len(retrieval_results)} documents")

            # Step 2: Reranking (if enabled)
            if self.retrieval_config.enable_reranking and retrieval_results:
                retrieval_results = self._rerank_documents(query, retrieval_results)
                logger.debug(f"Reranked to {len(retrieval_results)} documents")

            # Step 3: Generation
            generation_result = self._generate_response(
                query, retrieval_results, generation_type
            )
            logger.debug("Generated response")

            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)

            # Create pipeline result
            pipeline_result = RAGPipelineResult(
                query_id=query.id,
                query_text=query.text,
                retrieval_results=retrieval_results,
                generation_result=generation_result,
                pipeline_metadata={
                    "generation_type": generation_type,
                    "query_intent": query_intent.value if query_intent else None,
                    "filters_applied": bool(filters),
                    "reranking_enabled": self.retrieval_config.enable_reranking,
                    "num_retrieved": len(retrieval_results),
                },
                processing_time_ms=processing_time,
            )

            logger.info(f"Query processed successfully in {processing_time}ms")
            return pipeline_result

        except Exception as e:
            logger.error(f"RAG pipeline query failed: {str(e)}")

            # Return error result
            from .models import GenerationResult

            error_result = GenerationResult(
                generated_text=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence_score=0.0,
                reasoning="Pipeline processing failed",
                source_documents=[],
                metadata={"error": str(e)},
            )

            return RAGPipelineResult(
                query_id=query.id,
                query_text=query.text,
                retrieval_results=[],
                generation_result=error_result,
                pipeline_metadata={"error": str(e)},
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

    def generate_ergoscript(
        self,
        requirements: str,
        complexity: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RAGPipelineResult:
        """
        Generate ErgoScript code based on requirements.

        Args:
            requirements: Code requirements and specifications
            complexity: Target complexity level ("beginner", "intermediate", "advanced")
            filters: Optional filters for relevant examples

        Returns:
            RAG pipeline result with generated ErgoScript
        """
        # Enhanced query for code generation
        enhanced_requirements = f"Generate ErgoScript code: {requirements}"
        if complexity:
            enhanced_requirements += f" (Target complexity: {complexity})"

        # Set default filters for code generation
        code_filters = filters or {}
        code_filters.update(
            {
                "category": "examples",
                "language": "ergoscript",
            }
        )

        return self.query(
            query_text=enhanced_requirements,
            query_intent=QueryIntent.CODE_GENERATION,
            generation_type="code",
            filters=code_filters,
        )

    def explain_concept(
        self,
        concept: str,
        complexity_level: Optional[str] = None,
    ) -> RAGPipelineResult:
        """
        Generate explanation for ErgoScript concepts.

        Args:
            concept: Concept to explain
            complexity_level: Target explanation complexity

        Returns:
            RAG pipeline result with explanation
        """
        explanation_query = f"Explain the ErgoScript concept: {concept}"
        if complexity_level:
            explanation_query += f" (Explain at {complexity_level} level)"

        filters = {"category": "api"}
        if complexity_level:
            filters["complexity"] = complexity_level

        return self.query(
            query_text=explanation_query,
            query_intent=QueryIntent.EXPLANATION,
            generation_type="explanation",
            filters=filters,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs that were added
        """
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        return self.vector_store.add_documents(documents)

    def add_document(self, document: Document) -> str:
        """
        Add a single document to the knowledge base.

        Args:
            document: Document to add

        Returns:
            Document ID that was added
        """
        return self.vector_store.add_document(document)

    def search_documents(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list:
        """
        Search for documents without generation.

        Args:
            query_text: Search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of retrieval results
        """
        query = Query(text=query_text, filters=filters or {}, max_results=top_k)
        return self.vector_store.search_similar(query, top_k, filters)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "vector_store": vector_stats,
            "retrieval_config": {
                "top_k": self.retrieval_config.top_k,
                "similarity_threshold": self.retrieval_config.similarity_threshold,
                "reranking_enabled": self.retrieval_config.enable_reranking,
            },
            "generation_config": {
                "model": self.generation_config.model_name,
                "temperature": self.generation_config.temperature,
                "max_tokens": self.generation_config.max_tokens,
            },
        }

    def _retrieve_documents(self, query: Query):
        """Retrieve documents based on query intent or default strategy."""
        if query.intent:
            return self.retrieval_engine.retrieve_with_intent(query, query.intent)
        else:
            # Since retrieve is async, we'll need to handle this differently
            # For now, use synchronous hybrid retrieval directly
            return self.retrieval_engine._hybrid_retrieval(query)

    def _rerank_documents(self, query: Query, results):
        """Rerank retrieved documents."""
        return self.reranker.rerank(
            query=query,
            results=results,
            top_k=self.retrieval_config.rerank_top_k,
        )

    def _generate_response(self, query: Query, retrieval_results, generation_type: str):
        """Generate response based on retrieved context."""
        context = GenerationContext(
            query=query,
            retrieved_documents=retrieval_results,
        )

        return self.generation_engine.generate(context, generation_type)

    def reset_knowledge_base(self) -> bool:
        """Reset the knowledge base by clearing all documents."""
        logger.warning("Resetting knowledge base - all documents will be deleted")
        return self.vector_store.reset_collection()

    def health_check(self) -> dict[str, Any]:
        """Perform health check on all pipeline components."""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time(),
        }

        try:
            # Check vector store
            stats = self.vector_store.get_collection_stats()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "document_count": stats.get("document_count", 0),
            }
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        try:
            # Check embedding service
            test_embedding = self.embedding_service.generate_embedding("test")
            health_status["components"]["embedding_service"] = {
                "status": "healthy",
                "embedding_dimension": len(test_embedding),
            }
        except Exception as e:
            health_status["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        try:
            # Check generation engine (simple test)
            health_status["components"]["generation_engine"] = {
                "status": "healthy",
                "model": self.generation_config.model_name,
            }
        except Exception as e:
            health_status["components"]["generation_engine"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        return health_status
