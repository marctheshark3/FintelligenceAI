"""
Retrieval engine for document search and ranking.

This module implements the retrieval component of the RAG pipeline, providing
semantic search, keyword search, and hybrid retrieval strategies.
"""

import logging
import re
import time
from typing import Any, Optional, Union

import dspy
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingService
from .models import Document, Query, QueryIntent, RetrievalConfig, RetrievalResult
from .vectorstore import VectorStoreManager


class RetrievalError(Exception):
    """Exception raised when retrieval operations fail."""

    pass


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

    def forward(
        self, query_or_queries: Union[str, list[str]], k: Optional[int] = None
    ) -> dspy.Prediction:
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
            passages = [result.content for result in results]

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

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for the given query.

        Args:
            query: Query object containing the search text

        Returns:
            List of relevant documents ranked by relevance
        """
        start_time = time.time()

        try:
            # Enhanced query intent detection with priority order
            if self._is_eip_query(query.text):
                logger.info(
                    f"Detected EIP query, using specialized EIP retrieval: {query.text[:100]}..."
                )
                results = self._retrieve_eip_documents(query)
                query_type = "eip"
            elif self._is_ergoscript_query(query.text):
                logger.info(
                    f"Detected ErgoScript query, using specialized ErgoScript retrieval: {query.text[:100]}..."
                )
                results = self._retrieve_ergoscript_documents(query)
                query_type = "ergoscript"
            elif self._is_code_query(query.text):
                logger.info(
                    f"Detected code query, using specialized code retrieval: {query.text[:100]}..."
                )
                results = self._retrieve_code_documents(query)
                query_type = "code"
            else:
                logger.info(f"Using general retrieval for query: {query.text[:100]}...")
                results = self._general_retrieval(query)
                query_type = "general"

            # Apply reranking if enabled and we have multiple results
            if self.config.enable_reranking and len(results) > 1:
                logger.debug(f"Applying reranking to {len(results)} results")
                results = await self._rerank_results(query, results)

            # Final limit to configured top_k
            final_results = results[: self.config.top_k]

            retrieval_time = time.time() - start_time
            logger.info(
                f"Retrieved {len(final_results)} documents in {retrieval_time:.3f}s "
                f"(query_type: {query_type}, total_candidates: {len(results)})"
            )

            return final_results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e

    def retrieve_with_intent(
        self,
        query: Query,
        intent: Optional[QueryIntent] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents based on query intent.

        Args:
            query: Query object
            intent: Detected or specified query intent

        Returns:
            List of retrieval results optimized for the intent
        """
        intent = intent or query.intent or self._detect_intent(query.text)

        # Check for EIP-specific queries and handle specially
        if self._is_eip_query(query.text):
            return self._retrieve_eip_documents(query)

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

        # Use appropriate retrieval method based on strategy
        if strategy == "semantic":
            results = self._semantic_retrieval(query, filters)
        elif strategy == "keyword":
            results = self._keyword_retrieval(query, filters)
        else:  # hybrid
            results = self._hybrid_retrieval(query, filters)

        # Apply intent-specific post-processing
        return self._post_process_by_intent(results, intent)

    def _semantic_retrieval(
        self,
        query: Query,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Perform semantic search using vector similarity."""
        results = self.vector_store.search_similar(
            query,
            top_k=self.config.top_k,
            filters=filters,
        )

        # Filter by similarity threshold
        filtered_results = [
            result
            for result in results
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
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
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
        top_indices = scores.argsort()[-self.config.top_k :][::-1]

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
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
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
        return combined_results[: self.config.top_k]

    def _combine_results(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        alpha: float = 0.7,
    ) -> list[RetrievalResult]:
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

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for keyword search."""
        # Simple tokenization - could be enhanced with proper NLP
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]

    def _matches_filters(self, document: Document, filters: dict[str, Any]) -> bool:
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

        # EIP and standards queries
        if any(
            pattern in query_lower
            for pattern in [
                "eip",
                "improvement proposal",
                "standard",
                "token standard",
                "nft standard",
                "eip-",
                "eip ",
                "what is the eip",
            ]
        ):
            return QueryIntent.DOCUMENTATION

        # Code generation patterns
        if any(
            pattern in query_lower
            for pattern in [
                "create",
                "generate",
                "write",
                "implement",
                "build",
                "contract",
                "smart contract",
                "ergoscript",
            ]
        ):
            return QueryIntent.CODE_GENERATION

        # Examples and tutorials
        if any(
            pattern in query_lower
            for pattern in ["example", "tutorial", "how to", "show me", "demo"]
        ):
            return QueryIntent.EXAMPLES

        # Best practices
        if any(
            pattern in query_lower
            for pattern in [
                "best practice",
                "recommend",
                "should i",
                "pattern",
                "security",
                "optimize",
            ]
        ):
            return QueryIntent.BEST_PRACTICES

        # Documentation queries
        if any(
            pattern in query_lower
            for pattern in [
                "what is",
                "explain",
                "describe",
                "definition",
                "documentation",
                "api",
                "reference",
            ]
        ):
            return QueryIntent.DOCUMENTATION

        # Default to general
        return QueryIntent.DOCUMENTATION

    def _is_eip_query(self, query_text: str) -> bool:
        """
        Determine if a query is specifically asking about EIPs.

        Args:
            query_text: The user's query string

        Returns:
            True if the query appears to be EIP-related
        """
        query_lower = query_text.lower()

        # Direct EIP indicators
        eip_indicators = [
            "eip",
            "eip-",
            "improvement proposal",
            "ergo improvement",
            "standard",
            "specification",
            "protocol",
        ]

        # EIP-specific terms
        eip_terms = [
            "token standard",
            "collection",
            "nft standard",
            "wallet api",
            "stealth address",
            "payment request",
            "asset standard",
        ]

        # Check for direct indicators
        for indicator in eip_indicators:
            if indicator in query_lower:
                return True

        # Check for EIP-specific terms combined with question words
        question_words = ["what", "which", "how", "explain", "describe", "define"]
        if any(qw in query_lower for qw in question_words):
            if any(term in query_lower for term in eip_terms):
                return True

        return False

    def _is_code_query(self, query: str) -> bool:
        """
        Determine if a query is specifically asking about code, implementation, or GitHub content.

        Args:
            query: The user's query string

        Returns:
            True if the query appears to be code/GitHub-related
        """
        query_lower = query.lower()

        # Code implementation indicators
        code_indicators = [
            "implement",
            "implementation",
            "code",
            "function",
            "method",
            "class",
            "api",
            "library",
            "framework",
            "package",
            "module",
            "import",
            "github",
            "repository",
            "repo",
            "source",
            "example",
            "sample",
        ]

        # Programming language indicators
        language_indicators = [
            "python",
            "javascript",
            "js",
            "ergoscript",
            "scala",
            "java",
            "typescript",
            "react",
            "node",
            "npm",
            "pip",
            "cargo",
        ]

        # Development/technical terms
        dev_terms = [
            "debug",
            "error",
            "bug",
            "compile",
            "build",
            "test",
            "unit test",
            "integration",
            "deployment",
            "configuration",
            "setup",
            "install",
            "dependency",
            "version",
            "update",
            "documentation",
            "docs",
            "tutorial",
            "guide",
            "how to",
            "step by step",
        ]

        # Ergo-specific development terms
        ergo_dev_terms = [
            "box",
            "utxo",
            "transaction",
            "smart contract",
            "dapp",
            "appkit",
            "sigma",
            "sigmastate",
            "prover",
            "verifier",
            "address",
            "wallet",
        ]

        # Check for direct code indicators
        for indicator in code_indicators:
            if indicator in query_lower:
                return True

        # Check for programming language mentions
        for lang in language_indicators:
            if lang in query_lower:
                return True

        # Check for development terms with question context
        question_words = ["how", "what", "where", "when", "why", "show", "demonstrate"]
        if any(qw in query_lower for qw in question_words):
            if any(term in query_lower for term in dev_terms + ergo_dev_terms):
                return True

        return False

    def _is_ergoscript_query(self, query: str) -> bool:
        """
        Determine if a query is specifically about ErgoScript development.

        Args:
            query: The user's query string

        Returns:
            True if the query appears to be ErgoScript-specific
        """
        query_lower = query.lower()

        # ErgoScript specific indicators
        ergoscript_indicators = [
            "ergoscript",
            "ergo script",
            "sigma",
            "sigmastate",
            "sigmaprop",
            "box",
            "utxo",
            "registers",
            "context",
            "self",
            "inputs",
            "outputs",
        ]

        # ErgoScript functions and operations
        ergoscript_functions = [
            "alltrue",
            "anytrue",
            "blake2b256",
            "deserialize",
            "serialize",
            "provelog",
            "proveddlog",
            "getvar",
            "extract",
            "fold",
            "forall",
            "exists",
            "atmost",
            "byteand",
            "byteor",
            "bytexor",
        ]

        # Smart contract terms
        contract_terms = [
            "smart contract",
            "contract",
            "guard script",
            "spending condition",
            "emission",
            "token minting",
            "burning",
            "oracle",
            "pool",
        ]

        # Check for ErgoScript indicators
        for indicator in ergoscript_indicators:
            if indicator in query_lower:
                return True

        # Check for ErgoScript functions
        for func in ergoscript_functions:
            if func in query_lower:
                return True

        # Check for smart contract terms
        for term in contract_terms:
            if term in query_lower:
                return True

        return False

    def _retrieve_eip_documents(self, query: Query) -> list[RetrievalResult]:
        """
        Retrieve documents with EIP-optimized settings.

        Args:
            query: The query object

        Returns:
            List of retrieval results optimized for EIP content
        """
        # Use more lenient similarity threshold for technical documentation
        results = self.vector_store.similarity_search(
            query_text=query.text,
            k=15,  # Get more results initially
            similarity_threshold=0.45,  # Lower threshold for EIP docs
            alpha=0.4,  # Keyword-favored search for EIP terms
        )

        # Post-process to boost EIP-relevant results
        processed_results = self._post_process_eip_results(results, query.text)

        # Re-rank and limit results
        return processed_results[: self.config.top_k]

    def _retrieve_code_documents(self, query: Query) -> list[RetrievalResult]:
        """
        Retrieve documents with code/GitHub-optimized settings.

        Args:
            query: The query object

        Returns:
            List of retrieval results optimized for code content
        """
        # Use code-optimized similarity threshold
        results = self.vector_store.similarity_search(
            query_text=query.text,
            k=20,  # Get more results for comprehensive code examples
            similarity_threshold=0.45,  # Lower threshold for code semantic similarity
            alpha=0.4,  # Keyword-favored for exact function/class names
        )

        # Post-process to boost code-relevant results
        processed_results = self._post_process_code_results(results, query.text)

        # Re-rank and limit results
        return processed_results[: self.config.top_k]

    def _retrieve_ergoscript_documents(self, query: Query) -> list[RetrievalResult]:
        """
        Retrieve documents with ErgoScript-optimized settings.

        Args:
            query: The query object

        Returns:
            List of retrieval results optimized for ErgoScript content
        """
        # Use ErgoScript-optimized similarity threshold
        results = self.vector_store.similarity_search(
            query_text=query.text,
            k=15,  # Focused results for ErgoScript
            similarity_threshold=0.4,  # Even lower for ErgoScript specificity
            alpha=0.3,  # Heavy keyword focus for ErgoScript terms
        )

        # Post-process to boost ErgoScript-relevant results
        processed_results = self._post_process_ergoscript_results(results, query.text)

        # Re-rank and limit results
        return processed_results[: self.config.top_k]

    def _post_process_eip_results(
        self, results: list[RetrievalResult], query_text: str
    ) -> list[RetrievalResult]:
        """
        Post-process results to boost EIP-relevant content.

        Args:
            results: Raw retrieval results
            query_text: Original query text

        Returns:
            Processed and re-ranked results for EIP queries
        """
        query_lower = query_text.lower()

        # Boost scores for documents that appear to be EIP-related
        for result in results:
            content_lower = result.content.lower()
            title_lower = (result.title or "").lower()

            # Major boost for documents that contain EIP indicators
            eip_indicators = [
                "eip-",
                "improvement proposal",
                "ergo improvement",
                "standard",
                "specification",
            ]
            if any(
                indicator in content_lower or indicator in title_lower
                for indicator in eip_indicators
            ):
                result.score = min(1.0, result.score * 1.4)

            # Additional boost for EIP-specific terms that match the query
            eip_terms = [
                "token",
                "collection",
                "nft",
                "wallet",
                "api",
                "stealth",
                "address",
                "payment",
                "request",
                "asset",
            ]
            matching_terms = sum(
                1 for term in eip_terms if term in query_lower and term in content_lower
            )

            if matching_terms > 0:
                boost_factor = 1.0 + (matching_terms * 0.15)
                result.score = min(1.0, result.score * boost_factor)

        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _post_process_code_results(
        self, results: list[RetrievalResult], query_text: str
    ) -> list[RetrievalResult]:
        """
        Post-process results for code queries.

        Args:
            results: Raw retrieval results
            query_text: Original query text

        Returns:
            Processed and re-ranked results for code queries
        """
        query_lower = query_text.lower()

        # Boost scores for documents that appear to be code-related
        for result in results:
            content_lower = result.content.lower()
            title_lower = (result.title or "").lower()

            # Boost if content contains code indicators
            code_indicators = [
                "code",
                "function",
                "method",
                "class",
                "api",
                "library",
                "framework",
                "package",
                "module",
                "import",
                "github",
                "repository",
                "repo",
                "source",
                "example",
                "sample",
            ]
            if any(
                indicator in content_lower or indicator in title_lower
                for indicator in code_indicators
            ):
                result.score = min(1.0, result.score * 1.3)

            # Boost if content contains programming language terms
            lang_terms = [
                "python",
                "javascript",
                "js",
                "ergoscript",
                "scala",
                "java",
                "typescript",
                "react",
                "node",
                "npm",
                "pip",
                "cargo",
            ]
            matching_terms = sum(
                1
                for term in lang_terms
                if term in query_lower and term in content_lower
            )

            if matching_terms > 0:
                boost_factor = 1.0 + (matching_terms * 0.1)
                result.score = min(1.0, result.score * boost_factor)

        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _post_process_ergoscript_results(
        self, results: list[RetrievalResult], query_text: str
    ) -> list[RetrievalResult]:
        """
        Post-process results for ErgoScript queries.

        Args:
            results: Raw retrieval results
            query_text: Original query text

        Returns:
            Processed and re-ranked results for ErgoScript queries
        """
        query_lower = query_text.lower()

        # Boost scores for documents that appear to be ErgoScript-related
        for result in results:
            content_lower = result.content.lower()
            title_lower = (result.title or "").lower()

            # Boost if content contains ErgoScript indicators
            ergoscript_indicators = [
                "ergoscript",
                "ergo script",
                "sigma",
                "sigmastate",
                "sigmaprop",
                "box",
                "utxo",
                "registers",
                "context",
                "self",
                "inputs",
                "outputs",
            ]
            if any(
                indicator in content_lower or indicator in title_lower
                for indicator in ergoscript_indicators
            ):
                result.score = min(1.0, result.score * 1.3)

            # Boost if content contains ErgoScript function terms
            func_terms = [
                "alltrue",
                "anytrue",
                "blake2b256",
                "deserialize",
                "serialize",
                "provelog",
                "proveddlog",
                "getvar",
                "extract",
                "fold",
                "forall",
                "exists",
                "atmost",
                "byteand",
                "byteor",
                "bytexor",
            ]
            matching_terms = sum(
                1
                for term in func_terms
                if term in query_lower and term in content_lower
            )

            if matching_terms > 0:
                boost_factor = 1.0 + (matching_terms * 0.15)
                result.score = min(1.0, result.score * boost_factor)

        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _post_process_general_results(
        self, results: list[RetrievalResult], query_text: str
    ) -> list[RetrievalResult]:
        """
        Post-process results for general queries.

        Args:
            results: Raw retrieval results
            query_text: Original query text

        Returns:
            Processed results for general queries
        """
        # For general queries, we apply minimal processing
        # Just ensure proper ranking is maintained
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _post_process_by_intent(
        self,
        results: list[RetrievalResult],
        intent: QueryIntent,
    ) -> list[RetrievalResult]:
        """Apply intent-specific post-processing to results."""
        if intent == QueryIntent.CODE_GENERATION:
            # Prioritize tested and complete examples
            results.sort(
                key=lambda x: (
                    x.metadata.tested,
                    x.metadata.complexity.value == "beginner",
                    x.score,
                ),
                reverse=True,
            )

        elif intent == QueryIntent.EXAMPLES:
            # Prioritize examples with clear descriptions
            results.sort(
                key=lambda x: (
                    len(x.content) > 100,  # Prefer substantial examples
                    x.metadata.category.value == "examples",
                    x.score,
                ),
                reverse=True,
            )

        return results

    def _general_retrieval(self, query: Query) -> list[RetrievalResult]:
        """
        General retrieval method for non-specialized queries.

        Args:
            query: The query object

        Returns:
            List of retrieval results using standard settings
        """
        # Use standard similarity search with current config
        results = self.vector_store.similarity_search(
            query_text=query.text,
            k=self.config.top_k * 2,  # Get more results initially
            similarity_threshold=self.config.similarity_threshold,
            alpha=0.6,  # Balanced search for general queries
        )

        # Apply general post-processing
        processed_results = self._post_process_general_results(results, query.text)

        return processed_results
