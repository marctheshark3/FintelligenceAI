"""
Document reranker for improving retrieval result quality.

This module provides reranking capabilities to improve the ordering of retrieved
documents based on cross-encoder models and domain-specific scoring.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

from .models import Query, RetrievalConfig, RetrievalResult

logger = logging.getLogger(__name__)


class DocumentReranker:
    """
    Document reranker using cross-encoder models.

    This class reorders retrieved documents based on more sophisticated
    relevance scoring using cross-encoder models that consider the query
    and document content together.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: Optional[RetrievalConfig] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the document reranker.

        Args:
            model_name: Name of the cross-encoder model to use
            config: Retrieval configuration
            batch_size: Batch size for processing documents
        """
        self.model_name = model_name
        self.config = config or RetrievalConfig()
        self.batch_size = batch_size

        # Initialize cross-encoder model
        try:
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}")
            self.cross_encoder = None

    def rerank(
        self,
        query: Query,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results based on cross-encoder scoring.

        Args:
            query: Original query
            results: List of retrieval results to rerank
            top_k: Number of top results to return (uses config if None)

        Returns:
            Reranked list of retrieval results
        """
        if not results:
            return results

        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, returning original ranking")
            return results

        top_k = top_k or self.config.rerank_top_k

        # Limit to top_k for reranking efficiency
        candidates = results[: min(len(results), top_k * 2)]

        logger.debug(f"Reranking {len(candidates)} documents")

        # Prepare query-document pairs
        pairs = [(query.text, result.content) for result in candidates]

        # Get cross-encoder scores
        try:
            scores = self._compute_cross_encoder_scores(pairs)

            # Update result scores and rerank
            reranked_results = self._update_and_sort_results(candidates, scores)

            # Apply domain-specific scoring adjustments
            reranked_results = self._apply_domain_scoring(query, reranked_results)

            # Return top-k results
            final_results = reranked_results[:top_k]

            # Update ranks
            for i, result in enumerate(final_results):
                result.rank = i + 1

            logger.debug(f"Reranking complete, returning {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return results[:top_k]

    def _compute_cross_encoder_scores(
        self, pairs: list[tuple[str, str]]
    ) -> list[float]:
        """Compute cross-encoder scores for query-document pairs."""
        scores = []

        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = self.cross_encoder.predict(batch)

            # Convert to list if numpy array
            if isinstance(batch_scores, np.ndarray):
                batch_scores = batch_scores.tolist()
            elif not isinstance(batch_scores, list):
                batch_scores = [batch_scores]

            scores.extend(batch_scores)

        return scores

    def _update_and_sort_results(
        self,
        results: list[RetrievalResult],
        cross_encoder_scores: list[float],
    ) -> list[RetrievalResult]:
        """Update result scores and sort by cross-encoder scores."""
        # Normalize cross-encoder scores to 0-1 range
        if cross_encoder_scores:
            min_score = min(cross_encoder_scores)
            max_score = max(cross_encoder_scores)
            score_range = max_score - min_score

            if score_range > 0:
                normalized_scores = [
                    (score - min_score) / score_range for score in cross_encoder_scores
                ]
            else:
                normalized_scores = [0.5] * len(cross_encoder_scores)
        else:
            normalized_scores = [result.score for result in results]

        # Update result scores (combine original and cross-encoder scores)
        for result, ce_score in zip(results, normalized_scores):
            # Weighted combination: 30% original score, 70% cross-encoder score
            result.score = 0.3 * result.score + 0.7 * ce_score

        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _apply_domain_scoring(
        self,
        query: Query,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Apply ErgoScript domain-specific scoring adjustments.

        This method applies additional scoring based on ErgoScript-specific
        factors to improve relevance for code generation tasks.
        """
        query_lower = query.text.lower()

        # Define scoring factors
        scoring_factors = {
            "code_keywords": ["contract", "script", "function", "val", "def", "sigma"],
            "security_keywords": ["security", "safe", "secure", "vulnerability"],
            "performance_keywords": ["optimize", "efficient", "performance", "gas"],
            "complexity_preferences": {
                "beginner": ["simple", "basic", "easy", "tutorial"],
                "advanced": ["complex", "advanced", "sophisticated", "optimize"],
            },
        }

        for result in results:
            adjustment = 0.0

            # Boost for code-related content
            if any(
                keyword in query_lower for keyword in scoring_factors["code_keywords"]
            ):
                if result.metadata.category.value in ["examples", "patterns"]:
                    adjustment += 0.1

                # Prefer tested code examples
                if result.metadata.tested:
                    adjustment += 0.05

            # Boost for security-related queries
            if any(
                keyword in query_lower
                for keyword in scoring_factors["security_keywords"]
            ):
                if "security" in result.metadata.tags:
                    adjustment += 0.15

            # Boost for performance-related queries
            if any(
                keyword in query_lower
                for keyword in scoring_factors["performance_keywords"]
            ):
                if result.metadata.performance_notes:
                    adjustment += 0.1

            # Complexity matching
            for complexity, keywords in scoring_factors[
                "complexity_preferences"
            ].items():
                if any(keyword in query_lower for keyword in keywords):
                    if result.metadata.complexity.value == complexity:
                        adjustment += 0.05

            # Source preference (official docs get slight boost)
            if result.metadata.source.value == "official_docs":
                adjustment += 0.02

            # Apply adjustment
            result.score = min(1.0, result.score + adjustment)

        # Re-sort after adjustments
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def batch_rerank(
        self,
        query_result_pairs: list[tuple[Query, list[RetrievalResult]]],
        top_k: Optional[int] = None,
    ) -> list[list[RetrievalResult]]:
        """
        Rerank multiple query-result pairs efficiently.

        Args:
            query_result_pairs: List of (query, results) tuples
            top_k: Number of top results to return for each query

        Returns:
            List of reranked result lists
        """
        reranked_results = []

        for query, results in query_result_pairs:
            reranked = self.rerank(query, results, top_k)
            reranked_results.append(reranked)

        return reranked_results

    def get_relevance_score(self, query_text: str, document_text: str) -> float:
        """
        Get relevance score for a single query-document pair.

        Args:
            query_text: Query text
            document_text: Document text

        Returns:
            Relevance score between 0 and 1
        """
        if not self.cross_encoder:
            return 0.5  # Default neutral score

        try:
            score = self.cross_encoder.predict([(query_text, document_text)])

            if isinstance(score, np.ndarray):
                score = score[0]

            # Normalize to 0-1 range (assuming logits output)
            normalized_score = 1 / (1 + np.exp(-score))  # Sigmoid

            return float(normalized_score)

        except Exception as e:
            logger.error(f"Failed to compute relevance score: {str(e)}")
            return 0.5
