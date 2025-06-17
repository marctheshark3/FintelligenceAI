"""
Embedding generation service for FintelligenceAI.

This module provides embedding generation capabilities for the RAG pipeline,
using various embedding models, with primary support for OpenAI's embedding models
and local Ollama models for offline operation.
"""

import asyncio
import logging
from typing import Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from .models import Document, Query

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI or other embedding models.

    This service handles the generation of embeddings for documents and queries
    in the RAG pipeline, with support for batch processing and async operations.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use
            api_key: OpenAI API key (uses settings if not provided)
            dimensions: Number of embedding dimensions (uses model default if not provided)
            batch_size: Maximum number of texts to process in a single batch
        """
        self.model_name = model_name
        settings = get_settings()
        self.api_key = api_key or settings.openai.api_key
        self.dimensions = dimensions
        self.batch_size = batch_size

        # Initialize OpenAI client
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        else:
            raise ValueError("OpenAI API key is required for embedding service")

        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-large": {"max_tokens": 8192, "default_dimensions": 3072},
            "text-embedding-3-small": {"max_tokens": 8192, "default_dimensions": 1536},
            "text-embedding-ada-002": {"max_tokens": 8192, "default_dimensions": 1536},
        }

        # Set dimensions based on model if not specified
        if not self.dimensions and model_name in self.model_configs:
            self.dimensions = self.model_configs[model_name]["default_dimensions"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: Text content to embed

        Returns:
            List of float values representing the embedding vector

        Raises:
            Exception: If embedding generation fails after retries
        """
        try:
            # Clean and prepare text
            cleaned_text = self._prepare_text(text)

            # Generate embedding
            kwargs = {"model": self.model_name, "input": cleaned_text}
            if self.dimensions and self.model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)

            embedding = response.data[0].embedding
            logger.debug(
                f"Generated embedding of dimension {len(embedding)} for text length {len(text)}"
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def generate_embedding_async(self, text: str) -> list[float]:
        """
        Asynchronously generate an embedding for a single text string.

        Args:
            text: Text content to embed

        Returns:
            List of float values representing the embedding vector
        """
        try:
            cleaned_text = self._prepare_text(text)

            kwargs = {"model": self.model_name, "input": cleaned_text}
            if self.dimensions and self.model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = await self.async_client.embeddings.create(**kwargs)

            embedding = response.data[0].embedding
            logger.debug(f"Generated async embedding of dimension {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate async embedding: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(embeddings)} embeddings for {len(texts)} texts")
        return embeddings

    async def generate_embeddings_batch_async(
        self, texts: list[str]
    ) -> list[list[float]]:
        """
        Asynchronously generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        if not texts:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

        async def embed_with_semaphore(text: str) -> list[float]:
            async with semaphore:
                return await self.generate_embedding_async(text)

        # Create tasks for all texts
        tasks = [embed_with_semaphore(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)

        logger.info(
            f"Generated {len(embeddings)} async embeddings for {len(texts)} texts"
        )
        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a single batch of texts.

        Args:
            texts: List of text strings to embed (within batch size limit)

        Returns:
            List of embedding vectors
        """
        try:
            # Clean and prepare texts
            cleaned_texts = [self._prepare_text(text) for text in texts]

            kwargs = {"model": self.model_name, "input": cleaned_texts}
            if self.dimensions and self.model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)

            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated batch of {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise

    def _prepare_text(self, text: str) -> str:
        """
        Clean and prepare text for embedding generation.

        Args:
            text: Raw text content

        Returns:
            Cleaned text ready for embedding
        """
        if not text:
            return ""

        # Basic text cleaning
        cleaned = text.strip()

        # Remove excessive whitespace
        cleaned = " ".join(cleaned.split())

        # Truncate if too long (conservative estimate for token limits)
        max_chars = (
            self.model_configs.get(self.model_name, {}).get("max_tokens", 8192) * 3
        )
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            logger.warning(
                f"Text truncated from {len(text)} to {len(cleaned)} characters"
            )

        return cleaned

    def embed_document(self, document: Document) -> Document:
        """
        Generate and attach embedding to a document.

        Args:
            document: Document to embed

        Returns:
            Document with embedding attached
        """
        # Combine title and content for embedding
        text_to_embed = document.title or ""
        if text_to_embed and document.content:
            text_to_embed += " " + document.content
        else:
            text_to_embed = document.content

        embedding = self.generate_embedding(text_to_embed)
        document.embedding = embedding

        return document

    async def embed_document_async(self, document: Document) -> Document:
        """
        Asynchronously generate and attach embedding to a document.

        Args:
            document: Document to embed

        Returns:
            Document with embedding attached
        """
        # Combine title and content for embedding
        text_to_embed = document.title or ""
        if text_to_embed and document.content:
            text_to_embed += " " + document.content
        else:
            text_to_embed = document.content

        embedding = await self.generate_embedding_async(text_to_embed)
        document.embedding = embedding

        return document

    def embed_query(self, query: Query) -> Query:
        """
        Generate and attach embedding to a query.

        Args:
            query: Query to embed

        Returns:
            Query with embedding attached
        """
        embedding = self.generate_embedding(query.text)
        query.embedding = embedding

        return query

    async def embed_query_async(self, query: Query) -> Query:
        """
        Asynchronously generate and attach embedding to a query.

        Args:
            query: Query to embed

        Returns:
            Query with embedding attached
        """
        embedding = await self.generate_embedding_async(query.text)
        query.embedding = embedding

        return query

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this service."""
        return self.dimensions or self.model_configs.get(self.model_name, {}).get(
            "default_dimensions", 1536
        )


class LocalEmbeddingService:
    """
    Local embedding service that supports both OpenAI and Ollama embeddings.

    This service automatically chooses between OpenAI and Ollama based on
    the application configuration and local mode settings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the local embedding service.

        Args:
            model_name: Name of the embedding model (auto-detected if None)
            api_key: OpenAI API key (ignored for Ollama)
            dimensions: Number of embedding dimensions
            batch_size: Maximum number of texts to process in a single batch
        """
        self.settings = get_settings()
        self.batch_size = batch_size
        self.dimensions = dimensions

        # Determine which embedding provider to use
        if (
            self.settings.dspy.local_mode
            or self.settings.dspy.model_provider == "ollama"
        ):
            self._use_ollama = True
            self._model_name = model_name or self.settings.ollama.embedding_model

            # Initialize Ollama embedding service
            from ..core.ollama import get_ollama_embedding_service

            try:
                self._ollama_service = get_ollama_embedding_service(
                    model=self._model_name
                )
                logger.info(
                    f"✅ Local embedding service initialized with Ollama: {self._model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama embeddings: {e}")
                logger.info("Falling back to OpenAI embeddings...")
                self._use_ollama = False
        else:
            self._use_ollama = False

        if not self._use_ollama:
            # Initialize OpenAI embedding service
            self._model_name = model_name or self.settings.openai.embedding_model
            try:
                self._openai_service = EmbeddingService(
                    model_name=self._model_name,
                    api_key=api_key,
                    dimensions=dimensions,
                    batch_size=batch_size,
                )
                logger.info(
                    f"✅ Local embedding service initialized with OpenAI: {self._model_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise ValueError("No embedding service available")

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def is_local(self) -> bool:
        """Check if using local (Ollama) embeddings."""
        return self._use_ollama

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: Text content to embed

        Returns:
            List of float values representing the embedding vector
        """
        if self._use_ollama:
            return self._ollama_service.embed_text(text)
        else:
            return self._openai_service.generate_embedding(text)

    async def generate_embedding_async(self, text: str) -> list[float]:
        """
        Asynchronously generate an embedding for a single text string.

        Args:
            text: Text content to embed

        Returns:
            List of float values representing the embedding vector
        """
        if self._use_ollama:
            return await self._ollama_service.aembed_text(text)
        else:
            return await self._openai_service.generate_embedding_async(text)

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        if not texts:
            return []

        if self._use_ollama:
            return self._ollama_service.embed_texts(texts)
        else:
            return self._openai_service.generate_embeddings_batch(texts)

    async def generate_embeddings_batch_async(
        self, texts: list[str]
    ) -> list[list[float]]:
        """
        Asynchronously generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        if not texts:
            return []

        if self._use_ollama:
            return await self._ollama_service.aembed_texts(texts)
        else:
            return await self._openai_service.generate_embeddings_batch_async(texts)

    def embed_document(self, document: Document) -> Document:
        """
        Generate embeddings for a document and update it in place.

        Args:
            document: Document to embed

        Returns:
            Document with embeddings added
        """
        try:
            # Combine title and content for embedding
            text_to_embed = (
                f"{document.title}\n\n{document.content}"
                if document.title
                else document.content
            )

            embedding = self.generate_embedding(text_to_embed)
            document.embedding = embedding

            logger.debug(f"Generated embedding for document: {document.id}")
            return document

        except Exception as e:
            logger.error(f"Failed to embed document {document.id}: {e}")
            raise

    async def embed_document_async(self, document: Document) -> Document:
        """
        Asynchronously generate embeddings for a document.

        Args:
            document: Document to embed

        Returns:
            Document with embeddings added
        """
        try:
            text_to_embed = (
                f"{document.title}\n\n{document.content}"
                if document.title
                else document.content
            )

            embedding = await self.generate_embedding_async(text_to_embed)
            document.embedding = embedding

            logger.debug(f"Generated async embedding for document: {document.id}")
            return document

        except Exception as e:
            logger.error(f"Failed to async embed document {document.id}: {e}")
            raise

    def embed_query(self, query: Query) -> Query:
        """
        Generate embeddings for a query.

        Args:
            query: Query to embed

        Returns:
            Query with embeddings added
        """
        try:
            embedding = self.generate_embedding(query.text)
            query.embedding = embedding

            logger.debug(f"Generated embedding for query: {query.text[:50]}...")
            return query

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    async def embed_query_async(self, query: Query) -> Query:
        """
        Asynchronously generate embeddings for a query.

        Args:
            query: Query to embed

        Returns:
            Query with embeddings added
        """
        try:
            embedding = await self.generate_embedding_async(query.text)
            query.embedding = embedding

            logger.debug(f"Generated async embedding for query: {query.text[:50]}...")
            return query

        except Exception as e:
            logger.error(f"Failed to async embed query: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Number of dimensions in the embedding vectors
        """
        if self.dimensions:
            return self.dimensions

        if self._use_ollama:
            # For Ollama, we need to generate a test embedding to get dimensions
            # This is cached so it only happens once
            if not hasattr(self, "_cached_dimensions"):
                test_embedding = self.generate_embedding("test")
                self._cached_dimensions = len(test_embedding)
            return self._cached_dimensions
        else:
            return self._openai_service.get_embedding_dimension()


def get_embedding_service(
    model_name: Optional[str] = None, **kwargs
) -> LocalEmbeddingService:
    """
    Factory function to create an embedding service based on current settings.

    Args:
        model_name: Model name (auto-detected based on settings if None)
        **kwargs: Additional parameters for the embedding service

    Returns:
        Configured LocalEmbeddingService instance
    """
    return LocalEmbeddingService(model_name=model_name, **kwargs)
