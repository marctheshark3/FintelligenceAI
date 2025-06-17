"""
Ollama integration for local-only LLM usage in FintelligenceAI.

This module provides DSPy-compatible interfaces for using Ollama models
in a local-only environment, eliminating the need for external API calls.
"""

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from ..config import get_settings


class OllamaResponse(BaseModel):
    """Response model for Ollama API calls."""

    model: str
    response: str
    done: bool = True
    context: Optional[list[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaEmbeddingResponse(BaseModel):
    """Response model for Ollama embedding API calls."""

    embedding: list[float]


class OllamaClient:
    """
    Ollama client for making API calls to local Ollama server.

    This client handles communication with the Ollama server and provides
    methods for text generation and embeddings.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
        verify_ssl: bool = True,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama server
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama.url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(__name__)

        # Remove trailing slash
        self.base_url = self.base_url.rstrip("/")

        # Create HTTP client
        self.client = httpx.Client(timeout=self.timeout, verify=self.verify_ssl)
        self.async_client = httpx.AsyncClient(
            timeout=self.timeout, verify=self.verify_ssl
        )

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Ollama server not available: {e}")
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models on Ollama server."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    def generate(
        self, model: str, prompt: str, stream: bool = False, **kwargs
    ) -> OllamaResponse:
        """
        Generate text using Ollama model.

        Args:
            model: Model name to use
            prompt: Input prompt
            stream: Whether to stream response
            **kwargs: Additional model parameters

        Returns:
            Ollama response with generated text
        """
        payload = {"model": model, "prompt": prompt, "stream": stream, "options": {}}

        # Map common parameters
        if "temperature" in kwargs:
            payload["options"]["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            payload["options"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["options"]["top_k"] = kwargs["top_k"]
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]

        try:
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()
            return OllamaResponse(**data)

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    async def agenerate(
        self, model: str, prompt: str, stream: bool = False, **kwargs
    ) -> OllamaResponse:
        """Async version of generate method."""
        payload = {"model": model, "prompt": prompt, "stream": stream, "options": {}}

        # Map common parameters
        if "temperature" in kwargs:
            payload["options"]["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            payload["options"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["options"]["top_k"] = kwargs["top_k"]
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]

        try:
            response = await self.async_client.post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()

            data = response.json()
            return OllamaResponse(**data)

        except Exception as e:
            self.logger.error(f"Async generation failed: {e}")
            raise

    def embed(self, model: str, prompt: str) -> OllamaEmbeddingResponse:
        """
        Generate embeddings using Ollama model.

        Args:
            model: Embedding model name
            prompt: Text to embed

        Returns:
            Embedding response
        """
        payload = {"model": model, "prompt": prompt}

        try:
            response = self.client.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()

            data = response.json()
            return OllamaEmbeddingResponse(**data)

        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            raise

    async def aembed(self, model: str, prompt: str) -> OllamaEmbeddingResponse:
        """Async version of embed method."""
        payload = {"model": model, "prompt": prompt}

        try:
            response = await self.async_client.post(
                f"{self.base_url}/api/embeddings", json=payload
            )
            response.raise_for_status()

            data = response.json()
            return OllamaEmbeddingResponse(**data)

        except Exception as e:
            self.logger.error(f"Async embedding failed: {e}")
            raise

    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        payload = {"name": model}

        try:
            response = self.client.post(f"{self.base_url}/api/pull", json=payload)
            response.raise_for_status()
            return True

        except Exception as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            return False

    def delete_model(self, model: str) -> bool:
        """
        Delete a model from Ollama.

        Args:
            model: Model name to delete

        Returns:
            True if successful
        """
        try:
            response = self.client.delete(f"{self.base_url}/api/delete")
            payload = {"name": model}
            response = self.client.request(
                "DELETE", f"{self.base_url}/api/delete", json=payload
            )
            response.raise_for_status()
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model}: {e}")
            return False

    def __del__(self):
        """Cleanup HTTP clients."""
        try:
            self.client.close()
        except:
            pass


class OllamaDSPy:
    """
    DSPy-compatible Ollama language model.

    This class provides a DSPy-compatible interface for using Ollama models
    within the DSPy framework.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        top_k: int = 50,
        keep_alive: str = "5m",
        **kwargs,
    ):
        """
        Initialize Ollama DSPy model.

        Args:
            model: Ollama model name
            base_url: Ollama server base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            keep_alive: Keep model alive duration
            **kwargs: Additional parameters
        """
        self.settings = get_settings()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.keep_alive = keep_alive

        # Initialize Ollama client
        self.client = OllamaClient(
            base_url=base_url,
            timeout=kwargs.get("timeout", 300),
            verify_ssl=kwargs.get("verify_ssl", True),
        )

        self.logger = logging.getLogger(__name__)

        # Check if server is available
        if not self.client.is_available():
            raise ConnectionError(
                f"Ollama server not available at {self.client.base_url}. "
                "Please ensure Ollama server is running."
            )

        # Check if model is available
        available_models = self.client.list_models()
        model_names = [m.get("name", "").split(":")[0] for m in available_models]

        if model not in model_names and model.split(":")[0] not in model_names:
            self.logger.warning(
                f"Model {model} not found. Available models: {model_names}. "
                f"Attempting to pull model..."
            )
            if not self.client.pull_model(model):
                raise ValueError(f"Model {model} not available and could not be pulled")

    def basic_request(self, prompt: str, **kwargs) -> str:
        """
        Make a basic request to Ollama model.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Merge kwargs with instance defaults
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "keep_alive": self.keep_alive,
            **kwargs,
        }

        try:
            response = self.client.generate(model=self.model, prompt=prompt, **params)
            return response.response

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def __call__(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        """Call the model with a prompt or messages."""
        if messages:
            # Convert messages to a single prompt
            prompt = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                ]
            )
        elif prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        return self.basic_request(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text for DSPy compatibility."""
        return self.basic_request(prompt, **kwargs)


class OllamaEmbedding:
    """
    Ollama embedding service for local embeddings.

    Provides embedding generation using local Ollama models,
    compatible with the existing embedding service interface.
    """

    def __init__(
        self, model: str = "nomic-embed-text", base_url: Optional[str] = None, **kwargs
    ):
        """
        Initialize Ollama embedding service.

        Args:
            model: Embedding model name
            base_url: Ollama server base URL
            **kwargs: Additional parameters
        """
        self.settings = get_settings()
        self.model = model

        # Initialize Ollama client
        self.client = OllamaClient(
            base_url=base_url,
            timeout=kwargs.get("timeout", 300),
            verify_ssl=kwargs.get("verify_ssl", True),
        )

        self.logger = logging.getLogger(__name__)

        # Check if server is available
        if not self.client.is_available():
            raise ConnectionError(
                f"Ollama server not available at {self.client.base_url}"
            )

        # Check if embedding model is available
        available_models = self.client.list_models()
        model_names = [m.get("name", "").split(":")[0] for m in available_models]

        if model not in model_names and model.split(":")[0] not in model_names:
            self.logger.warning(
                f"Embedding model {model} not found, attempting to pull..."
            )
            if not self.client.pull_model(model):
                raise ValueError(f"Embedding model {model} not available")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embed(model=self.model, prompt=text)
            return response.embedding
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    async def aembed_text(self, text: str) -> list[float]:
        """Async version of embed_text."""
        try:
            response = await self.client.aembed(model=self.model, prompt=text)
            return response.embedding
        except Exception as e:
            self.logger.error(f"Async embedding failed: {e}")
            raise

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_texts."""
        import asyncio

        tasks = [self.aembed_text(text) for text in texts]
        return await asyncio.gather(*tasks)


def get_ollama_dspy_model(model: Optional[str] = None, **kwargs) -> OllamaDSPy:
    """
    Factory function to create Ollama DSPy model with settings.

    Args:
        model: Model name (uses settings default if None)
        **kwargs: Additional model parameters

    Returns:
        Configured OllamaDSPy instance
    """
    settings = get_settings()

    model = model or settings.ollama.model

    # Merge settings with kwargs, giving priority to kwargs
    config = {
        "model": model,
        "base_url": settings.ollama.url,
        "temperature": settings.ollama.temperature,
        "max_tokens": settings.ollama.max_tokens,
        "keep_alive": settings.ollama.keep_alive,
        "timeout": settings.ollama.timeout,
        "verify_ssl": settings.ollama.verify_ssl,
    }
    config.update(kwargs)

    return OllamaDSPy(**config)


def get_ollama_embedding_service(
    model: Optional[str] = None, **kwargs
) -> OllamaEmbedding:
    """
    Factory function to create Ollama embedding service with settings.

    Args:
        model: Embedding model name (uses settings default if None)
        **kwargs: Additional parameters

    Returns:
        Configured OllamaEmbedding instance
    """
    settings = get_settings()

    model = model or settings.ollama.embedding_model

    return OllamaEmbedding(
        model=model,
        base_url=settings.ollama.url,
        timeout=settings.ollama.timeout,
        verify_ssl=settings.ollama.verify_ssl,
        **kwargs,
    )
