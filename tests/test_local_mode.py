"""
Test suite for local-only mode functionality using Ollama.

This module tests the Ollama integration and local-only mode configuration
to ensure FintelligenceAI can operate without external API dependencies.
"""

import os
from unittest.mock import Mock, patch

import pytest
from fintelligence_ai.config import get_settings
from fintelligence_ai.core.ollama import (
    OllamaClient,
    OllamaDSPy,
    OllamaEmbedding,
    get_ollama_dspy_model,
    get_ollama_embedding_service,
)


class TestOllamaClient:
    """Test cases for OllamaClient."""

    @pytest.fixture
    def client(self):
        """Create an OllamaClient instance for testing."""
        return OllamaClient(
            base_url="http://localhost:11434", timeout=30, verify_ssl=False
        )

    def test_client_initialization(self, client):
        """Test OllamaClient initialization."""
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30
        assert not client.verify_ssl

    @patch("httpx.Client")
    def test_is_available_success(self, mock_client_class, client):
        """Test successful availability check."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Re-initialize client to use mocked httpx
        client = OllamaClient()
        client.client = mock_client

        assert client.is_available() is True
        mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")

    @patch("httpx.Client")
    def test_is_available_failure(self, mock_client_class, client):
        """Test failed availability check."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        client = OllamaClient()
        client.client = mock_client

        assert client.is_available() is False

    @patch("httpx.Client")
    def test_list_models(self, mock_client_class, client):
        """Test listing available models."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2", "size": 4000000000},
                {"name": "nomic-embed-text", "size": 500000000},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = OllamaClient()
        client.client = mock_client

        models = client.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "llama3.2"
        assert models[1]["name"] == "nomic-embed-text"

    @patch("httpx.Client")
    def test_generate(self, mock_client_class, client):
        """Test text generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "response": "Hello! How can I help you today?",
            "done": True,
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = OllamaClient()
        client.client = mock_client

        response = client.generate("llama3.2", "Hello")
        assert response.model == "llama3.2"
        assert response.response == "Hello! How can I help you today?"
        assert response.done is True


class TestOllamaDSPy:
    """Test cases for OllamaDSPy DSPy integration."""

    @pytest.fixture
    def ollama_dspy(self):
        """Create an OllamaDSPy instance for testing."""
        return OllamaDSPy(
            model="llama3.2", base_url="http://localhost:11434", temperature=0.1
        )

    def test_initialization(self, ollama_dspy):
        """Test OllamaDSPy initialization."""
        assert ollama_dspy.model == "llama3.2"
        assert ollama_dspy.temperature == 0.1
        assert ollama_dspy.max_tokens == 4096

    @patch.object(OllamaClient, "generate")
    def test_basic_request(self, mock_generate, ollama_dspy):
        """Test basic request method."""
        mock_response = Mock()
        mock_response.response = "Test response"
        mock_generate.return_value = mock_response

        response = ollama_dspy.basic_request("Test prompt")
        assert response == "Test response"
        mock_generate.assert_called_once()


class TestOllamaEmbedding:
    """Test cases for OllamaEmbedding service."""

    @pytest.fixture
    def embedding_service(self):
        """Create an OllamaEmbedding instance for testing."""
        return OllamaEmbedding(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )

    def test_initialization(self, embedding_service):
        """Test OllamaEmbedding initialization."""
        assert embedding_service.model == "nomic-embed-text"

    @patch.object(OllamaClient, "embed")
    def test_embed_text(self, mock_embed, embedding_service):
        """Test text embedding."""
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embed.return_value = mock_response

        embedding = embedding_service.embed_text("Test text")
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embed.assert_called_once_with(model="nomic-embed-text", prompt="Test text")

    @patch.object(OllamaClient, "embed")
    def test_embed_texts(self, mock_embed, embedding_service):
        """Test multiple text embeddings."""
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_embed.return_value = mock_response

        texts = ["Text 1", "Text 2"]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 2
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)
        assert mock_embed.call_count == 2


class TestLocalModeConfiguration:
    """Test cases for local mode configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        settings = get_settings()

        # Default should be OpenAI mode
        assert settings.dspy.local_mode is False
        assert settings.dspy.model_provider == "openai"

        # Check Ollama settings exist
        assert hasattr(settings, "ollama")
        assert settings.ollama.host == "localhost"
        assert settings.ollama.port == 11434
        assert settings.ollama.model == "llama3.2"
        assert settings.ollama.embedding_model == "nomic-embed-text"

    @patch.dict(
        os.environ,
        {
            "DSPY_LOCAL_MODE": "true",
            "DSPY_MODEL_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3.1",
            "OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large",
        },
    )
    def test_local_mode_configuration(self):
        """Test local mode configuration from environment."""
        # Clear settings cache to force reload
        get_settings.cache_clear()

        settings = get_settings()

        assert settings.dspy.local_mode is True
        assert settings.dspy.model_provider == "ollama"
        assert settings.ollama.model == "llama3.1"
        assert settings.ollama.embedding_model == "mxbai-embed-large"


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_get_ollama_dspy_model(self):
        """Test DSPy model factory function."""
        model = get_ollama_dspy_model(model="llama3.2", temperature=0.2)

        assert isinstance(model, OllamaDSPy)
        assert model.model == "llama3.2"
        assert model.temperature == 0.2

    def test_get_ollama_embedding_service(self):
        """Test embedding service factory function."""
        service = get_ollama_embedding_service(model="nomic-embed-text")

        assert isinstance(service, OllamaEmbedding)
        assert service.model == "nomic-embed-text"


class TestIntegration:
    """Integration tests for local mode components."""

    @patch.dict(
        os.environ, {"DSPY_LOCAL_MODE": "true", "DSPY_MODEL_PROVIDER": "ollama"}
    )
    @patch("fintelligence_ai.core.ollama.OllamaClient.is_available")
    def test_local_mode_availability_check(self, mock_available):
        """Test that local mode properly checks Ollama availability."""
        mock_available.return_value = True

        # Clear settings cache
        get_settings.cache_clear()

        client = OllamaClient()
        assert client.is_available() is True

    def test_model_parameter_mapping(self):
        """Test that model parameters are properly mapped."""
        model = OllamaDSPy(
            model="llama3.2", temperature=0.5, max_tokens=2048, top_p=0.9, top_k=40
        )

        assert model.temperature == 0.5
        assert model.max_tokens == 2048
        assert model.top_p == 0.9
        assert model.top_k == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
