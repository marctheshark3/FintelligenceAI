"""
Configuration settings for FintelligenceAI using Pydantic.

This module provides centralized configuration management for the entire application,
supporting environment-specific settings and validation.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = Field(
        default="postgresql+asyncpg://fintelligence_user:fintelligence_pass@localhost:5432/fintelligence_ai",
        description="Database connection URL",
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Database connection pool size")
    max_overflow: int = Field(default=10, description="Maximum pool overflow")

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", case_sensitive=False)

    def get_provider_specific_url(self, model_provider: str) -> str:
        """Get database URL with provider-specific database name."""
        base_url = self.url

        # Extract the base part before the database name
        if "/fintelligence_ai" in base_url:
            base_part = base_url.split("/fintelligence_ai")[0]
            provider_db_name = f"fintelligence_ai_{model_provider}"
            return f"{base_part}/{provider_db_name}"

        # If it doesn't match expected pattern, append provider suffix
        if base_url.endswith("/"):
            return f"{base_url}fintelligence_ai_{model_provider}"
        else:
            return f"{base_url}-{model_provider}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    model_config = SettingsConfigDict(env_prefix="REDIS_", case_sensitive=False)


class ChromaDBSettings(BaseSettings):
    """ChromaDB configuration settings."""

    host: str = Field(default="localhost", description="ChromaDB host")
    port: int = Field(default=8100, description="ChromaDB port")
    persist_directory: str = Field(
        default="./data/chroma", description="ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="fintelligence_ai", description="Default collection name"
    )

    model_config = SettingsConfigDict(env_prefix="CHROMA_", case_sensitive=False)

    def get_provider_specific_collection(self, model_provider: str) -> str:
        """Get collection name with provider-specific suffix."""
        base_name = self.collection_name

        # More robust suffix detection - check for both underscore and hyphen variants
        provider_patterns = [
            f"_{model_provider}",  # underscore variant
            f"-{model_provider}",  # hyphen variant
        ]

        # If collection name already has a provider suffix, use as-is
        for pattern in provider_patterns:
            if base_name.endswith(pattern):
                return base_name

        # Check if base name contains provider name anywhere (legacy handling)
        if f"_{model_provider}" in base_name or f"-{model_provider}" in base_name:
            return base_name

        # Add provider suffix to base collection name (prefer underscore for consistency)
        return f"{base_name}_{model_provider}"

    def get_provider_specific_directory(self, model_provider: str) -> str:
        """Get persist directory with provider-specific subdirectory."""
        base_dir = self.persist_directory.rstrip("/")
        return f"{base_dir}/{model_provider}"


class OpenAISettings(BaseSettings):
    """OpenAI configuration settings."""

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Default OpenAI model")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum tokens for generation"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large", description="Embedding model"
    )

    model_config = SettingsConfigDict(env_prefix="OPENAI_", case_sensitive=False)


class OllamaSettings(BaseSettings):
    """Ollama configuration settings for local-only mode."""

    host: str = Field(default="localhost", description="Ollama server host")
    port: int = Field(default=11434, description="Ollama server port")
    model: str = Field(default="llama3.2", description="Default Ollama model")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum tokens for generation"
    )
    embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama embedding model"
    )
    timeout: int = Field(default=300, description="Request timeout in seconds")
    keep_alive: str = Field(default="5m", description="Keep model alive duration")
    # Connection settings
    base_url: Optional[str] = Field(
        default=None, description="Full base URL (overrides host:port)"
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", case_sensitive=False)

    @property
    def url(self) -> str:
        """Get the full Ollama server URL."""
        if self.base_url:
            return self.base_url
        return f"http://{self.host}:{self.port}"


class DSPySettings(BaseSettings):
    """DSPy configuration settings."""

    model_provider: str = Field(default="openai", description="DSPy model provider")
    local_mode: bool = Field(
        default=False, description="Enable local-only mode (uses Ollama)"
    )
    cache_dir: str = Field(
        default="./data/dspy_cache", description="DSPy cache directory"
    )
    experiment_dir: str = Field(
        default="./data/experiments", description="DSPy experiment directory"
    )
    optimizer: str = Field(default="MIPROv2", description="Default DSPy optimizer")
    training_size: int = Field(default=100, gt=0, description="Training dataset size")
    validation_size: int = Field(
        default=50, gt=0, description="Validation dataset size"
    )
    max_iterations: int = Field(
        default=50, gt=0, description="Maximum optimization iterations"
    )

    model_config = SettingsConfigDict(env_prefix="DSPY_", case_sensitive=False)

    @validator("model_provider")
    def validate_model_provider(cls, v):
        """Validate model provider is supported."""
        valid_providers = ["openai", "ollama", "claude", "cohere"]
        if v not in valid_providers:
            raise ValueError(f"model_provider must be one of: {valid_providers}")
        return v


class ErgoSettings(BaseSettings):
    """Ergo blockchain configuration settings."""

    node_url: str = Field(default="http://localhost:9053", description="Ergo node URL")
    node_api_key: Optional[str] = Field(default=None, description="Ergo node API key")
    explorer_url: str = Field(
        default="https://api.ergoplatform.com", description="Ergo explorer URL"
    )

    model_config = SettingsConfigDict(env_prefix="ERGO_", case_sensitive=False)


class APISettings(BaseSettings):
    """API configuration settings."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    workers: int = Field(default=1, description="Number of worker processes")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS"
    )

    model_config = SettingsConfigDict(env_prefix="API_", case_sensitive=False)


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    jwt_secret_key: Optional[str] = Field(
        default=None, description="JWT secret key for token signing"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(
        default=30, description="JWT expiration time in minutes"
    )
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_calls: int = Field(
        default=100, description="Rate limit calls per period"
    )
    rate_limit_period: int = Field(
        default=60, description="Rate limit period in seconds"
    )

    model_config = SettingsConfigDict(env_prefix="JWT_", case_sensitive=False)


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""

    langchain_tracing_v2: bool = Field(
        default=True, description="Enable LangChain tracing"
    )
    langchain_api_key: Optional[str] = Field(
        default=None, description="LangChain API key"
    )
    langchain_project: str = Field(
        default="fintelligence-ai", description="LangChain project name"
    )
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking"
    )
    sentry_environment: str = Field(
        default="development", description="Sentry environment"
    )
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")

    model_config = SettingsConfigDict(env_prefix="LANGCHAIN_", case_sensitive=False)


class FileSettings(BaseSettings):
    """File processing and storage settings."""

    upload_dir: str = Field(default="./data/uploads", description="Upload directory")
    max_file_size: int = Field(
        default=10485760, description="Maximum file size in bytes"
    )
    allowed_file_types: list[str] = Field(
        default=["pdf", "txt", "md", "json"],
        description="Allowed file types for upload",
    )
    chunk_size: int = Field(
        default=1000, description="Document chunk size for processing"
    )
    chunk_overlap: int = Field(default=200, description="Chunk overlap for processing")
    max_docs_per_query: int = Field(
        default=10, description="Maximum documents per query"
    )

    model_config = SettingsConfigDict(env_prefix="FILE_", case_sensitive=False)


class Settings(BaseSettings):
    """Main application settings."""

    # Application metadata
    app_name: str = Field(default="FintelligenceAI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_environment: str = Field(
        default="development", description="Application environment"
    )
    app_debug: bool = Field(default=True, description="Enable debug mode")
    app_log_level: str = Field(default="INFO", description="Logging level")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    dspy: DSPySettings = Field(default_factory=DSPySettings)
    ergo: ErgoSettings = Field(default_factory=ErgoSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    files: FileSettings = Field(default_factory=FileSettings)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @validator("app_environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production", "testing"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @validator("app_log_level")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_environment == "production"

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.app_environment == "testing"

    def get_provider_database_url(self) -> str:
        """Get database URL with provider-specific database name."""
        return self.database.get_provider_specific_url(self.dspy.model_provider)

    def get_provider_collection_name(self) -> str:
        """Get ChromaDB collection name with provider-specific suffix."""
        return self.chromadb.get_provider_specific_collection(self.dspy.model_provider)

    def get_provider_persist_directory(self) -> str:
        """Get ChromaDB persist directory with provider-specific subdirectory."""
        return self.chromadb.get_provider_specific_directory(self.dspy.model_provider)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function uses LRU cache to ensure settings are loaded only once
    and reused across the application.

    Returns:
        Settings: Application configuration settings
    """
    return Settings()
