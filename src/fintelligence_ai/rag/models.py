"""
Data models for the RAG pipeline components.

This module defines the core data structures used throughout the RAG pipeline
for documents, queries, retrievals, and generation results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class DocumentSource(str, Enum):
    """Enumeration of document sources in the knowledge base."""

    OFFICIAL_DOCS = "official_docs"
    GITHUB = "github"
    EXAMPLES = "examples"
    COMMUNITY = "community"
    TUTORIALS = "tutorials"
    LOCAL_FILES = "local_files"
    WEB_SCRAPING = "web_scraping"


class DocumentCategory(str, Enum):
    """Categorization of document types."""

    SYNTAX = "syntax"
    API = "api"
    EXAMPLES = "examples"
    BEST_PRACTICES = "best_practices"
    PATTERNS = "patterns"
    TUTORIALS = "tutorials"
    REFERENCE = "reference"
    GUIDES = "guides"
    GENERAL = "general"


class ComplexityLevel(str, Enum):
    """Complexity levels for content categorization."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document in the knowledge base."""

    source: DocumentSource
    category: DocumentCategory
    complexity: ComplexityLevel
    tags: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    language: str = "ergoscript"
    tested: bool = False
    performance_notes: Optional[str] = None


class Document(BaseModel):
    """A document in the knowledge base."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    title: Optional[str] = None
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("content")
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class CodeExample(BaseModel):
    """A code example document with ErgoScript-specific fields."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    code: str
    description: str
    title: Optional[str] = None
    use_case: str
    type: str = "snippet"  # complete_contract, snippet, pattern
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("code")
    def code_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Code cannot be empty")
        return v


class QueryIntent(str, Enum):
    """Types of user query intents."""

    CODE_GENERATION = "code_generation"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    BEST_PRACTICES = "best_practices"
    EXAMPLES = "examples"


class Query(BaseModel):
    """A user query to the RAG system."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    intent: Optional[QueryIntent] = None
    context: Optional[dict[str, Any]] = Field(default_factory=dict)
    filters: Optional[dict[str, Any]] = Field(default_factory=dict)
    max_results: int = Field(default=10, ge=1, le=100)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("text")
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query text cannot be empty")
        return v


class RetrievalResult(BaseModel):
    """Result from the retrieval engine."""

    document_id: str
    content: str
    title: Optional[str] = None
    score: float = Field(ge=0.0, le=1.0)
    metadata: DocumentMetadata
    rank: int = Field(ge=1)
    retrieval_method: str = "semantic"  # semantic, keyword, hybrid

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class GenerationContext(BaseModel):
    """Context provided to the generation engine."""

    query: Query
    retrieved_documents: list[RetrievalResult]
    additional_context: Optional[dict[str, Any]] = Field(default_factory=dict)
    generation_params: Optional[dict[str, Any]] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Result from the generation engine."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    generated_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    source_documents: list[str] = Field(default_factory=list)  # document IDs
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("generated_text")
    def generated_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Generated text cannot be empty")
        return v


class ErgoScriptGenerationResult(GenerationResult):
    """Enhanced generation result for ErgoScript code."""

    code: str
    explanation: str
    compilation_status: Optional[str] = None
    validation_errors: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)
    complexity_estimate: Optional[ComplexityLevel] = None


class RAGPipelineResult(BaseModel):
    """Complete result from the RAG pipeline."""

    query_id: str
    query_text: str
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult
    pipeline_metadata: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store operations."""

    collection_name: str
    embedding_dimension: int = 1536  # OpenAI text-embedding-3-large
    distance_metric: str = "cosine"
    index_type: str = "hnsw"
    connection_string: Optional[str] = None


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations."""

    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0
    )  # Lowered from 0.7 to 0.3 for better recall
    enable_reranking: bool = True
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    hybrid_search_alpha: float = Field(
        default=0.7, ge=0.0, le=1.0
    )  # Weight for semantic vs keyword
    include_metadata_filter: bool = True


class GenerationConfig(BaseModel):
    """Configuration for generation operations."""

    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: list[str] = Field(default_factory=list)
