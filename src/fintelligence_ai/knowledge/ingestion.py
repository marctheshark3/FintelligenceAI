"""
Knowledge Base Ingestion Module

Manages the ingestion of processed documents into the vector database.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..config import get_settings
from ..rag.embeddings import EmbeddingService
from ..rag.models import Document
from ..rag.vectorstore import VectorStoreManager
from .collector import collect_ergoscript_knowledge_base
from .processor import DocumentChunk, DocumentProcessor, ErgoScriptProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionResult(BaseModel):
    """Result of knowledge base ingestion."""

    success: bool
    documents_processed: int
    chunks_created: int
    chunks_stored: int
    errors: list[str]
    processing_time_seconds: float
    storage_time_seconds: float


class KnowledgeBaseManager:
    """Manages the complete knowledge base lifecycle."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        processor: Optional[DocumentProcessor] = None,
    ):
        from ..config import get_settings
        from ..rag.models import VectorStoreConfig

        # Initialize components with proper dependencies
        if embedding_service is None:
            settings = get_settings()
            self.embedding_service = EmbeddingService(
                model_name="text-embedding-3-large", api_key=settings.openai.api_key
            )
        else:
            self.embedding_service = embedding_service

        if vector_store is None:
            # Create default vector store config
            config = VectorStoreConfig(
                collection_name="ergoscript_examples",
                embedding_dimension=3072,  # text-embedding-3-large dimension
                distance_metric="cosine",
            )

            self.vector_store = VectorStoreManager(
                config=config,
                embedding_service=self.embedding_service,
                persist_directory="./data/chroma",
            )
        else:
            self.vector_store = vector_store

        self.processor = processor or ErgoScriptProcessor()

    async def initialize(self):
        """Initialize the knowledge base manager."""
        # Components are already initialized in __init__
        # This method exists for interface compatibility
        logger.info("Knowledge base manager initialized successfully")

    async def setup_ergoscript_knowledge_base(self) -> IngestionResult:
        """Set up the complete ErgoScript knowledge base."""
        start_time = datetime.utcnow()
        errors = []

        try:
            logger.info("Starting ErgoScript knowledge base setup...")

            # Step 1: Collect documents
            logger.info("Collecting ErgoScript examples...")
            documents = await collect_ergoscript_knowledge_base()

            if not documents:
                raise Exception("No documents collected from ErgoScript repository")

            logger.info(f"Collected {len(documents)} documents")

            # Step 2: Process documents into chunks
            logger.info("Processing documents into chunks...")
            processing_start = datetime.utcnow()
            chunks, processing_stats = self.processor.process_documents(documents)
            processing_time = (datetime.utcnow() - processing_start).total_seconds()

            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            # Step 3: Store chunks in vector database
            logger.info("Storing chunks in vector database...")
            storage_start = datetime.utcnow()
            stored_count = await self._store_chunks(chunks)
            storage_time = (datetime.utcnow() - storage_start).total_seconds()

            total_time = (datetime.utcnow() - start_time).total_seconds()

            result = IngestionResult(
                success=True,
                documents_processed=len(documents),
                chunks_created=len(chunks),
                chunks_stored=stored_count,
                errors=errors,
                processing_time_seconds=processing_time,
                storage_time_seconds=storage_time,
            )

            logger.info(f"Knowledge base setup completed in {total_time:.2f} seconds")
            logger.info(
                f"Processed {result.documents_processed} documents into {result.chunks_created} chunks"
            )
            logger.info(f"Stored {result.chunks_stored} chunks in vector database")

            return result

        except Exception as e:
            error_msg = f"Knowledge base setup failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

            return IngestionResult(
                success=False,
                documents_processed=0,
                chunks_created=0,
                chunks_stored=0,
                errors=errors,
                processing_time_seconds=(
                    datetime.utcnow() - start_time
                ).total_seconds(),
                storage_time_seconds=0,
            )

    async def _store_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Store document chunks in the vector database."""
        stored_count = 0
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            try:
                # Convert chunks to Documents for storage
                documents = []
                for chunk in batch:
                    # Create metadata as a dict first, then convert to DocumentMetadata
                    metadata_dict = {
                        **chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "parent_document_id": chunk.parent_document_id,
                    }

                    # Create a basic DocumentMetadata for the chunk
                    from ..rag.models import DocumentMetadata

                    doc_metadata = DocumentMetadata(
                        source=metadata_dict.get("source", "examples"),
                        category=metadata_dict.get("category", "examples"),
                        complexity=metadata_dict.get("complexity", "beginner"),
                        tags=metadata_dict.get("tags", []),
                        file_path=metadata_dict.get("file_path", ""),
                    )

                    doc = Document(
                        id=chunk.id, content=chunk.content, metadata=doc_metadata
                    )
                    documents.append(doc)

                # Store batch
                self.vector_store.add_documents(documents)

                stored_count += len(documents)
                logger.debug(
                    f"Stored batch {i//batch_size + 1}: {len(documents)} chunks"
                )

            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size + 1}: {e}")

        return stored_count

    async def get_knowledge_base_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        try:
            collection_stats = self.vector_store.get_collection_stats()

            return {
                "collection_name": collection_stats.get(
                    "collection_name", "ergoscript_examples"
                ),
                "document_count": collection_stats.get("document_count", 0),
                "embedding_dimension": collection_stats.get("embedding_dimension", 0),
                "distance_metric": collection_stats.get("distance_metric", "cosine"),
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {}

    async def refresh_knowledge_base(self) -> IngestionResult:
        """Refresh the knowledge base with latest data."""
        logger.info("Refreshing ErgoScript knowledge base...")

        # Clear existing collection
        try:
            self.vector_store.reset_collection()
            logger.info("Cleared existing knowledge base")
        except Exception as e:
            logger.warning(f"Error clearing existing collection: {e}")

        # Rebuild knowledge base
        return await self.setup_ergoscript_knowledge_base()

    async def search_knowledge_base(
        self, query: str, limit: int = 10, filter_metadata: Optional[dict] = None
    ) -> list[dict]:
        """Search the knowledge base."""
        try:
            results = self.vector_store.search_by_text(
                query_text=query, top_k=limit, filters=filter_metadata
            )

            return [
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []


class IngestionPipeline:
    """Pipeline for ingesting various types of data sources."""

    def __init__(self, knowledge_manager: Optional[KnowledgeBaseManager] = None):
        self.knowledge_manager = knowledge_manager or KnowledgeBaseManager()

    async def run_ergoscript_ingestion(self) -> IngestionResult:
        """Run the ErgoScript knowledge base ingestion pipeline."""
        await self.knowledge_manager.initialize()
        return await self.knowledge_manager.setup_ergoscript_knowledge_base()

    async def run_custom_ingestion(
        self, documents: list[Document], collection_name: str = "custom_documents"
    ) -> IngestionResult:
        """Run ingestion for custom documents."""
        start_time = datetime.utcnow()
        errors = []

        try:
            await self.knowledge_manager.initialize()

            # Process documents
            processor = ErgoScriptProcessor()
            chunks, stats = processor.process_documents(documents)

            # Store in custom collection
            stored_count = 0
            for chunk in chunks:
                # Convert chunk metadata to DocumentMetadata if needed
                if isinstance(chunk.metadata, dict):
                    from ..rag.models import DocumentMetadata

                    doc_metadata = DocumentMetadata(
                        source=chunk.metadata.get("source", "examples"),
                        category=chunk.metadata.get("category", "examples"),
                        complexity=chunk.metadata.get("complexity", "beginner"),
                        tags=chunk.metadata.get("tags", []),
                        file_path=chunk.metadata.get("file_path", ""),
                    )
                else:
                    doc_metadata = chunk.metadata

                doc = Document(
                    id=chunk.id, content=chunk.content, metadata=doc_metadata
                )

                try:
                    self.knowledge_manager.vector_store.add_documents([doc])
                    stored_count += 1
                except Exception as e:
                    errors.append(f"Error storing chunk {chunk.id}: {e}")

            return IngestionResult(
                success=True,
                documents_processed=len(documents),
                chunks_created=len(chunks),
                chunks_stored=stored_count,
                errors=errors,
                processing_time_seconds=stats.processing_time_seconds,
                storage_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

        except Exception as e:
            error_msg = f"Custom ingestion failed: {e}"
            errors.append(error_msg)

            return IngestionResult(
                success=False,
                documents_processed=0,
                chunks_created=0,
                chunks_stored=0,
                errors=errors,
                processing_time_seconds=(
                    datetime.utcnow() - start_time
                ).total_seconds(),
                storage_time_seconds=0,
            )


# Convenience functions
async def setup_ergoscript_knowledge_base() -> IngestionResult:
    """Set up the ErgoScript knowledge base."""
    pipeline = IngestionPipeline()
    return await pipeline.run_ergoscript_ingestion()


async def get_knowledge_base_stats() -> dict:
    """Get knowledge base statistics."""
    manager = KnowledgeBaseManager()
    await manager.initialize()
    return await manager.get_knowledge_base_stats()
