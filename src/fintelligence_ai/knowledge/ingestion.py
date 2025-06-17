"""
Knowledge Base Ingestion Module

Manages the ingestion of processed documents into the vector database.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..config import get_settings
from ..rag.embeddings import LocalEmbeddingService
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
        embedding_service: Optional[LocalEmbeddingService] = None,
        processor: Optional[DocumentProcessor] = None,
    ):
        from ..rag.models import VectorStoreConfig

        # Initialize components with proper dependencies
        if embedding_service is None:
            from ..rag.embeddings import get_embedding_service

            self.embedding_service = get_embedding_service()
            logger.info(
                f"✅ Initialized embedding service: {self.embedding_service.model_name} (local: {self.embedding_service.is_local})"
            )
        else:
            self.embedding_service = embedding_service

        if vector_store is None:
            # Get embedding dimension from the service
            embedding_dim = self.embedding_service.get_embedding_dimension()

            # Get provider-specific configuration
            from ..config import get_settings

            settings = get_settings()
            collection_name = settings.get_provider_collection_name()
            persist_directory = settings.get_provider_persist_directory()

            # Create provider-specific vector store config
            config = VectorStoreConfig(
                collection_name=collection_name,
                embedding_dimension=embedding_dim,
                distance_metric="cosine",
            )

            self.vector_store = VectorStoreManager(
                config=config,
                embedding_service=self.embedding_service,
                persist_directory=persist_directory,
            )
            logger.info(
                f"✅ Initialized vector store '{collection_name}' with {embedding_dim} dimensions"
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
        """Get comprehensive statistics about the knowledge base."""
        try:
            collection_stats = self.vector_store.get_collection_stats()

            # Get basic collection info
            collection_name = collection_stats.get(
                "collection_name", self.vector_store.config.collection_name
            )
            document_count = collection_stats.get("document_count", 0)

            # Calculate storage size (rough estimate based on document count)
            # This is a simplified calculation - in production you might want more accurate sizing
            storage_size_mb = document_count * 0.01  # Rough estimate: 10KB per document

            # Get available categories by querying the vector store for unique metadata values
            available_categories = []
            try:
                # Try to get some sample documents to extract categories
                if document_count > 0:
                    sample_results = self.vector_store.search_by_text(
                        "", top_k=min(100, document_count)
                    )
                    categories_set = set()
                    for result in sample_results:
                        if (
                            hasattr(result.metadata, "category")
                            and result.metadata.category
                        ):
                            categories_set.add(result.metadata.category)
                        elif (
                            isinstance(result.metadata, dict)
                            and "category" in result.metadata
                        ):
                            categories_set.add(result.metadata["category"])
                    available_categories = list(categories_set)
            except Exception as e:
                logger.debug(f"Could not extract categories: {e}")
                available_categories = [
                    "examples",
                    "tutorials",
                    "reference",
                    "guides",
                ]  # Default categories

            # Create collections info
            collections = {
                collection_name: {
                    "count": document_count,
                    "embedding_dimension": collection_stats.get(
                        "embedding_dimension", 0
                    ),
                    "distance_metric": collection_stats.get(
                        "distance_metric", "cosine"
                    ),
                }
            }

            # Get last updated time (simplified - you might want to track this more accurately)
            from datetime import datetime

            last_updated = datetime.now().isoformat() if document_count > 0 else None

            return {
                # Original fields for backward compatibility
                "collection_name": collection_name,
                "document_count": document_count,
                "embedding_dimension": collection_stats.get("embedding_dimension", 0),
                "distance_metric": collection_stats.get("distance_metric", "cosine"),
                # New comprehensive fields expected by the API
                "collections": collections,
                "total_documents": document_count,
                "total_chunks": document_count,  # Assuming 1 chunk per document for now
                "storage_size_mb": storage_size_mb,
                "last_updated": last_updated,
                "available_categories": available_categories,
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {
                "collections": {},
                "total_documents": 0,
                "total_chunks": 0,
                "storage_size_mb": 0.0,
                "last_updated": None,
                "available_categories": [],
                # Backward compatibility
                "collection_name": self.vector_store.config.collection_name
                if self.vector_store
                else "unknown",
                "document_count": 0,
                "embedding_dimension": 0,
                "distance_metric": "cosine",
            }

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

    async def clear_knowledge_base(self) -> dict:
        """Clear all documents from the knowledge base."""
        try:
            logger.info("Clearing entire knowledge base...")

            # Reset the collection (clears all documents)
            self.vector_store.reset_collection()

            logger.info("Knowledge base cleared successfully")
            return {"success": True, "message": "Knowledge base cleared successfully"}

        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return {
                "success": False,
                "message": f"Failed to clear knowledge base: {str(e)}",
            }

    async def clear_knowledge_base_by_category(self, category: str) -> dict:
        """Clear documents from a specific category in the knowledge base."""
        try:
            logger.info(f"Clearing documents from category: {category}")

            # Get all documents in the collection
            collection_stats = self.vector_store.get_collection_stats()
            document_count = collection_stats.get("document_count", 0)

            if document_count == 0:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": f"No documents found in category '{category}'",
                }

            # Search for documents in the category to get their IDs
            # We'll use a broad search to get all documents, then filter by category
            try:
                all_results = self.vector_store.search_by_text("", top_k=document_count)
                category_ids = []

                for result in all_results:
                    # Check if this document belongs to the target category
                    doc_category = None
                    if hasattr(result.metadata, "category"):
                        doc_category = result.metadata.category
                    elif (
                        isinstance(result.metadata, dict)
                        and "category" in result.metadata
                    ):
                        doc_category = result.metadata["category"]

                    if doc_category == category:
                        # Extract document ID from metadata or use a unique identifier
                        doc_id = None
                        if hasattr(result.metadata, "id"):
                            doc_id = result.metadata.id
                        elif (
                            isinstance(result.metadata, dict)
                            and "id" in result.metadata
                        ):
                            doc_id = result.metadata["id"]
                        elif hasattr(result, "id"):
                            doc_id = result.id

                        if doc_id:
                            category_ids.append(doc_id)

                # Delete documents by ID if the vector store supports it
                deleted_count = 0
                if hasattr(self.vector_store, "delete_by_ids") and category_ids:
                    deleted_count = self.vector_store.delete_by_ids(category_ids)
                else:
                    # Fallback: If we can't delete by ID, we'll need to recreate the collection
                    # without the documents from this category
                    logger.warning(
                        f"Vector store doesn't support deletion by ID. Cannot selectively delete category '{category}'"
                    )
                    return {
                        "success": False,
                        "deleted_count": 0,
                        "message": "Selective deletion not supported. Use full knowledge base clear instead.",
                    }

                logger.info(
                    f"Deleted {deleted_count} documents from category '{category}'"
                )
                return {
                    "success": True,
                    "deleted_count": deleted_count,
                    "message": f"Deleted {deleted_count} documents from category '{category}'",
                }

            except Exception as search_error:
                logger.error(f"Error searching for category documents: {search_error}")
                return {
                    "success": False,
                    "deleted_count": 0,
                    "message": f"Failed to find documents in category '{category}': {str(search_error)}",
                }

        except Exception as e:
            logger.error(f"Error clearing category {category}: {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "message": f"Failed to clear category '{category}': {str(e)}",
            }


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
