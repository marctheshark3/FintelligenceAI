"""
Vector store management using ChromaDB for document storage and retrieval.

This module provides functionality for storing document embeddings and performing
similarity searches using ChromaDB as the vector database backend.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .embeddings import EmbeddingService
from .models import Document, VectorStoreConfig, Query, RetrievalResult, DocumentMetadata

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for vector store operations using ChromaDB.
    
    This class handles the storage, retrieval, and management of document embeddings
    in ChromaDB, providing similarity search capabilities for the RAG pipeline.
    """
    
    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_service: Optional[EmbeddingService] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.
        
        Args:
            config: Configuration for vector store operations
            embedding_service: Service for generating embeddings
            persist_directory: Directory to persist ChromaDB data
        """
        self.config = config
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_directory = persist_directory or "./data/chroma"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized vector store with collection: {config.collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.config.collection_name,
            )
            logger.info(f"Found existing collection: {self.config.collection_name}")
            
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "description": "FintelligenceAI ErgoScript knowledge base",
                },
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
            
        return collection
    
    def add_document(self, document: Document) -> str:
        """
        Add a single document to the vector store.
        
        Args:
            document: Document to add
            
        Returns:
            Document ID that was added
        """
        if not document.embedding:
            document = self.embedding_service.embed_document(document)
            
        # Prepare metadata for ChromaDB (must be strings, numbers, or booleans)
        metadata = self._prepare_metadata(document.metadata)
        metadata.update({
            "document_id": document.id,
            "title": document.title or "",
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        })
        
        # Add to collection
        self.collection.add(
            ids=[document.id],
            embeddings=[document.embedding],
            documents=[document.content],
            metadatas=[metadata],
        )
        
        logger.debug(f"Added document {document.id} to vector store")
        return document.id
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        # Generate embeddings for documents that don't have them
        docs_to_embed = [doc for doc in documents if not doc.embedding]
        if docs_to_embed:
            texts = [
                (doc.title + " " + doc.content if doc.title else doc.content)
                for doc in docs_to_embed
            ]
            embeddings = self.embedding_service.generate_embeddings_batch(texts)
            
            for doc, embedding in zip(docs_to_embed, embeddings):
                doc.embedding = embedding
        
        # Prepare data for batch insertion
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = []
        
        for doc in documents:
            metadata = self._prepare_metadata(doc.metadata)
            metadata.update({
                "document_id": doc.id,
                "title": doc.title or "",
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
            })
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search_similar(
        self,
        query: Query,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query object with text and optional embedding
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results ranked by similarity
        """
        # Generate embedding if not present
        if not query.embedding:
            query = self.embedding_service.embed_query(query)
        
        # Prepare where clause for filtering
        where_clause = self._prepare_where_clause(filters or query.filters)
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query.embedding],
            n_results=min(top_k, 100),  # ChromaDB limit
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert results to RetrievalResult objects
        retrieval_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, content, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (assuming cosine distance)
                similarity_score = max(0.0, 1.0 - distance)
                
                # Reconstruct document metadata
                doc_metadata = self._reconstruct_metadata(metadata)
                
                result = RetrievalResult(
                    document_id=doc_id,
                    content=content,
                    title=metadata.get("title", ""),
                    score=similarity_score,
                    metadata=doc_metadata,
                    rank=i + 1,
                    retrieval_method="semantic",
                )
                
                retrieval_results.append(result)
        
        logger.debug(f"Found {len(retrieval_results)} similar documents for query")
        return retrieval_results
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results
        """
        query = Query(text=query_text, filters=filters or {})
        return self.search_similar(query, top_k, filters)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"],
            )
            
            if results["ids"] and results["ids"][0]:
                content = results["documents"][0][0]
                metadata_dict = results["metadatas"][0][0]
                embedding = results["embeddings"][0][0] if results["embeddings"] else None
                
                # Reconstruct document metadata
                doc_metadata = self._reconstruct_metadata(metadata_dict)
                
                document = Document(
                    id=document_id,
                    content=content,
                    title=metadata_dict.get("title", ""),
                    metadata=doc_metadata,
                    embedding=embedding,
                )
                
                return document
                
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {str(e)}")
            
        return None
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document in the vector store.
        
        Args:
            document: Updated document
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Delete existing document
            self.delete_document(document.id)
            
            # Add updated document
            self.add_document(document)
            
            logger.debug(f"Updated document {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {str(e)}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.collection_name,
                "document_count": count,
                "embedding_dimension": self.config.embedding_dimension,
                "distance_metric": self.config.distance_metric,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        Reset the collection by deleting all documents.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(self.config.collection_name)
            
            # Recreate the collection
            self.collection = self._get_or_create_collection()
            
            logger.info(f"Reset collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False
    
    def _prepare_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB only supports strings, numbers, and booleans in metadata.
        """
        return {
            "source": metadata.source.value,
            "category": metadata.category.value,
            "complexity": metadata.complexity.value,
            "tags": ",".join(metadata.tags) if metadata.tags else "",
            "last_updated": metadata.last_updated.isoformat(),
            "author": metadata.author or "",
            "url": metadata.url or "",
            "file_path": metadata.file_path or "",
            "language": metadata.language,
            "tested": metadata.tested,
            "performance_notes": metadata.performance_notes or "",
        }
    
    def _reconstruct_metadata(self, metadata_dict: Dict[str, Any]) -> DocumentMetadata:
        """Reconstruct DocumentMetadata from ChromaDB metadata."""
        from .models import DocumentSource, DocumentCategory, ComplexityLevel
        from datetime import datetime
        
        return DocumentMetadata(
            source=DocumentSource(metadata_dict.get("source", "examples")),
            category=DocumentCategory(metadata_dict.get("category", "examples")),
            complexity=ComplexityLevel(metadata_dict.get("complexity", "beginner")),
            tags=metadata_dict.get("tags", "").split(",") if metadata_dict.get("tags") else [],
            last_updated=datetime.fromisoformat(metadata_dict.get("last_updated", datetime.utcnow().isoformat())),
            author=metadata_dict.get("author") or None,
            url=metadata_dict.get("url") or None,
            file_path=metadata_dict.get("file_path") or None,
            language=metadata_dict.get("language", "ergoscript"),
            tested=metadata_dict.get("tested", False),
            performance_notes=metadata_dict.get("performance_notes") or None,
        )
    
    def _prepare_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare where clause for ChromaDB filtering."""
        if not filters:
            return None
            
        where_clause = {}
        
        for key, value in filters.items():
            if key in ["source", "category", "complexity", "language"]:
                where_clause[key] = value
            elif key == "tested":
                where_clause[key] = bool(value)
            elif key == "tags":
                # For tags, we'd need to use $contains operator
                # This is a simplified implementation
                if isinstance(value, list):
                    # Use the first tag for simplicity
                    where_clause["tags"] = {"$contains": value[0]}
                else:
                    where_clause["tags"] = {"$contains": str(value)}
        
        return where_clause if where_clause else None 