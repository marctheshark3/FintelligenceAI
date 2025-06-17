"""
Knowledge Base API endpoints for FintelligenceAI.

This module provides REST API endpoints for managing the knowledge base,
including document ingestion, file uploads, and knowledge base statistics.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from fintelligence_ai.knowledge.ingestion import (
    IngestionPipeline,
    IngestionResult,
    KnowledgeBaseManager,
)

# Create router
router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

logger = logging.getLogger(__name__)

# Global knowledge base manager instance
_knowledge_manager: Optional[KnowledgeBaseManager] = None


class FileUploadResponse(BaseModel):
    """Response model for file upload."""

    success: bool
    filename: str
    file_id: str
    file_path: str
    size_bytes: int
    message: str


class IngestionRequest(BaseModel):
    """Request model for knowledge base ingestion."""

    source_type: str = Field(
        default="files",
        description="Source type: files, urls, github, or ergoscript",
    )
    file_paths: list[str] = Field(
        default_factory=list, description="List of file paths to process"
    )
    urls: list[str] = Field(default_factory=list, description="List of URLs to process")
    github_repos: list[str] = Field(
        default_factory=list, description="List of GitHub repositories"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Target collection name"
    )
    overwrite: bool = Field(default=False, description="Overwrite existing documents")


class KnowledgeStatsResponse(BaseModel):
    """Response model for knowledge base statistics."""

    collections: dict[str, Any]
    total_documents: int
    total_chunks: int
    storage_size_mb: float
    last_updated: Optional[str]
    available_categories: list[str]


class IngestionJobResponse(BaseModel):
    """Response model for ingestion job status."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[IngestionResult]


# Global job tracking
_ingestion_jobs: dict[str, IngestionJobResponse] = {}


async def get_knowledge_manager() -> KnowledgeBaseManager:
    """Get or create the knowledge base manager instance."""
    global _knowledge_manager
    if _knowledge_manager is None:
        logger.info("Initializing Knowledge Base Manager")
        _knowledge_manager = KnowledgeBaseManager()
        await _knowledge_manager.initialize()
        logger.info("Knowledge Base Manager initialized successfully")
    return _knowledge_manager


@router.post("/upload-files", response_model=list[FileUploadResponse])
async def upload_files(
    files: list[UploadFile] = File(...),
    category: str = Form(default="documents"),
    preserve_structure: bool = Form(default=True),
) -> list[FileUploadResponse]:
    """
    Upload multiple files (including entire directories) to the knowledge base.

    Files will be saved to the appropriate directory in the knowledge-base folder
    structure and can then be ingested into the vector database.
    """
    try:
        allowed_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".py",
            ".js",
            ".json",
            ".rst",
            ".tex",
        }
        uploaded_files = []
        upload_errors = []

        # Process each file
        for file in files:
            try:
                # Validate file type
                file_extension = Path(file.filename).suffix.lower()

                if file_extension not in allowed_extensions:
                    upload_errors.append(
                        f"Skipped {file.filename}: unsupported file type {file_extension}"
                    )
                    continue

                # Generate unique file ID
                file_id = str(uuid4())

                # Handle directory structure preservation
                if preserve_structure and "/" in file.filename:
                    # Extract directory path from filename (for directory uploads)
                    file_path_parts = Path(file.filename).parts

                    # Determine target directory
                    kb_base = Path("knowledge-base")
                    if category in ["tutorials", "guides", "examples", "reference"]:
                        target_dir = kb_base / "categories" / category
                    else:
                        target_dir = kb_base / "documents"

                    # Preserve directory structure
                    if len(file_path_parts) > 1:
                        # Create subdirectories
                        for part in file_path_parts[:-1]:
                            target_dir = target_dir / part

                    # Create final filename
                    base_name = Path(file.filename).stem
                    final_filename = f"{base_name}_{file_id[:8]}{file_extension}"
                    file_path = target_dir / final_filename
                else:
                    # Standard file upload without structure preservation
                    kb_base = Path("knowledge-base")
                    if category in ["tutorials", "guides", "examples", "reference"]:
                        target_dir = kb_base / "categories" / category
                    else:
                        target_dir = kb_base / "documents"

                    base_name = Path(file.filename).stem
                    safe_filename = f"{base_name}_{file_id[:8]}{file_extension}"
                    file_path = target_dir / safe_filename

                # Create directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Save file
                content = await file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)

                logger.info(f"File uploaded: {file_path} ({len(content)} bytes)")

                uploaded_files.append(
                    FileUploadResponse(
                        success=True,
                        filename=file.filename,
                        file_id=file_id,
                        file_path=str(file_path),
                        size_bytes=len(content),
                        message=f"File uploaded successfully to {category} category",
                    )
                )

            except Exception as e:
                error_msg = f"Failed to upload {file.filename}: {str(e)}"
                logger.error(error_msg)
                upload_errors.append(error_msg)
                continue

        # If no files were uploaded successfully, raise an error
        if not uploaded_files and upload_errors:
            raise HTTPException(
                status_code=400,
                detail=f"No files uploaded successfully. Errors: {'; '.join(upload_errors)}",
            )

        # Log summary
        logger.info(
            f"Upload summary: {len(uploaded_files)} files uploaded, {len(upload_errors)} errors"
        )

        if upload_errors:
            logger.warning(f"Upload errors: {'; '.join(upload_errors)}")

        return uploaded_files

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk file upload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Bulk file upload failed: {str(e)}"
        ) from e


@router.post("/upload-file", response_model=FileUploadResponse)
async def upload_single_file(
    file: UploadFile = File(...),
    category: str = Form(default="documents"),
) -> FileUploadResponse:
    """
    Upload a single file to the knowledge base (legacy endpoint).

    This endpoint is provided for backward compatibility.
    For multiple files or directories, use /upload-files instead.
    """
    try:
        # Use the new multi-file endpoint with a single file
        results = await upload_files([file], category, preserve_structure=False)

        if results:
            return results[0]
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single file upload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Single file upload failed: {str(e)}"
        ) from e


@router.post("/ingest", response_model=IngestionJobResponse)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
) -> IngestionJobResponse:
    """
    Start knowledge base ingestion process.

    This endpoint starts an asynchronous ingestion job that processes documents
    and adds them to the vector database. Use the job ID to check progress.
    """
    try:
        job_id = str(uuid4())

        # Create initial job response
        job_response = IngestionJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Ingestion job queued",
            result=None,
        )

        # Store job in global tracking
        _ingestion_jobs[job_id] = job_response

        # Start background task
        background_tasks.add_task(run_ingestion_task, job_id, request)

        return job_response

    except Exception as e:
        logger.error(f"Failed to start ingestion: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start ingestion: {str(e)}"
        ) from e


async def run_ingestion_task(job_id: str, request: IngestionRequest):
    """Run the ingestion task in the background."""
    try:
        # Update job status
        _ingestion_jobs[job_id].status = "running"
        _ingestion_jobs[job_id].message = "Starting ingestion process..."
        _ingestion_jobs[job_id].progress = 0.1

        # Get knowledge manager
        knowledge_manager = await get_knowledge_manager()
        pipeline = IngestionPipeline(knowledge_manager)

        if request.source_type == "ergoscript":
            # Run ErgoScript knowledge base setup
            _ingestion_jobs[job_id].message = "Ingesting ErgoScript knowledge base..."
            _ingestion_jobs[job_id].progress = 0.3

            result = await pipeline.run_ergoscript_ingestion()

        elif request.source_type == "files" and request.file_paths:
            # Process uploaded files
            _ingestion_jobs[job_id].message = "Processing uploaded files..."
            _ingestion_jobs[job_id].progress = 0.2

            # Create documents from file paths
            from fintelligence_ai.rag.models import Document, DocumentMetadata

            documents = []
            for i, file_path in enumerate(request.file_paths):
                path = Path(file_path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="ignore")

                    # Determine category from path
                    if "categories" in path.parts:
                        category_idx = path.parts.index("categories")
                        if category_idx + 1 < len(path.parts):
                            category = path.parts[category_idx + 1]
                        else:
                            category = "documents"
                    else:
                        category = "documents"

                    metadata = DocumentMetadata(
                        source="uploaded_file",
                        category=category,
                        complexity="intermediate",
                        tags=[category],
                        file_path=str(path),
                    )

                    doc = Document(
                        id=str(uuid4()),
                        content=content,
                        metadata=metadata,
                    )
                    documents.append(doc)

                # Update progress during file processing
                file_progress = 0.2 + (0.3 * (i + 1) / len(request.file_paths))
                _ingestion_jobs[job_id].progress = file_progress
                _ingestion_jobs[
                    job_id
                ].message = (
                    f"Processing file {i + 1}/{len(request.file_paths)}: {path.name}"
                )

            # Update progress for vector embedding and storage phase
            _ingestion_jobs[
                job_id
            ].message = "Creating embeddings and storing in vector database..."
            _ingestion_jobs[job_id].progress = 0.6

            # Ingest custom documents
            result = await pipeline.run_custom_ingestion(
                documents, request.collection_name or "uploaded_documents"
            )

            # Update progress after storage
            _ingestion_jobs[job_id].message = "Finalizing vector database operations..."
            _ingestion_jobs[job_id].progress = 0.9

        else:
            raise ValueError(
                f"Unsupported source type or missing data: {request.source_type}"
            )

        # Add a delay to ensure vector database operations are fully committed
        await asyncio.sleep(2)  # Give the vector database time to commit changes

        # Final progress update before completion
        _ingestion_jobs[job_id].message = "Verifying ingestion results..."
        _ingestion_jobs[job_id].progress = 0.95

        # Verify the ingestion by checking stats
        try:
            stats = await knowledge_manager.get_knowledge_base_stats()
            logger.info(
                f"Post-ingestion stats: {stats.get('total_documents', 0)} documents in knowledge base"
            )
        except Exception as e:
            logger.warning(f"Could not verify ingestion stats: {e}")

        # Update job with results
        _ingestion_jobs[job_id].status = "completed" if result.success else "failed"
        _ingestion_jobs[job_id].progress = 1.0
        _ingestion_jobs[job_id].result = result
        _ingestion_jobs[job_id].message = (
            f"Ingestion completed successfully! Processed {result.documents_processed} documents, "
            f"created {result.chunks_created} chunks, stored {result.chunks_stored} chunks in vector database"
            if result.success
            else f"Ingestion failed: {'; '.join(result.errors)}"
        )

    except Exception as e:
        logger.error(f"Ingestion task failed: {e}")
        _ingestion_jobs[job_id].status = "failed"
        _ingestion_jobs[job_id].progress = 1.0
        _ingestion_jobs[job_id].message = f"Ingestion failed: {str(e)}"


@router.get("/ingest/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_status(job_id: str) -> IngestionJobResponse:
    """Get the status of an ingestion job."""
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Ingestion job {job_id} not found")

    return _ingestion_jobs[job_id]


@router.get("/ingest", response_model=list[IngestionJobResponse])
async def list_ingestion_jobs() -> list[IngestionJobResponse]:
    """List all active ingestion jobs."""
    return list(_ingestion_jobs.values())


@router.delete("/clear")
async def clear_knowledge_base(
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> dict[str, str]:
    """Clear all documents from the knowledge base."""
    try:
        # Clear the vector database
        await knowledge_manager.clear_knowledge_base()

        return {"message": "Knowledge base cleared successfully"}

    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.delete("/clear/{category}")
async def clear_knowledge_base_by_category(
    category: str,
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> dict[str, str]:
    """Clear documents from a specific category in the knowledge base."""
    try:
        # Clear documents by category
        result = await knowledge_manager.clear_knowledge_base_by_category(category)

        return {
            "message": f"Cleared {result.get('deleted_count', 0)} documents from category '{category}'"
        }

    except Exception as e:
        logger.error(f"Failed to clear category {category}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear category: {str(e)}"
        )


@router.get("/categories")
async def get_available_categories(
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> dict[str, list[str]]:
    """Get list of available categories in the knowledge base."""
    try:
        stats = await knowledge_manager.get_knowledge_base_stats()
        categories = stats.get("available_categories", [])

        return {"categories": categories}

    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get categories: {str(e)}"
        )


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats(
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> KnowledgeStatsResponse:
    """Get current knowledge base statistics."""
    try:
        stats = await knowledge_manager.get_knowledge_base_stats()

        return KnowledgeStatsResponse(
            collections=stats.get("collections", {}),
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            storage_size_mb=stats.get("storage_size_mb", 0.0),
            last_updated=stats.get("last_updated"),
            available_categories=stats.get("available_categories", []),
        )

    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge stats: {str(e)}"
        )


@router.post("/refresh")
async def refresh_knowledge_base(
    background_tasks: BackgroundTasks,
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> dict[str, str]:
    """Refresh the knowledge base by re-ingesting all documents."""
    try:
        job_id = str(uuid4())

        # Create job response
        job_response = IngestionJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Knowledge base refresh queued",
            result=None,
        )

        _ingestion_jobs[job_id] = job_response

        # Start refresh task
        background_tasks.add_task(run_refresh_task, job_id, knowledge_manager)

        return {"job_id": job_id, "message": "Knowledge base refresh started"}

    except Exception as e:
        logger.error(f"Failed to start refresh: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start refresh: {str(e)}"
        )


async def run_refresh_task(job_id: str, knowledge_manager: KnowledgeBaseManager):
    """Run the refresh task in the background."""
    try:
        _ingestion_jobs[job_id].status = "running"
        _ingestion_jobs[job_id].message = "Refreshing knowledge base..."
        _ingestion_jobs[job_id].progress = 0.2

        result = await knowledge_manager.refresh_knowledge_base()

        _ingestion_jobs[job_id].status = "completed" if result.success else "failed"
        _ingestion_jobs[job_id].progress = 1.0
        _ingestion_jobs[job_id].result = result
        _ingestion_jobs[job_id].message = (
            f"Refresh completed: {result.documents_processed} documents processed"
            if result.success
            else f"Refresh failed: {'; '.join(result.errors)}"
        )

    except Exception as e:
        logger.error(f"Refresh task failed: {e}")
        _ingestion_jobs[job_id].status = "failed"
        _ingestion_jobs[job_id].progress = 1.0
        _ingestion_jobs[job_id].message = f"Refresh failed: {str(e)}"


@router.delete("/clear")
async def clear_knowledge_base(
    confirm: bool = False,
    knowledge_manager: KnowledgeBaseManager = Depends(get_knowledge_manager),
) -> dict[str, str]:
    """Clear all documents from the knowledge base."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Please set confirm=true to clear the knowledge base",
        )

    try:
        # This would need to be implemented in the knowledge manager
        # For now, return a placeholder response
        return {"message": "Knowledge base clearing is not yet implemented"}

    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.get("/health")
async def knowledge_health_check() -> dict[str, Any]:
    """Health check for knowledge base API."""
    try:
        knowledge_manager = await get_knowledge_manager()
        stats = await knowledge_manager.get_knowledge_base_stats()

        return {
            "status": "healthy",
            "knowledge_base": "available",
            "total_documents": stats.get("total_documents", 0),
            "vector_store": "connected",
            "embedding_service": "available",
        }

    except Exception as e:
        logger.error(f"Knowledge health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "knowledge_base": "unavailable",
        }
