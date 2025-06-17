"""
Integration tests for knowledge base ingestion functionality.

This module tests the complete knowledge ingestion pipeline to ensure it works
correctly with both OpenAI and Ollama embedding providers.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from fintelligence_ai.config import get_settings
from fintelligence_ai.knowledge import KnowledgeBaseManager

# Add scripts directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestKnowledgeIngestion:
    """Test cases for knowledge base ingestion functionality."""

    @pytest.fixture
    def temp_knowledge_dir(self):
        """Create a temporary knowledge base directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_knowledge_")
        knowledge_dir = Path(temp_dir) / "knowledge-base"
        knowledge_dir.mkdir(exist_ok=True)

        # Create test directory structure
        (knowledge_dir / "documents").mkdir(exist_ok=True)
        (knowledge_dir / "processed").mkdir(exist_ok=True)
        (knowledge_dir / "categories").mkdir(exist_ok=True)

        # Create test content
        test_doc_content = """# Test ErgoScript Document

This is a test document for verifying knowledge base ingestion.

## ErgoScript Example

Here's a simple ErgoScript example:

```ergoscript
{
    val threshold = 1000000L
    sigmaProp(OUTPUTS(0).value >= threshold)
}
```

This contract ensures that the first output has a value of at least 1 ERG.

## Testing Notes

- **Category**: general
- **Source**: local_files
- **Complexity**: beginner
- **Tags**: test, ergoscript, example
"""

        test_doc_path = knowledge_dir / "documents" / "test-ergoscript.md"
        test_doc_path.write_text(test_doc_content)

        # Create a test URLs file (but keep it minimal for testing)
        urls_file = knowledge_dir / "urls.txt"
        urls_file.write_text("# Test URLs file\n# No URLs for testing\n")

        # Create test GitHub repos file (minimal for testing)
        github_file = knowledge_dir / "github-repos.txt"
        github_file.write_text("# Test GitHub repos file\n# No repos for testing\n")

        yield knowledge_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def temp_vector_db(self):
        """Create a temporary vector database directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_vectordb_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def knowledge_manager(self, temp_vector_db):
        """Create a KnowledgeBaseManager instance for testing."""
        # Temporarily override the vector database path
        original_env = os.environ.get("VECTOR_DB_PATH")
        os.environ["VECTOR_DB_PATH"] = temp_vector_db

        try:
            manager = KnowledgeBaseManager()
            yield manager
        finally:
            # Restore original environment
            if original_env:
                os.environ["VECTOR_DB_PATH"] = original_env
            elif "VECTOR_DB_PATH" in os.environ:
                del os.environ["VECTOR_DB_PATH"]

    @pytest.mark.asyncio
    async def test_knowledge_manager_initialization(self, knowledge_manager):
        """Test that KnowledgeBaseManager initializes correctly."""
        await knowledge_manager.initialize()

        # Check that components are properly initialized
        assert knowledge_manager.vector_store is not None
        assert knowledge_manager.embedding_service is not None
        assert knowledge_manager.processor is not None

        # Get stats to verify initialization
        stats = await knowledge_manager.get_knowledge_base_stats()
        assert isinstance(stats, dict)
        assert "collection_name" in stats
        assert "document_count" in stats
        assert "embedding_dimension" in stats

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_stats(self, knowledge_manager):
        """Test getting stats from an empty knowledge base."""
        await knowledge_manager.initialize()

        stats = await knowledge_manager.get_knowledge_base_stats()
        assert stats["document_count"] == 0
        assert stats["collection_name"] == "ergoscript_examples"
        assert stats["embedding_dimension"] > 0  # Should have some dimension

    @pytest.mark.asyncio
    async def test_knowledge_ingestion_with_local_files(
        self, temp_knowledge_dir, temp_vector_db
    ):
        """Test knowledge ingestion with local files."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Create orchestrator with temporary directory
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)

        # Override vector database path
        original_data_path = (
            orchestrator.knowledge_manager.vector_store.persist_directory
        )
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        try:
            # Initialize knowledge manager and completely reset collection to avoid dimension mismatch
            await orchestrator.knowledge_manager.initialize()
            orchestrator.knowledge_manager.vector_store.reset_collection()

            # Run ingestion with force flag to ensure processing
            stats = await orchestrator.run_full_ingestion(force=True)

            # Verify ingestion results
            # Success is determined by having no errors and processing at least one file
            has_success = len(stats.errors) == 0 and stats.files_processed >= 1
            assert has_success, f"Ingestion failed. Files processed: {stats.files_processed}, Errors: {stats.errors}"
            assert stats.files_processed >= 1, "Should have processed at least 1 file"
            assert (
                stats.documents_created >= 1
            ), "Should have created at least 1 document"

            # Verify knowledge base has documents
            kb_stats = await orchestrator.knowledge_manager.get_knowledge_base_stats()
            assert (
                kb_stats["document_count"] > 0
            ), "Knowledge base should contain documents"

        finally:
            # Restore original path
            orchestrator.knowledge_manager.vector_store.persist_directory = (
                original_data_path
            )

    @pytest.mark.asyncio
    async def test_knowledge_base_search_functionality(
        self, temp_knowledge_dir, temp_vector_db
    ):
        """Test that knowledge base search works after ingestion."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Setup and run ingestion
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run ingestion
        stats = await orchestrator.run_full_ingestion(force=True)
        assert len(stats.errors) == 0, f"Test failed with errors: {stats.errors}"
        assert stats.documents_created > 0

        # Test search functionality
        search_queries = [
            "ErgoScript",
            "test document",
            "threshold value",
            "smart contract",
        ]

        for query in search_queries:
            results = await orchestrator.knowledge_manager.search_knowledge_base(
                query, limit=3
            )

            # Should return results (even if not perfect matches)
            assert isinstance(
                results, list
            ), f"Search should return a list for query: {query}"

            # If we get results, they should have the expected structure
            if results:
                result = results[0]
                assert "content" in result, "Result should have content"
                assert "metadata" in result, "Result should have metadata"
                assert "score" in result, "Result should have similarity score"
                assert isinstance(
                    result["score"], (int, float)
                ), "Score should be numeric"

    @pytest.mark.asyncio
    async def test_document_metadata_preservation(
        self, temp_knowledge_dir, temp_vector_db
    ):
        """Test that document metadata is properly preserved during ingestion."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Setup and run ingestion
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run ingestion
        stats = await orchestrator.run_full_ingestion(force=True)
        assert len(stats.errors) == 0, f"Test failed with errors: {stats.errors}"

        # Search for our test document
        results = await orchestrator.knowledge_manager.search_knowledge_base(
            "test ErgoScript document", limit=1
        )

        assert len(results) > 0, "Should find the test document"

        result = results[0]
        metadata = result["metadata"]

        # Check that metadata has expected fields
        expected_fields = ["source", "category", "file_path"]
        for field in expected_fields:
            assert (
                hasattr(metadata, field) or field in metadata
            ), f"Metadata should have {field}"

    @pytest.mark.asyncio
    async def test_knowledge_base_clearing(self, temp_knowledge_dir, temp_vector_db):
        """Test that knowledge base can be properly cleared."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Setup and run ingestion
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run ingestion
        stats = await orchestrator.run_full_ingestion(force=True)
        assert len(stats.errors) == 0, f"Test failed with errors: {stats.errors}"
        assert stats.documents_created > 0

        # Verify documents exist
        kb_stats = await orchestrator.knowledge_manager.get_knowledge_base_stats()
        assert kb_stats["document_count"] > 0

        # Clear the database
        orchestrator.clear_vector_database()

        # Verify it's empty (create new manager to avoid caching)
        new_manager = KnowledgeBaseManager()
        new_manager.vector_store.persist_directory = temp_vector_db
        await new_manager.initialize()

        new_stats = await new_manager.get_knowledge_base_stats()
        assert (
            new_stats["document_count"] == 0
        ), "Knowledge base should be empty after clearing"

    @pytest.mark.asyncio
    async def test_duplicate_file_handling(self, temp_knowledge_dir, temp_vector_db):
        """Test that duplicate files are handled correctly."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Setup orchestrator
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run ingestion twice
        stats1 = await orchestrator.run_full_ingestion(force=True)
        stats2 = await orchestrator.run_full_ingestion(force=False)  # Don't force

        assert stats1.success is True
        assert stats2.success is True

        # Second run should not process files again (they're already processed)
        assert (
            stats2.files_processed == 0 or stats2.documents_created == 0
        ), "Second run should skip already processed files"

    @pytest.mark.asyncio
    async def test_provider_agnostic_functionality(self, knowledge_manager):
        """Test that the ingestion works regardless of the configured provider."""
        settings = get_settings()

        # Test should work with whatever provider is configured
        await knowledge_manager.initialize()

        # Check embedding service is properly configured
        embedding_service = knowledge_manager.embedding_service
        assert embedding_service is not None

        # Test embedding dimension is reasonable
        stats = await knowledge_manager.get_knowledge_base_stats()
        embedding_dim = stats["embedding_dimension"]

        # Common embedding dimensions: 768 (OpenAI ada-002), 1536 (OpenAI-3-small),
        # 3072 (OpenAI-3-large), 768+ (various Ollama models)
        assert embedding_dim >= 384, f"Embedding dimension too small: {embedding_dim}"
        assert embedding_dim <= 4096, f"Embedding dimension too large: {embedding_dim}"

    @pytest.mark.asyncio
    async def test_error_handling_during_ingestion(
        self, temp_knowledge_dir, temp_vector_db
    ):
        """Test error handling during ingestion process."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Create a file with problematic content
        problem_file = temp_knowledge_dir / "documents" / "problem.txt"
        problem_file.write_text("" * 0)  # Empty file

        # Setup orchestrator
        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run ingestion - should handle empty files gracefully
        stats = await orchestrator.run_full_ingestion(force=True)

        # Ingestion should still succeed overall even with problematic files
        assert len(stats.errors) == 0, f"Test failed with errors: {stats.errors}"
        # Should still process the good file
        assert stats.files_processed >= 1


class TestKnowledgeIngestionCLI:
    """Test cases for the knowledge ingestion CLI interface."""

    @pytest.fixture
    def temp_env_setup(self, temp_knowledge_dir, temp_vector_db):
        """Setup temporary environment for CLI testing."""
        original_knowledge_dir = os.environ.get("KNOWLEDGE_BASE_DIR")
        original_vector_db = os.environ.get("VECTOR_DB_PATH")

        os.environ["KNOWLEDGE_BASE_DIR"] = str(temp_knowledge_dir)
        os.environ["VECTOR_DB_PATH"] = temp_vector_db

        yield temp_knowledge_dir, temp_vector_db

        # Restore environment
        if original_knowledge_dir:
            os.environ["KNOWLEDGE_BASE_DIR"] = original_knowledge_dir
        elif "KNOWLEDGE_BASE_DIR" in os.environ:
            del os.environ["KNOWLEDGE_BASE_DIR"]

        if original_vector_db:
            os.environ["VECTOR_DB_PATH"] = original_vector_db
        elif "VECTOR_DB_PATH" in os.environ:
            del os.environ["VECTOR_DB_PATH"]

    @pytest.mark.asyncio
    async def test_dry_run_functionality(self, temp_env_setup):
        """Test that dry run mode works correctly."""
        temp_knowledge_dir, temp_vector_db = temp_env_setup

        from ingest_knowledge import KnowledgeIngestionOrchestrator

        orchestrator = KnowledgeIngestionOrchestrator(temp_knowledge_dir)
        orchestrator.knowledge_manager.vector_store.persist_directory = temp_vector_db

        # Clear any existing data to avoid dimension mismatch issues
        orchestrator.clear_vector_database()

        # Run in dry-run mode
        stats = await orchestrator.run_full_ingestion(dry_run=True, force=True)

        # Should complete successfully but not actually create documents
        assert len(stats.errors) == 0, f"Test failed with errors: {stats.errors}"

        # Check that no documents were actually stored
        kb_stats = await orchestrator.knowledge_manager.get_knowledge_base_stats()
        assert kb_stats["document_count"] == 0, "Dry run should not store any documents"


@pytest.mark.integration
class TestProviderSpecificBehavior:
    """Test provider-specific behavior differences."""

    @pytest.mark.asyncio
    async def test_openai_provider_specific(self):
        """Test OpenAI-specific functionality if configured."""
        settings = get_settings()

        if settings.dspy.model_provider == "openai" and not settings.dspy.local_mode:
            # Test OpenAI-specific behavior
            manager = KnowledgeBaseManager()
            await manager.initialize()

            embedding_service = manager.embedding_service
            stats = await manager.get_knowledge_base_stats()

            # OpenAI models typically have these dimensions
            valid_openai_dims = [768, 1536, 3072]  # ada-002, 3-small, 3-large
            assert (
                stats["embedding_dimension"] in valid_openai_dims
            ), f"Unexpected OpenAI embedding dimension: {stats['embedding_dimension']}"

    @pytest.mark.asyncio
    async def test_ollama_provider_specific(self):
        """Test Ollama-specific functionality if configured."""
        settings = get_settings()

        if settings.dspy.model_provider == "ollama" or settings.dspy.local_mode:
            # Test Ollama-specific behavior
            manager = KnowledgeBaseManager()
            await manager.initialize()

            embedding_service = manager.embedding_service
            stats = await manager.get_knowledge_base_stats()

            # Ollama models vary, but should be reasonable
            assert (
                stats["embedding_dimension"] >= 384
            ), f"Ollama embedding dimension too small: {stats['embedding_dimension']}"
            assert (
                stats["embedding_dimension"] <= 2048
            ), f"Ollama embedding dimension too large: {stats['embedding_dimension']}"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
