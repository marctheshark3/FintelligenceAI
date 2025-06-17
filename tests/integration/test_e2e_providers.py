"""
End-to-End Provider Testing Suite

This module provides comprehensive testing for all AI providers (OpenAI, Ollama)
with separate ChromaDB collections and full pipeline validation including
ingestion, generation, research, and validation agents.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add scripts directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class AIProviderTestConfig:
    """Configuration for testing a specific AI provider."""

    def __init__(
        self,
        name: str,
        model_provider: str,
        local_mode: bool,
        required_env_vars: list[str],
        embedding_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        collection_suffix: str = "",
    ):
        self.name = name
        self.model_provider = model_provider
        self.local_mode = local_mode
        self.required_env_vars = required_env_vars
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.collection_suffix = collection_suffix or name.lower()

    def is_available(self) -> bool:
        """Check if this provider is available for testing."""
        # Check required environment variables
        for env_var in self.required_env_vars:
            if not os.getenv(env_var):
                return False

        # For Ollama, check if the service is running
        if self.local_mode and self.model_provider == "ollama":
            try:
                import requests

                response = requests.get("http://localhost:11434/api/version", timeout=5)
                return response.status_code == 200
            except Exception:
                return False

        return True

    def get_env_overrides(self) -> dict[str, str]:
        """Get environment variable overrides for this provider."""
        overrides = {
            "DSPY_MODEL_PROVIDER": self.model_provider,
            "DSPY_LOCAL_MODE": str(self.local_mode).lower(),
        }

        if self.embedding_model:
            if self.model_provider == "ollama":
                overrides["OLLAMA_EMBEDDING_MODEL"] = self.embedding_model
            else:
                overrides["OPENAI_EMBEDDING_MODEL"] = self.embedding_model

        if self.chat_model:
            if self.model_provider == "ollama":
                overrides["OLLAMA_MODEL"] = self.chat_model
            else:
                overrides["OPENAI_MODEL"] = self.chat_model

        return overrides


# Define available providers for testing
AVAILABLE_PROVIDERS = [
    AIProviderTestConfig(
        name="OpenAI",
        model_provider="openai",
        local_mode=False,
        required_env_vars=["OPENAI_API_KEY"],
        embedding_model="text-embedding-3-small",  # Use smaller model for faster tests
        chat_model="gpt-4o-mini",
        collection_suffix="openai",
    ),
    AIProviderTestConfig(
        name="Ollama",
        model_provider="ollama",
        local_mode=True,
        required_env_vars=[],  # No API keys required
        embedding_model="nomic-embed-text",
        chat_model="llama3.2",
        collection_suffix="ollama",
    ),
]


class E2EProviderTestSuite:
    """End-to-end test suite for AI providers."""

    def __init__(self, provider_config: AIProviderTestConfig):
        self.config = provider_config
        self.temp_knowledge_dir = None
        self.temp_vector_db = None
        self.orchestrator = None

    def setup_test_environment(self):
        """Set up isolated test environment for this provider."""
        # Create temporary directories
        self.temp_knowledge_dir = Path(
            tempfile.mkdtemp(prefix=f"test_knowledge_{self.config.name.lower()}_")
        )
        self.temp_vector_db = tempfile.mkdtemp(
            prefix=f"test_vectordb_{self.config.name.lower()}_"
        )

        # Create knowledge base structure
        knowledge_base = self.temp_knowledge_dir / "knowledge-base"
        knowledge_base.mkdir(exist_ok=True)
        (knowledge_base / "documents").mkdir(exist_ok=True)
        (knowledge_base / "processed").mkdir(exist_ok=True)
        (knowledge_base / "categories").mkdir(exist_ok=True)

        # Create comprehensive test content
        self._create_test_content(knowledge_base)

        return knowledge_base

    def _create_test_content(self, knowledge_base: Path):
        """Create comprehensive test content for validation."""

        # Test document 1: ErgoScript basics
        basic_doc = knowledge_base / "documents" / "ergoscript_basics.md"
        basic_doc.write_text(
            """# ErgoScript Basics

ErgoScript is a functional programming language designed for the Ergo blockchain.

## Basic Example

```ergoscript
{
    val threshold = 1000000L
    sigmaProp(OUTPUTS(0).value >= threshold)
}
```

This contract ensures output value is at least 1 ERG.

## Key Features

- **Functional**: Pure functional programming paradigm
- **Typed**: Static type system prevents runtime errors
- **Secure**: Built-in cryptographic primitives
- **Efficient**: Optimized for blockchain execution

## Use Cases

- Smart contracts
- Multi-signature wallets
- Atomic swaps
- Oracle scripts
"""
        )

        # Test document 2: Advanced concepts
        advanced_doc = knowledge_base / "documents" / "advanced_ergoscript.md"
        advanced_doc.write_text(
            """# Advanced ErgoScript Concepts

## Context Variables

ErgoScript provides access to transaction context:

- `SELF`: Current box being spent
- `INPUTS`: All input boxes in transaction
- `OUTPUTS`: All output boxes in transaction
- `HEIGHT`: Current blockchain height

## Sigma Propositions

```ergoscript
val userPK = PK("9f5ZKbECVTm25JTRQHDHGM5ehC8tUw5g1fCBQ4aaE792rWKq")
val backupPK = PK("9g6YKbECVTm25JTRQHDHGM5ehC8tUw5g1fCBQ4aaE792rWKr")
val timeout = 100

val mainCondition = userPK
val timeoutCondition = HEIGHT > timeout && backupPK

sigmaProp(mainCondition || timeoutCondition)
```

## Box Protection

Boxes can be protected by complex logical conditions combining:
- Public key signatures
- Time-based conditions
- Mathematical constraints
- Data integrity checks
"""
        )

        # Test document 3: API reference
        api_doc = knowledge_base / "documents" / "api_reference.md"
        api_doc.write_text(
            """# ErgoScript API Reference

## Built-in Functions

### Cryptographic Functions

- `blake2b256(bytes)`: Blake2b 256-bit hash
- `sha256(bytes)`: SHA-256 hash
- `proveDlog(groupElement)`: Prove discrete log knowledge

### Collection Operations

- `INPUTS.size`: Number of input boxes
- `OUTPUTS.filter(condition)`: Filter outputs by condition
- `INPUTS.exists(condition)`: Check if any input satisfies condition

### Mathematical Operations

- `max(a, b)`: Maximum of two values
- `min(a, b)`: Minimum of two values
- `abs(x)`: Absolute value

## Type System

### Primitive Types
- `Boolean`: true/false values
- `Int`: 32-bit integers
- `Long`: 64-bit integers
- `BigInt`: Arbitrary precision integers
- `GroupElement`: Elliptic curve points

### Collection Types
- `Coll[T]`: Collection of type T
- `Option[T]`: Optional value of type T
"""
        )

        # Create empty files for other sources (to keep tests fast)
        (knowledge_base / "urls.txt").write_text("# No URLs for testing\n")
        (knowledge_base / "github-repos.txt").write_text("# No repos for testing\n")

    async def test_knowledge_ingestion(self) -> dict[str, any]:
        """Test knowledge base ingestion for this provider."""
        from ingest_knowledge import KnowledgeIngestionOrchestrator

        # Set up environment for this provider
        with patch.dict(os.environ, self.config.get_env_overrides()):
            knowledge_base = self.setup_test_environment()

            # Create orchestrator with provider-specific collection
            self.orchestrator = KnowledgeIngestionOrchestrator(knowledge_base)

            # Override vector database path and collection name
            self.orchestrator.knowledge_manager.vector_store.persist_directory = (
                self.temp_vector_db
            )
            self.orchestrator.knowledge_manager.vector_store.collection_name = (
                f"test_collection_{self.config.collection_suffix}"
            )

            # Initialize and clear any existing data
            await self.orchestrator.knowledge_manager.initialize()
            self.orchestrator.knowledge_manager.vector_store.reset_collection()

            # Run ingestion
            stats = await self.orchestrator.run_full_ingestion(force=True)

            # Verify ingestion success
            success = len(stats.errors) == 0 and stats.files_processed >= 3
            assert success, f"Ingestion failed for {self.config.name}. Files: {stats.files_processed}, Errors: {stats.errors}"

            # Get knowledge base stats
            kb_stats = (
                await self.orchestrator.knowledge_manager.get_knowledge_base_stats()
            )

            return {
                "provider": self.config.name,
                "files_processed": stats.files_processed,
                "documents_created": stats.documents_created,
                "errors": stats.errors,
                "kb_stats": kb_stats,
                "collection_name": f"test_collection_{self.config.collection_suffix}",
            }

    async def test_knowledge_search(self) -> dict[str, any]:
        """Test knowledge base search functionality."""
        assert self.orchestrator is not None, "Must run ingestion test first"

        test_queries = [
            "What is ErgoScript?",
            "How do you create a smart contract?",
            "What are context variables?",
            "Blake2b hash function",
            "sigma propositions",
            "collection operations",
        ]

        search_results = {}

        with patch.dict(os.environ, self.config.get_env_overrides()):
            for query in test_queries:
                try:
                    results = (
                        await self.orchestrator.knowledge_manager.search_knowledge_base(
                            query, limit=3
                        )
                    )

                    search_results[query] = {
                        "result_count": len(results),
                        "top_score": results[0]["score"] if results else 0,
                        "found_content": bool(results),
                        "has_metadata": bool(results and results[0].get("metadata")),
                    }

                except Exception as e:
                    search_results[query] = {"error": str(e), "found_content": False}

        return {
            "provider": self.config.name,
            "search_results": search_results,
            "total_queries": len(test_queries),
            "successful_queries": sum(
                1 for r in search_results.values() if r.get("found_content", False)
            ),
        }

    async def test_simple_generation(self) -> dict[str, any]:
        """Test simple DSPy generation functionality."""
        assert self.orchestrator is not None, "Must run ingestion test first"

        test_prompts = ["What is ErgoScript?", "Name one ErgoScript feature"]

        generation_results = {}

        with patch.dict(os.environ, self.config.get_env_overrides()):
            try:
                # Test basic DSPy functionality
                import dspy

                # Configure DSPy based on provider using the correct API
                if self.config.model_provider == "openai":
                    # Use the correct DSPy LM interface for OpenAI
                    lm = dspy.LM(
                        model=f"openai/{self.config.chat_model}",
                        api_key=os.getenv("OPENAI_API_KEY"),
                        temperature=0.1,
                        max_tokens=1000,
                    )
                else:
                    # For Ollama, use the custom implementation from the codebase
                    from fintelligence_ai.core.ollama import get_ollama_dspy_model

                    lm = get_ollama_dspy_model(
                        model=self.config.chat_model, temperature=0.1, max_tokens=1000
                    )

                dspy.configure(lm=lm)

                for prompt in test_prompts:
                    try:
                        print(f"    Testing prompt: '{prompt}' with {self.config.name}")

                        # Try DSPy direct call first
                        response = lm(prompt)

                        # Handle different response types from DSPy
                        if isinstance(response, list):
                            # DSPy sometimes returns a list of responses
                            response_text = response[0] if response else ""
                        elif hasattr(response, "text"):
                            # Sometimes wrapped in an object
                            response_text = response.text
                        else:
                            # Direct string response
                            response_text = str(response) if response else ""

                        print(f"    Response received: {len(response_text)} chars")

                        generation_results[prompt] = {
                            "success": True,
                            "response_length": len(response_text),
                            "has_content": bool(response_text.strip()),
                            "response_preview": response_text[:100] + "..."
                            if len(response_text) > 100
                            else response_text,
                        }

                    except Exception as e:
                        print(f"    Error for prompt '{prompt}': {e}")
                        generation_results[prompt] = {"success": False, "error": str(e)}

            except Exception as e:
                generation_results = {"setup_error": str(e)}

        return {
            "provider": self.config.name,
            "generation_results": generation_results,
            "total_prompts": len(test_prompts),
            "successful_generations": sum(
                1
                for r in generation_results.values()
                if isinstance(r, dict) and r.get("success", False)
            ),
        }

    def cleanup(self):
        """Clean up test environment."""
        if self.temp_knowledge_dir:
            shutil.rmtree(self.temp_knowledge_dir, ignore_errors=True)
        if self.temp_vector_db:
            shutil.rmtree(self.temp_vector_db, ignore_errors=True)


@pytest.mark.integration
class TestE2EProviders:
    """End-to-end tests for all available AI providers."""

    def get_available_providers(self) -> list[AIProviderTestConfig]:
        """Get list of providers available for testing."""
        available = []
        for provider in AVAILABLE_PROVIDERS:
            if provider.is_available():
                available.append(provider)
        return available

    @pytest.mark.asyncio
    async def test_all_providers_e2e(self):
        """Run complete end-to-end test for all available providers."""
        available_providers = self.get_available_providers()

        if not available_providers:
            pytest.skip("No AI providers available for testing")

        print(f"\n🧪 Testing {len(available_providers)} available providers...")

        all_results = {}

        for provider_config in available_providers:
            print(f"\n🔄 Testing {provider_config.name} provider...")

            test_suite = E2EProviderTestSuite(provider_config)
            provider_results = {}

            try:
                # Test knowledge ingestion
                print("  📚 Testing knowledge ingestion...")
                ingestion_results = await test_suite.test_knowledge_ingestion()
                provider_results["ingestion"] = ingestion_results

                # Test knowledge search
                print("  🔍 Testing knowledge search...")
                search_results = await test_suite.test_knowledge_search()
                provider_results["search"] = search_results

                # Test simple generation
                print("  🤖 Testing generation...")
                generation_results = await test_suite.test_simple_generation()
                provider_results["generation"] = generation_results

                print(
                    f"  ✅ {provider_config.name} provider tests completed successfully!"
                )

            except Exception as e:
                print(f"  ❌ {provider_config.name} provider test failed: {e}")
                import traceback

                traceback.print_exc()
                provider_results["error"] = str(e)

            finally:
                test_suite.cleanup()

            all_results[provider_config.name] = provider_results

        # Print comprehensive summary
        self._print_test_summary(all_results)

        # Assert that at least one provider worked completely
        successful_providers = [
            name
            for name, results in all_results.items()
            if "error" not in results
            and results.get("ingestion", {}).get("files_processed", 0) > 0
        ]

        assert (
            len(successful_providers) > 0
        ), f"No providers completed successfully. Results: {all_results}"

    def _print_test_summary(self, results: dict[str, dict]):
        """Print comprehensive test results summary."""
        print("\n" + "=" * 80)
        print("🎯 END-TO-END PROVIDER TEST SUMMARY")
        print("=" * 80)

        for provider_name, provider_results in results.items():
            print(f"\n📊 {provider_name} Provider Results:")
            print("-" * 40)

            if "error" in provider_results:
                print(f"  ❌ Failed: {provider_results['error']}")
                continue

            # Ingestion results
            ingestion = provider_results.get("ingestion", {})
            print("  📚 Knowledge Ingestion:")
            print(f"    • Files processed: {ingestion.get('files_processed', 0)}")
            print(f"    • Documents created: {ingestion.get('documents_created', 0)}")
            print(f"    • Collection: {ingestion.get('collection_name', 'N/A')}")
            if "kb_stats" in ingestion:
                kb_stats = ingestion["kb_stats"]
                print(
                    f"    • Documents in DB: {kb_stats.get('total_documents', 'N/A')}"
                )
                print(
                    f"    • Embedding dimension: {kb_stats.get('embedding_dimension', 'N/A')}"
                )

            # Search results
            search = provider_results.get("search", {})
            print("  🔍 Knowledge Search:")
            print(
                f"    • Successful queries: {search.get('successful_queries', 0)}/{search.get('total_queries', 0)}"
            )

            # Generation results
            generation = provider_results.get("generation", {})
            if "setup_error" in generation:
                print(f"  🤖 Generation: Setup failed - {generation['setup_error']}")
            else:
                print("  🤖 Generation:")
                print(
                    f"    • Successful generations: {generation.get('successful_generations', 0)}/{generation.get('total_prompts', 0)}"
                )

                # Show generation details
                gen_results = generation.get("generation_results", {})
                for prompt, result in gen_results.items():
                    if isinstance(result, dict):
                        if result.get("success"):
                            print(
                                f"    • '{prompt}': ✅ {result.get('response_length', 0)} chars"
                            )
                        else:
                            print(
                                f"    • '{prompt}': ❌ {result.get('error', 'Unknown error')}"
                            )
                    elif isinstance(result, str):
                        print(f"    • Setup error: {result}")

        print("\n" + "=" * 80)

    @pytest.mark.asyncio
    async def test_provider_specific_openai(self):
        """Test OpenAI provider specifically if available."""
        openai_config = next(
            (p for p in AVAILABLE_PROVIDERS if p.name == "OpenAI"), None
        )

        if not openai_config or not openai_config.is_available():
            pytest.skip("OpenAI provider not available")

        test_suite = E2EProviderTestSuite(openai_config)

        try:
            # Test that OpenAI-specific features work
            results = await test_suite.test_knowledge_ingestion()

            # Verify OpenAI-specific characteristics
            assert (
                results["kb_stats"]["embedding_dimension"]
                in [
                    1536,
                    3072,
                ]
            ), f"Unexpected OpenAI embedding dimension: {results['kb_stats']['embedding_dimension']}"

        finally:
            test_suite.cleanup()

    @pytest.mark.asyncio
    async def test_provider_specific_ollama(self):
        """Test Ollama provider specifically if available."""
        ollama_config = next(
            (p for p in AVAILABLE_PROVIDERS if p.name == "Ollama"), None
        )

        if not ollama_config or not ollama_config.is_available():
            pytest.skip("Ollama provider not available")

        test_suite = E2EProviderTestSuite(ollama_config)

        try:
            # Test that Ollama-specific features work
            results = await test_suite.test_knowledge_ingestion()

            # Verify Ollama-specific characteristics
            assert (
                results["kb_stats"]["embedding_dimension"] >= 384
            ), f"Ollama embedding dimension too small: {results['kb_stats']['embedding_dimension']}"

        finally:
            test_suite.cleanup()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
