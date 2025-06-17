#!/usr/bin/env python3
"""
Simplified End-to-End Test for Enhanced RAG Retrieval System

This script validates the enhanced retrieval optimizations using real ingested data
from our datasets: EIPs, ergo-python-appkit, ergoscript-by-example, and more.
"""

import os
import sys
from pathlib import Path

# Set local mode for testing
os.environ["DSPY_LOCAL_MODE"] = "true"
os.environ["DSPY_MODEL_PROVIDER"] = "ollama"

# Load environment variables first
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")

print("🦙 Using Ollama local mode for testing")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fintelligence_ai.config.settings import get_settings
from fintelligence_ai.rag.factory import (
    create_code_repository_pipeline,
    create_default_rag_pipeline,
    create_eip_documentation_pipeline,
    create_ergoscript_pipeline,
    create_flexible_documentation_pipeline,
)
from fintelligence_ai.rag.models import (
    Document,
    DocumentCategory,
    DocumentMetadata,
    DocumentSource,
)


def test_query_detection():
    """Test the query detection functionality without requiring pre-ingested data."""
    print("🧠 Testing Query Detection Logic")
    print("=" * 50)

    # Import the enhanced retrieval class to test query detection
    from fintelligence_ai.rag.embeddings import EmbeddingService
    from fintelligence_ai.rag.models import RetrievalConfig, VectorStoreConfig
    from fintelligence_ai.rag.retrieval import RetrievalEngine
    from fintelligence_ai.rag.vectorstore import VectorStoreManager

    # Create minimal configs for testing
    vector_config = VectorStoreConfig(
        collection_name="test_collection", persist_directory="data/chroma"
    )

    retrieval_config = RetrievalConfig(top_k=10, similarity_threshold=0.7)

    # Create vector store and retrieval engine for testing
    embedding_service = EmbeddingService()
    vector_store = VectorStoreManager(vector_config, embedding_service)
    retrieval_engine = RetrievalEngine(
        vector_store, retrieval_config, embedding_service
    )

    test_cases = [
        # EIP Queries
        {
            "query": "what is the eip for token collections on ergo?",
            "expected_type": "EIP",
            "description": "Original problem query - should detect EIP",
        },
        {
            "query": "which eip defines nft standards?",
            "expected_type": "EIP",
            "description": "NFT standard EIP query",
        },
        {
            "query": "explain eip-001",
            "expected_type": "EIP",
            "description": "Specific EIP number query",
        },
        # Code Queries
        {
            "query": "python examples in the appkit",
            "expected_type": "Code",
            "description": "Python appkit code query",
        },
        {
            "query": "how to create ErgoBox in python",
            "expected_type": "Code",
            "description": "Python API code query",
        },
        {
            "query": "GitHub repository setup.py",
            "expected_type": "Code",
            "description": "GitHub code repository query",
        },
        # ErgoScript Queries
        {
            "query": "how to create smart contracts in ergoscript",
            "expected_type": "ErgoScript",
            "description": "ErgoScript smart contract query",
        },
        {
            "query": "what are sigma props in ergoscript",
            "expected_type": "ErgoScript",
            "description": "Sigma propositions ErgoScript query",
        },
        {
            "query": "box registers ergoscript",
            "expected_type": "ErgoScript",
            "description": "Box registers ErgoScript query",
        },
        # General Queries
        {
            "query": "what is ergo blockchain",
            "expected_type": "General",
            "description": "General Ergo information",
        },
        {
            "query": "documentation overview",
            "expected_type": "General",
            "description": "General documentation query",
        },
    ]

    results = {
        "EIP": {"correct": 0, "total": 0},
        "Code": {"correct": 0, "total": 0},
        "ErgoScript": {"correct": 0, "total": 0},
        "General": {"correct": 0, "total": 0},
    }

    for test_case in test_cases:
        query = test_case["query"]
        expected_type = test_case["expected_type"]
        description = test_case["description"]

        print(f"\n🧪 {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_type}")

        # Test each detection method
        detected_type = "General"  # Default

        if retrieval_engine._is_eip_query(query):
            detected_type = "EIP"
        elif retrieval_engine._is_ergoscript_query(query):
            detected_type = "ErgoScript"
        elif retrieval_engine._is_code_query(query):
            detected_type = "Code"

        print(f"   Detected: {detected_type}")

        # Track results
        results[expected_type]["total"] += 1
        if detected_type == expected_type:
            results[expected_type]["correct"] += 1
            print("   ✅ Correct detection")
        else:
            print("   ❌ Incorrect detection")

    # Print summary
    print("\n🎯 Query Detection Accuracy Summary")
    print("=" * 40)

    total_correct = 0
    total_tests = 0

    for query_type, stats in results.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(
                f"{query_type:12}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)"
            )
            total_correct += stats["correct"]
            total_tests += stats["total"]

    overall_accuracy = (total_correct / total_tests) * 100
    print(f"{'Overall':12}: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")

    if overall_accuracy >= 90:
        print("🎉 Excellent query detection performance!")
    elif overall_accuracy >= 80:
        print("👍 Good query detection performance!")
    else:
        print("⚠️  Query detection needs improvement.")

    return overall_accuracy


def test_pipeline_configurations():
    """Test that our specialized pipelines can be created with different configurations."""
    print("\n🔧 Testing Pipeline Configurations")
    print("=" * 50)

    settings = get_settings()
    collection_name = f"{settings.chromadb.collection_name}-test"

    configs_to_test = [
        (
            "Default Pipeline",
            lambda: create_default_rag_pipeline(collection_name=collection_name),
        ),
        ("EIP Pipeline", lambda: create_eip_documentation_pipeline()),
        ("Code Pipeline", lambda: create_code_repository_pipeline(collection_name)),
        ("ErgoScript Pipeline", lambda: create_ergoscript_pipeline(collection_name)),
        (
            "Flexible Pipeline",
            lambda: create_flexible_documentation_pipeline(collection_name),
        ),
    ]

    successful_configs = 0

    for config_name, config_factory in configs_to_test:
        print(f"\n🧪 Testing {config_name}")
        try:
            pipeline = config_factory()

            # Verify pipeline has expected components
            assert hasattr(pipeline, "query"), "Pipeline missing query method"
            assert hasattr(
                pipeline, "retrieval_engine"
            ), "Pipeline missing retrieval engine"
            assert hasattr(pipeline, "vector_store"), "Pipeline missing vector store"

            # Check configuration specifics if available
            retrieval_config = pipeline.retrieval_config
            print(f"   ⚙️  Top-k: {retrieval_config.top_k}")
            print(
                f"   ⚙️  Similarity threshold: {retrieval_config.similarity_threshold}"
            )
            print(f"   ⚙️  Hybrid search alpha: {retrieval_config.hybrid_search_alpha}")

            print("   ✅ Configuration valid")
            successful_configs += 1

        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")

    print(
        f"\n📊 Pipeline Configuration Results: {successful_configs}/{len(configs_to_test)} successful"
    )

    return successful_configs == len(configs_to_test)


def test_document_processing():
    """Test document processing and retrieval without requiring full ingestion."""
    print("\n📄 Testing Document Processing")
    print("=" * 50)

    # Create a small test document
    test_doc = Document(
        id="test-eip-034",
        title="EIP-0034: Token Collection Standard",
        content="""
        # EIP-0034: Token Collection Standard

        This EIP defines a standard for token collections on the Ergo blockchain.
        Token collections allow grouping related NFTs together, enabling
        marketplaces and wallets to display collections properly.

        ## Specification

        A token collection MUST include:
        - Collection name
        - Collection description
        - Collection banner/logo
        - Individual token metadata

        ## Implementation

        Collections are implemented using ErgoScript contracts that enforce
        the collection standard and manage token minting/burning.
        """,
        metadata=DocumentMetadata(
            source=DocumentSource.GITHUB,
            category=DocumentCategory.GENERAL,
            url="https://github.com/ergoplatform/eips/blob/master/eip-0034.md",
            tags=["eip", "nft", "collection", "standard"],
            complexity="intermediate",
        ),
    )

    try:
        # Test that we can create a pipeline and add a document
        settings = get_settings()
        collection_name = f"{settings.chromadb.collection_name}-test-doc"

        pipeline = create_default_rag_pipeline(collection_name=collection_name)

        # Add the test document
        print("🔄 Adding test document...")
        doc_id = pipeline.add_document(test_doc)
        print(f"   ✅ Document added with ID: {doc_id}")

        # Test querying for the document
        print("🔍 Testing retrieval...")
        result = pipeline.query("what is the eip for token collections?")

        print(f"   📊 Found {len(result.retrieval_results)} documents")

        if result.retrieval_results:
            top_result = result.retrieval_results[0]
            print(f"   🏆 Top result: {top_result.title}")
            print(f"   ⭐ Relevance score: {top_result.score:.3f}")

            # Check if it found our test document
            if (
                "eip-0034" in top_result.content.lower()
                or "collection" in top_result.content.lower()
            ):
                print("   ✅ Successfully retrieved relevant content")
                return True
            else:
                print("   ⚠️  Retrieved content may not be relevant")
                return False
        else:
            print("   ❌ No documents retrieved")
            return False

    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        return False


def test_enhanced_retrieval():
    """Main test function."""
    print("🚀 Enhanced RAG Retrieval System - Testing")
    print("=" * 60)
    print("Testing core functionality without requiring full data ingestion")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Query Detection
    print("\n" + "=" * 60)
    try:
        accuracy = test_query_detection()
        if accuracy >= 80:
            tests_passed += 1
            print("✅ Query detection test PASSED")
        else:
            print("❌ Query detection test FAILED")
    except Exception as e:
        print(f"❌ Query detection test ERROR: {e}")

    # Test 2: Pipeline Configurations
    print("\n" + "=" * 60)
    try:
        if test_pipeline_configurations():
            tests_passed += 1
            print("✅ Pipeline configuration test PASSED")
        else:
            print("❌ Pipeline configuration test FAILED")
    except Exception as e:
        print(f"❌ Pipeline configuration test ERROR: {e}")

    # Test 3: Document Processing
    print("\n" + "=" * 60)
    try:
        if test_document_processing():
            tests_passed += 1
            print("✅ Document processing test PASSED")
        else:
            print("❌ Document processing test FAILED")
    except Exception as e:
        print(f"❌ Document processing test ERROR: {e}")

    # Final Results
    print("\n" + "=" * 60)
    print("🎉 ENHANCED RETRIEVAL SYSTEM - TEST SUMMARY")
    print("=" * 60)

    success_rate = (tests_passed / total_tests) * 100
    print(f"🎯 Tests Passed: {tests_passed}/{total_tests} ({success_rate:.1f}%)")

    if success_rate == 100:
        print("🎉 All tests passed! Enhanced retrieval system is working correctly.")
    elif success_rate >= 67:
        print("👍 Most tests passed! System is largely functional.")
    else:
        print("⚠️  Multiple test failures. System needs investigation.")

    print("\n✅ Testing completed!")


if __name__ == "__main__":
    test_enhanced_retrieval()
