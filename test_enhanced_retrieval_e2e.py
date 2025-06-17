#!/usr/bin/env python3
"""
End-to-End Test for Enhanced RAG Retrieval System

This script validates the enhanced retrieval optimizations using real ingested data
from our datasets: EIPs, ergo-python-appkit, ergoscript-by-example, and more.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Load environment variables first
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")

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
from fintelligence_ai.rag.models import Query
from fintelligence_ai.rag.pipeline import RAGPipeline


class EnhancedRetrievalTester:
    """Comprehensive tester for the enhanced retrieval system."""

    def __init__(self):
        self.settings = get_settings()
        self.test_results = []

    def setup_pipelines(self):
        """Initialize all pipeline types for testing."""
        print("🔧 Setting up test pipelines...")

        try:
            # Default pipeline (old behavior)
            self.default_pipeline = create_default_rag_pipeline(
                collection_name=self.settings.chromadb.collection_name
            )

            # Specialized pipelines (new behavior)
            collection_name = self.settings.chromadb.collection_name
            self.eip_pipeline = create_eip_documentation_pipeline()
            self.code_pipeline = create_code_repository_pipeline(collection_name)
            self.ergoscript_pipeline = create_ergoscript_pipeline(collection_name)
            self.flexible_pipeline = create_flexible_documentation_pipeline(
                collection_name
            )

            print("✅ All pipelines initialized successfully")

        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            raise

    def test_eip_queries(self):
        """Test EIP-specific queries against our EIPs dataset."""
        print("\n🔍 Testing EIP Queries")
        print("=" * 50)

        eip_test_cases = [
            {
                "query": "what is the eip for token collections on ergo?",
                "expected_content": ["eip-0034", "collection", "token", "nft"],
                "description": "Original problem query - should find EIP-034",
            },
            {
                "query": "which eip defines nft standards?",
                "expected_content": ["eip-0034", "nft", "standard"],
                "description": "NFT standard query",
            },
            {
                "query": "eip for stealth addresses",
                "expected_content": ["stealth", "address", "eip"],
                "description": "Stealth address EIP query",
            },
            {
                "query": "wallet api improvement proposal",
                "expected_content": ["wallet", "api", "eip"],
                "description": "Wallet API EIP query",
            },
            {
                "query": "explain eip-001",
                "expected_content": ["eip-001", "eip-0001"],
                "description": "Specific EIP number query",
            },
        ]

        for test_case in eip_test_cases:
            self._run_comparison_test(
                test_case, "EIP", self.eip_pipeline, self.default_pipeline
            )

    def test_code_queries(self):
        """Test code-specific queries against our GitHub repositories."""
        print("\n🔍 Testing Code Repository Queries")
        print("=" * 50)

        code_test_cases = [
            {
                "query": "python examples in the appkit",
                "expected_content": ["python", "appkit", "example", ".py"],
                "description": "Python appkit examples query",
            },
            {
                "query": "how to create ErgoBox in python",
                "expected_content": ["ErgoBox", "python", "create", "box"],
                "description": "ErgoBox creation in Python",
            },
            {
                "query": "github repository setup.py",
                "expected_content": ["setup.py", "python", "setup", "install"],
                "description": "Python setup file query",
            },
            {
                "query": "api documentation python",
                "expected_content": ["api", "python", "documentation", "function"],
                "description": "Python API documentation",
            },
            {
                "query": "function implementation examples",
                "expected_content": ["function", "implementation", "example", "def"],
                "description": "Function implementation examples",
            },
        ]

        for test_case in code_test_cases:
            self._run_comparison_test(
                test_case, "Code", self.code_pipeline, self.default_pipeline
            )

    def test_ergoscript_queries(self):
        """Test ErgoScript-specific queries against ergoscript-by-example."""
        print("\n🔍 Testing ErgoScript Development Queries")
        print("=" * 50)

        ergoscript_test_cases = [
            {
                "query": "how to create smart contracts in ergoscript",
                "expected_content": ["ergoscript", "smart", "contract", "box"],
                "description": "ErgoScript smart contract development",
            },
            {
                "query": "what are sigma props in ergoscript",
                "expected_content": ["sigma", "prop", "sigmaprop", "ergoscript"],
                "description": "Sigma propositions explanation",
            },
            {
                "query": "box registers ergoscript",
                "expected_content": ["box", "register", "R0", "R1", "ergoscript"],
                "description": "Box registers in ErgoScript",
            },
            {
                "query": "provelog function usage",
                "expected_content": ["provelog", "function", "ergoscript", "prove"],
                "description": "ProveLog function documentation",
            },
            {
                "query": "utxo handling in ergo",
                "expected_content": ["utxo", "box", "input", "output"],
                "description": "UTXO/Box model handling",
            },
            {
                "query": "stealth address implementation",
                "expected_content": ["stealth", "address", "implementation"],
                "description": "Stealth address implementation (from ergoscript-by-example)",
            },
        ]

        for test_case in ergoscript_test_cases:
            self._run_comparison_test(
                test_case, "ErgoScript", self.ergoscript_pipeline, self.default_pipeline
            )

    def test_general_queries(self):
        """Test general queries to ensure we haven't broken basic functionality."""
        print("\n🔍 Testing General Queries")
        print("=" * 50)

        general_test_cases = [
            {
                "query": "what is ergo blockchain",
                "expected_content": ["ergo", "blockchain", "platform"],
                "description": "General Ergo information",
            },
            {
                "query": "documentation overview",
                "expected_content": ["documentation", "overview", "guide"],
                "description": "General documentation query",
            },
            {
                "query": "tutorial for beginners",
                "expected_content": ["tutorial", "beginner", "guide", "example"],
                "description": "Beginner tutorial query",
            },
        ]

        for test_case in general_test_cases:
            await self._run_comparison_test(
                test_case,
                "General",
                self.default_pipeline,
                self.default_pipeline,
                compare_pipelines=False,
            )

    async def test_flexible_pipeline(self):
        """Test the flexible pipeline's automatic query type detection."""
        print("\n🔍 Testing Flexible Pipeline Auto-Detection")
        print("=" * 50)

        flexible_test_cases = [
            ("what is the eip for token collections on ergo?", "EIP"),
            ("python examples in the appkit", "Code"),
            ("how to create smart contracts in ergoscript", "ErgoScript"),
            ("what is ergo blockchain", "General"),
        ]

        for query_text, expected_type in flexible_test_cases:
            print(f"\n🧪 Testing: '{query_text[:50]}...'")
            print(f"   Expected type: {expected_type}")

            query = Query(text=query_text)

            try:
                start_time = time.time()
                results = self.flexible_pipeline.query(query_text)
                end_time = time.time()

                print(f"   ⏱️  Response time: {(end_time - start_time):.2f}s")
                print(f"   📊 Found {len(results.retrieval_results)} documents")

                if results.retrieval_results:
                    # Show top result
                    top_doc = results.retrieval_results[0]
                    title = getattr(top_doc, "title", "No title")
                    content_preview = (
                        top_doc.content[:100] + "..."
                        if len(top_doc.content) > 100
                        else top_doc.content
                    )
                    score = getattr(top_doc, "score", "No score")

                    print(f"   🏆 Top result: {title}")
                    print(f"   📝 Content: {content_preview}")
                    print(f"   ⭐ Score: {score}")

                    # Check if it contains expected content indicators
                    content_lower = top_doc.content.lower()
                    if expected_type == "EIP" and any(
                        term in content_lower
                        for term in ["eip", "improvement", "standard"]
                    ):
                        print("   ✅ Correctly identified as EIP content")
                    elif expected_type == "Code" and any(
                        term in content_lower
                        for term in ["python", "code", "function", "implementation"]
                    ):
                        print("   ✅ Correctly identified as Code content")
                    elif expected_type == "ErgoScript" and any(
                        term in content_lower
                        for term in ["ergoscript", "sigma", "box", "contract"]
                    ):
                        print("   ✅ Correctly identified as ErgoScript content")
                    else:
                        print(
                            "   ℹ️  General content (expected for non-specific queries)"
                        )

            except Exception as e:
                print(f"   ❌ Query failed: {e}")

    async def _run_comparison_test(
        self,
        test_case: dict[str, Any],
        query_type: str,
        optimized_pipeline: RAGPipeline,
        default_pipeline: RAGPipeline,
        compare_pipelines: bool = True,
    ):
        """Run a comparison test between optimized and default pipelines."""
        query_text = test_case["query"]
        expected_content = test_case["expected_content"]
        description = test_case["description"]

        print(f"\n🧪 {description}")
        print(f"   Query: '{query_text}'")
        print(f"   Expected content: {', '.join(expected_content)}")

        query = Query(text=query_text)

        # Test optimized pipeline
        try:
            start_time = time.time()
            optimized_results = optimized_pipeline.query(query_text)
            optimized_time = time.time() - start_time

            print(f"\n   🚀 Optimized {query_type} Pipeline:")
            print(f"      ⏱️  Response time: {optimized_time:.2f}s")
            print(
                f"      📊 Documents found: {len(optimized_results.retrieval_results)}"
            )

            optimized_relevance = self._calculate_relevance_score(
                optimized_results.retrieval_results, expected_content
            )
            print(f"      ⭐ Relevance score: {optimized_relevance:.2f}")

            if optimized_results.retrieval_results:
                top_doc = optimized_results.retrieval_results[0]
                title = getattr(top_doc, "title", "No title")
                print(f"      🏆 Top result: {title[:80]}...")

        except Exception as e:
            print(f"   ❌ Optimized pipeline failed: {e}")
            optimized_relevance = 0
            optimized_time = 0

        # Test default pipeline for comparison (if requested)
        if compare_pipelines:
            try:
                start_time = time.time()
                default_results = default_pipeline.query(query_text)
                default_time = time.time() - start_time

                print("\n   📊 Default Pipeline (for comparison):")
                print(f"      ⏱️  Response time: {default_time:.2f}s")
                print(
                    f"      📊 Documents found: {len(default_results.retrieval_results)}"
                )

                default_relevance = self._calculate_relevance_score(
                    default_results.retrieval_results, expected_content
                )
                print(f"      ⭐ Relevance score: {default_relevance:.2f}")

                # Show improvement
                relevance_improvement = optimized_relevance - default_relevance
                time_ratio = default_time / optimized_time if optimized_time > 0 else 1

                print("\n   📈 Improvement Analysis:")
                print(f"      🎯 Relevance improvement: +{relevance_improvement:.2f}")
                print(f"      ⚡ Speed ratio: {time_ratio:.2f}x")

                if relevance_improvement > 0:
                    print("      ✅ Enhanced pipeline performs better!")
                elif relevance_improvement == 0:
                    print("      ➡️  Similar performance (expected for some queries)")
                else:
                    print(
                        "      ⚠️  Default pipeline performed better (needs investigation)"
                    )

            except Exception as e:
                print(f"   ❌ Default pipeline failed: {e}")

        # Store results for summary
        self.test_results.append(
            {
                "query_type": query_type,
                "query": query_text,
                "description": description,
                "optimized_relevance": optimized_relevance,
                "optimized_time": optimized_time,
                "default_relevance": getattr(self, "default_relevance", 0),
                "default_time": getattr(self, "default_time", 0),
            }
        )

    def _calculate_relevance_score(
        self, documents: list[Any], expected_content: list[str]
    ) -> float:
        """Calculate a relevance score based on expected content matches."""
        if not documents:
            return 0.0

        total_score = 0.0
        for doc in documents[:5]:  # Check top 5 documents
            content_lower = doc.content.lower()
            title_lower = getattr(doc, "title", "").lower()
            combined_text = content_lower + " " + title_lower

            matches = sum(
                1 for term in expected_content if term.lower() in combined_text
            )
            doc_score = matches / len(expected_content)
            total_score += doc_score

        return total_score / min(len(documents), 5)

    def print_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🎉 ENHANCED RETRIEVAL SYSTEM - TEST SUMMARY")
        print("=" * 60)

        if not self.test_results:
            print("No test results to summarize.")
            return

        # Group by query type
        by_type = {}
        for result in self.test_results:
            query_type = result["query_type"]
            if query_type not in by_type:
                by_type[query_type] = []
            by_type[query_type].append(result)

        overall_improvements = 0
        total_tests = 0

        for query_type, results in by_type.items():
            print(f"\n📊 {query_type} Queries Summary:")
            print("-" * 40)

            avg_relevance = sum(r["optimized_relevance"] for r in results) / len(
                results
            )
            avg_time = sum(r["optimized_time"] for r in results) / len(results)

            print(f"   Tests run: {len(results)}")
            print(f"   Average relevance: {avg_relevance:.2f}")
            print(f"   Average response time: {avg_time:.2f}s")

            improvements = sum(1 for r in results if r["optimized_relevance"] > 0.5)
            print(f"   High-relevance results: {improvements}/{len(results)}")

            overall_improvements += improvements
            total_tests += len(results)

        print("\n🎯 Overall Performance:")
        print(f"   Total tests: {total_tests}")
        print(
            f"   High-relevance results: {overall_improvements}/{total_tests} ({(overall_improvements/total_tests)*100:.1f}%)"
        )

        if overall_improvements / total_tests >= 0.8:
            print("   🎉 Excellent! Enhanced system is working very well.")
        elif overall_improvements / total_tests >= 0.6:
            print("   👍 Good! Enhanced system shows significant improvement.")
        else:
            print("   ⚠️  System needs further optimization.")


async def main():
    """Run the comprehensive end-to-end test suite."""
    print("🚀 Enhanced RAG Retrieval System - End-to-End Testing")
    print("=" * 60)
    print("Testing against real ingested datasets:")
    print("  📄 EIPs repository (23 files)")
    print("  🐍 ergo-python-appkit (6 files)")
    print("  📜 ergoscript-by-example (14 files)")
    print("  📚 15 total repositories")
    print("=" * 60)

    tester = EnhancedRetrievalTester()

    try:
        # Setup
        await tester.setup_pipelines()

        # Run all test suites
        await tester.test_eip_queries()
        await tester.test_code_queries()
        await tester.test_ergoscript_queries()
        await tester.test_general_queries()
        await tester.test_flexible_pipeline()

        # Summary
        tester.print_summary()

        print("\n✅ End-to-end testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
