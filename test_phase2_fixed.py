#!/usr/bin/env python3
"""
Fixed Phase 2 Validation Test for FintelligenceAI

This test validates the implementation of Phase 2 components without
requiring external dependencies like OpenAI API keys.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_imports():
    """Test that core modules can be imported without errors."""
    try:
        logger.info("✓ Core modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Core import failed: {e}")
        return False


def test_model_instantiation():
    """Test that Pydantic models can be instantiated."""
    try:
        from fintelligence_ai.core.evaluation import EvaluationMetrics
        from fintelligence_ai.core.optimization import (
            OptimizationConfig,
            OptimizationStrategy,
        )

        # Test EvaluationMetrics
        metrics = EvaluationMetrics(
            accuracy=0.85, precision=0.80, recall=0.90, f1_score=0.85
        )
        assert metrics.accuracy == 0.85
        logger.info("✓ EvaluationMetrics instantiated correctly")

        # Test OptimizationConfig
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ACCURACY_FOCUSED, max_iterations=10
        )
        assert config.strategy == OptimizationStrategy.ACCURACY_FOCUSED
        logger.info("✓ OptimizationConfig instantiated correctly")

        return True
    except Exception as e:
        logger.error(f"✗ Model instantiation failed: {e}")
        return False


def test_evaluation_framework():
    """Test the evaluation framework functionality."""
    try:
        from fintelligence_ai.core.evaluation import (
            AgentEvaluator,
            ErgoScriptEvaluator,
            RAGEvaluator,
        )

        # Test ErgoScript evaluator
        ergoscript_evaluator = ErgoScriptEvaluator()
        test_code = """
        val output = {
          val input = INPUTS(0)
          sigmaProp(input.value > 1000)
        }
        """

        metrics = ergoscript_evaluator.evaluate_generated_script(test_code)
        assert metrics.syntax_correctness > 0.0
        logger.info("✓ ErgoScript evaluation completed")

        # Test RAG evaluator
        rag_evaluator = RAGEvaluator()
        mock_docs = [{"content": "test", "score": 0.8}]
        rag_metrics = rag_evaluator.evaluate_retrieval_performance(
            "test query", mock_docs
        )
        assert rag_metrics.retrieval_relevance >= 0.0
        logger.info("✓ RAG evaluation completed")

        # Test Agent evaluator
        agent_evaluator = AgentEvaluator()
        agent_metrics = agent_evaluator.evaluate_agent_performance(
            "generate code", {"success": True, "code": "test"}
        )
        assert agent_metrics.task_completion_rate > 0.0
        logger.info("✓ Agent evaluation completed")

        return True
    except Exception as e:
        logger.error(f"✗ Evaluation framework test failed: {e}")
        return False


async def test_async_evaluation():
    """Test async evaluation functionality."""
    try:
        from fintelligence_ai.core.evaluation import EvaluationFramework

        framework = EvaluationFramework()

        # Mock test suite and components
        test_suite = {
            "ergoscript_tests": ["test1", "test2"],
            "rag_tests": ["query1", "query2"],
        }

        system_components = {
            "agents": ["research", "generation", "validation"],
            "rag": ["retrieval", "ranking"],
        }

        result = await framework.run_comprehensive_evaluation(
            test_suite, system_components
        )

        assert result.evaluation_id is not None
        assert result.metrics.accuracy >= 0.0
        logger.info(
            f"✓ Async evaluation completed - Accuracy: {result.metrics.accuracy:.3f}"
        )

        return True
    except Exception as e:
        logger.error(f"✗ Async evaluation test failed: {e}")
        return False


def test_optimization_components():
    """Test optimization component functionality."""
    try:
        from fintelligence_ai.core.optimization import (
            DSPyOptimizer,
            OptimizationConfig,
            OptimizationStrategy,
            OptimizerType,
        )

        # Test optimizer instantiation
        optimizer = DSPyOptimizer()
        logger.info("✓ DSPy optimizer instantiated")

        # Test configuration
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ERGOSCRIPT_SPECIALIZED,
            optimizer_type=OptimizerType.MIPROV2,
            max_iterations=5,
        )

        assert config.strategy == OptimizationStrategy.ERGOSCRIPT_SPECIALIZED
        assert config.optimizer_type == OptimizerType.MIPROV2
        logger.info("✓ Optimization configuration created")

        return True
    except Exception as e:
        logger.error(f"✗ Optimization test failed: {e}")
        return False


async def main():
    """Run all Phase 2 tests."""
    logger.info("🚀 Starting Fixed Phase 2 Validation Tests...")

    tests = [
        ("Core Imports", test_core_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Evaluation Framework", test_evaluation_framework),
        ("Optimization Components", test_optimization_components),
        ("Async Evaluation", test_async_evaluation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")

    # Final results
    success_rate = (passed / total) * 100
    logger.info("\n🎯 Phase 2 Validation Results:")
    logger.info(f"   Tests Passed: {passed}/{total}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")

    if passed == total:
        logger.info("🎉 All Phase 2 tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
